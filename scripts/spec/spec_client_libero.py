import collections
import dataclasses
import datetime as dt
import json
import logging
import math
import pathlib
import re
import time
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 12  # client-side upper bound

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = (
        "libero_goal"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    )
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 50  # Number of rollouts per task
    task: Optional[str] = None  # e.g. "0", "0-9", "0,2,5-7"

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "data/test"  # Base path to save per-run directories
    run_name: Optional[str] = None  # Optional run directory name under video_out_path
    seed: int = 7  # Random Seed (for reproducibility)


def eval_libero(args: Args) -> None:
    np.random.seed(args.seed)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info("Task suite: %s", args.task_suite_name)

    run_dir = _make_run_dir(args.video_out_path, args.run_name)
    run_id = run_dir.name
    episode_log_path = run_dir / "episode_log.json"
    episode_records: List[Dict[str, Any]] = []
    selected_task_ids = _parse_task_spec(args.task, num_tasks_in_suite)
    logging.info(
        "Client will run task id(s): %s (LIBERO '[info] using task orders [...]' is suite reordering, not this list)",
        selected_task_ids,
    )

    _write_json(
        run_dir / "manifest.json",
        {
            "run_id": str(run_id),
            "created_at": dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
            "task_suite_name": str(args.task_suite_name),
            "task_spec": None if args.task is None else str(args.task),
            "selected_task_ids": list(map(int, selected_task_ids)),
            "num_trials_per_task": int(args.num_trials_per_task),
            "replan_steps": int(args.replan_steps),
            "resize_size": int(args.resize_size),
            "video_fps": 10,
            "seed": int(args.seed),
        },
    )

    if args.task_suite_name == "libero_spatial":
        max_steps = 220
    elif args.task_suite_name == "libero_object":
        max_steps = 280
    elif args.task_suite_name == "libero_goal":
        max_steps = 300
    elif args.task_suite_name == "libero_10":
        max_steps = 520
    elif args.task_suite_name == "libero_90":
        max_steps = 400
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(selected_task_ids):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            logging.info("\nTask: %s", task_description)
            env.reset()
            action_plan = collections.deque()

            obs = env.set_init_state(initial_states[episode_idx])

            t = 0
            replay_images = []
            done = False
            episode_error: Optional[str] = None
            infer_calls = 0
            infer_latency_sum_ms = 0.0
            policy_time_sum_ms = 0.0
            policy_time_count = 0
            serve_time_sum_ms = 0.0
            serve_time_count = 0
            accepted_action_sum = 0.0
            draft_rounds = 0
            vlm_rounds = 0
            full_rounds = 0
            infer_id_next = 0
            episode_trace: List[Dict[str, Any]] = []
            episode_infers: List[Dict[str, Any]] = []

            executed_steps = 0
            need_reset = True

            logging.info("Starting episode %s...", task_episodes + 1)
            while t < max_steps + args.num_steps_wait:
                try:
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    # IMPORTANT: rotate 180 degrees to match train preprocessing
                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, args.resize_size, args.resize_size))
                    wrist_img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                    )
                    replay_images.append(img)
                    frame_idx = int(len(replay_images) - 1)

                    if not action_plan:
                        infer_id = int(infer_id_next)
                        infer_id_next += 1
                        element = {
                            "observation/image": img,
                            "observation/wrist_image": wrist_img,
                            "observation/state": np.concatenate(
                                (
                                    obs["robot0_eef_pos"],
                                    _quat2axisangle(obs["robot0_eef_quat"]),
                                    obs["robot0_gripper_qpos"],
                                )
                            ),
                            "prompt": str(task_description),
                            "__executed_steps__": int(executed_steps),
                            "__trace_run_id__": str(run_id),
                            "__trace_task_id__": int(task_id),
                            "__trace_episode_idx__": int(episode_idx),
                            "__trace_infer_id__": int(infer_id),
                        }
                        if need_reset:
                            element["__reset_policy_state__"] = True
                            need_reset = False

                        client_send_timestamp_s = time.time()
                        client_t0 = time.perf_counter()
                        out = client.infer(element)
                        client_roundtrip_ms = (time.perf_counter() - client_t0) * 1000.0
                        client_recv_timestamp_s = time.time()
                        action_chunk = out["actions"]
                        policy_timing = out.get("policy_timing", {})
                        server_timing = out.get("server_timing", {})
                        if not isinstance(server_timing, dict):
                            server_timing = {}
                        infer_calls += 1
                        infer_latency_ms = policy_timing.get("sample_actions_ms", None)
                        if infer_latency_ms is not None:
                            infer_latency_sum_ms += float(infer_latency_ms)
                        policy_time_ms, serve_time_ms = _policy_and_serve_time_ms(server_timing)
                        if policy_time_ms is not None:
                            policy_time_sum_ms += float(policy_time_ms)
                            policy_time_count += 1
                        if serve_time_ms is not None:
                            serve_time_sum_ms += float(serve_time_ms)
                            serve_time_count += 1
                        route_type = _route_type_from_timing(policy_timing)
                        if route_type == "full":
                            full_rounds += 1
                        else:
                            draft_rounds += 1

                        accepted = int(out.get("accepted_prefix_len", args.replan_steps))
                        if bool(round(float(policy_timing.get("include_in_draft_accept_metrics", 1.0)))):
                            accepted_action_sum += float(accepted)
                        exec_len = int(min(int(args.replan_steps), int(accepted))) if accepted > 0 else 0
                        chunk_actions = np.asarray(action_chunk, dtype=np.float32)
                        infer_record = _make_infer_record(
                            run_id=run_id,
                            task_suite_name=args.task_suite_name,
                            task_id=task_id,
                            episode_idx=episode_idx,
                            infer_id=infer_id,
                            frame_idx=frame_idx,
                            env_step=t,
                            route_type=route_type,
                            accepted_prefix_len=accepted,
                            chunk_exec_len=exec_len,
                            policy_timing=policy_timing,
                            server_timing=server_timing,
                            client_send_timestamp_s=client_send_timestamp_s,
                            client_recv_timestamp_s=client_recv_timestamp_s,
                            client_roundtrip_ms=client_roundtrip_ms,
                            chunk_actions=chunk_actions,
                        )
                        episode_infers.append(infer_record)
                        if exec_len <= 0:
                            episode_trace.append(
                                _make_trace_record(
                                    run_id=run_id,
                                    task_suite_name=args.task_suite_name,
                                    task_id=task_id,
                                    episode_idx=episode_idx,
                                    frame_idx=frame_idx,
                                    env_step=t,
                                    infer_record=infer_record,
                                    action=None,
                                    action_offset_in_chunk=None,
                                    infer_start_frame=True,
                                    reward=None,
                                    done_after_step=False,
                                )
                            )
                            executed_steps = 0
                            continue
                        for action_offset in range(exec_len):
                            action_plan.append(
                                {
                                    "action": chunk_actions[action_offset].astype(np.float32, copy=True),
                                    "infer_record": infer_record,
                                    "action_offset_in_chunk": int(action_offset),
                                    "infer_start_frame": bool(action_offset == 0),
                                }
                            )

                        executed_steps = exec_len

                    action_meta = action_plan.popleft()
                    action = np.asarray(action_meta["action"], dtype=np.float32)
                    obs, reward, done, info = env.step(action.tolist())
                    episode_trace.append(
                        _make_trace_record(
                            run_id=run_id,
                            task_suite_name=args.task_suite_name,
                            task_id=task_id,
                            episode_idx=episode_idx,
                            frame_idx=frame_idx,
                            env_step=t,
                            infer_record=action_meta["infer_record"],
                            action=action,
                            action_offset_in_chunk=int(action_meta["action_offset_in_chunk"]),
                            infer_start_frame=bool(action_meta["infer_start_frame"]),
                            reward=float(reward),
                            done_after_step=bool(done),
                        )
                    )
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception as e:
                    logging.error("Caught exception: %s", e)
                    episode_error = str(e)
                    break

            task_episodes += 1
            total_episodes += 1

            suffix = "success" if done else "failure"
            episode_dir = _episode_output_dir(
                run_dir=run_dir,
                task_description=task_description,
                suffix=suffix,
                task_id=task_id,
                episode_idx=episode_idx,
            )
            video_path = episode_dir / f"rollout_{suffix}.mp4"
            imageio.mimwrite(
                video_path,
                [np.asarray(x) for x in replay_images],
                fps=10,
            )
            trace_path = episode_dir / "trace.jsonl"
            infer_path = episode_dir / "infer.jsonl"
            _write_jsonl(trace_path, episode_trace)
            _write_jsonl(infer_path, episode_infers)
            mean_infer_latency_ms = (infer_latency_sum_ms / float(infer_calls)) if infer_calls > 0 else None
            mean_policy_time_ms = (
                policy_time_sum_ms / float(policy_time_count)
            ) if policy_time_count > 0 else None
            mean_serve_time_ms = (
                serve_time_sum_ms / float(serve_time_count)
            ) if serve_time_count > 0 else None
            accepted_metric_rounds = draft_rounds
            mean_accepted_action_len = (
                accepted_action_sum / float(accepted_metric_rounds)
            ) if accepted_metric_rounds > 0 else None
            infer_route_counts = _count_by_key(episode_infers, "route_type")
            executed_action_count = int(sum(1 for r in episode_trace if r.get("executed_action") is not None))
            action_route_counts = _count_by_key(
                [r for r in episode_trace if r.get("executed_action") is not None],
                "route_type",
            )
            episode_records.append(
                {
                    "run_id": str(run_id),
                    "task_suite_name": args.task_suite_name,
                    "task_id": int(task_id),
                    "task_description": str(task_description),
                    "episode_idx": int(episode_idx),
                    "success": bool(done),
                    "failure_reason": episode_error,
                    "max_steps": int(max_steps),
                    "num_steps_wait": int(args.num_steps_wait),
                    "env_steps_taken": int(t),
                    "infer_calls": int(infer_calls),
                    "draft_rounds": int(draft_rounds),
                    "vlm_rounds": int(vlm_rounds),
                    "full_rounds": int(full_rounds),
                    "executed_action_count": int(executed_action_count),
                    "infer_latency_mean_ms": None if mean_infer_latency_ms is None else float(mean_infer_latency_ms),
                    "infer_latency_sum_ms": float(infer_latency_sum_ms),
                    "policy_time_mean_ms": None if mean_policy_time_ms is None else float(mean_policy_time_ms),
                    "policy_time_sum_ms": float(policy_time_sum_ms),
                    "policy_time_count": int(policy_time_count),
                    "serve_time_mean_ms": None if mean_serve_time_ms is None else float(mean_serve_time_ms),
                    "serve_time_sum_ms": float(serve_time_sum_ms),
                    "serve_time_count": int(serve_time_count),
                    "avg_latency_per_action_ms": (
                        float(infer_latency_sum_ms) / float(executed_action_count) if executed_action_count > 0 else None
                    ),
                    "avg_policy_time_per_action_ms": (
                        float(policy_time_sum_ms) / float(executed_action_count)
                        if policy_time_count > 0 and executed_action_count > 0
                        else None
                    ),
                    "avg_serve_time_per_action_ms": (
                        float(serve_time_sum_ms) / float(executed_action_count)
                        if serve_time_count > 0 and executed_action_count > 0
                        else None
                    ),
                    "accepted_action_len_mean": None if mean_accepted_action_len is None else float(mean_accepted_action_len),
                    "route_ratio_by_infer": _ratio_dict(infer_route_counts, denom=int(len(episode_infers))),
                    "route_ratio_by_action": _ratio_dict(action_route_counts, denom=int(executed_action_count)),
                    "video_path": str(video_path.relative_to(run_dir)),
                    "trace_path": str(trace_path.relative_to(run_dir)),
                    "infer_path": str(infer_path.relative_to(run_dir)),
                    "frame_count": int(len(replay_images)),
                    "trace_record_count": int(len(episode_trace)),
                    "infer_record_count": int(len(episode_infers)),
                    "task_spec": None if args.task is None else str(args.task),
                    "seed": int(args.seed),
                }
            )
            _write_episode_log(episode_log_path, episode_records)

            logging.info("Success: %s", done)
            logging.info("# episodes completed so far: %s", total_episodes)
            logging.info("# successes: %s (%.1f%%)", total_successes, total_successes / total_episodes * 100.0)

        logging.info("Current task success rate: %.3f", float(task_successes) / float(task_episodes))
        logging.info("Current total success rate: %.3f", float(total_successes) / float(total_episodes))

    logging.info("Total success rate: %.3f", float(total_successes) / float(total_episodes))
    logging.info("Total episodes: %s", total_episodes)


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description


def _parse_task_spec(spec: Optional[str], num_tasks: int) -> List[int]:
    if spec is None or str(spec).strip() == "":
        return list(range(int(num_tasks)))
    task_ids: Set[int] = set()
    for part in str(spec).split(","):
        token = part.strip()
        if not token:
            continue
        match = re.fullmatch(r"(\d+)\s*-\s*(\d+)", token)
        if match:
            lo = int(match.group(1))
            hi = int(match.group(2))
            if lo > hi:
                raise ValueError(f"invalid task range {token!r}: start > end")
            for task_id in range(lo, hi + 1):
                if task_id < 0 or task_id >= int(num_tasks):
                    raise ValueError(f"task id {task_id} out of range [0, {int(num_tasks) - 1}]")
                task_ids.add(int(task_id))
            continue
        if not token.isdigit():
            raise ValueError(f"unsupported task selector token {token!r}")
        task_id = int(token)
        if task_id < 0 or task_id >= int(num_tasks):
            raise ValueError(f"task id {task_id} out of range [0, {int(num_tasks) - 1}]")
        task_ids.add(int(task_id))
    return sorted(task_ids)


def _make_run_dir(video_out_path: str, run_name: Optional[str]) -> pathlib.Path:
    base_dir = pathlib.Path(video_out_path)
    base_dir.mkdir(parents=True, exist_ok=True)
    if run_name is None or str(run_name).strip() == "":
        run_name = dt.datetime.utcnow().strftime("run_%Y%m%d_%H%M%S")
    run_dir = base_dir / str(run_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _episode_output_dir(
    *,
    run_dir: pathlib.Path,
    task_description: str,
    suffix: str,
    task_id: int,
    episode_idx: int,
) -> pathlib.Path:
    task_segment = task_description.replace(" ", "_")
    path = run_dir / "episodes" / f"task{task_id:02d}_ep{episode_idx:03d}_{task_segment}_{suffix}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_episode_log(path: pathlib.Path, records: List[Dict[str, Any]]) -> None:
    _write_json(path, records)


def _write_json(path: pathlib.Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_jsonl(path: pathlib.Path, records: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _route_type_from_timing(policy_timing: Dict[str, Any]) -> str:
    is_full_round = bool(round(float(policy_timing.get("is_full_pipeline_round", 0.0))))
    if is_full_round:
        return "full"
    return "draft"


def _float_or_none(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _policy_and_serve_time_ms(server_timing: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    policy_time_ms = _float_or_none(server_timing.get("policy_time_ms"))
    if policy_time_ms is None:
        policy_time_ms = _float_or_none(server_timing.get("infer_ms"))

    serve_time_ms = _float_or_none(server_timing.get("serve_time_ms"))
    if serve_time_ms is None and policy_time_ms is not None:
        ws_unpack_ms = _float_or_none(server_timing.get("ws_unpack_ms"))
        ws_pack_ms = _float_or_none(server_timing.get("ws_pack_ms"))
        if ws_unpack_ms is not None and ws_pack_ms is not None:
            serve_time_ms = float(ws_unpack_ms) + float(policy_time_ms) + float(ws_pack_ms)
    return policy_time_ms, serve_time_ms


def _make_infer_record(
    *,
    run_id: str,
    task_suite_name: str,
    task_id: int,
    episode_idx: int,
    infer_id: int,
    frame_idx: int,
    env_step: int,
    route_type: str,
    accepted_prefix_len: int,
    chunk_exec_len: int,
    policy_timing: Dict[str, Any],
    server_timing: Dict[str, Any],
    client_send_timestamp_s: float,
    client_recv_timestamp_s: float,
    client_roundtrip_ms: float,
    chunk_actions: np.ndarray,
) -> Dict[str, Any]:
    policy_time_ms, serve_time_ms = _policy_and_serve_time_ms(server_timing)
    return {
        "timestamp": float(client_recv_timestamp_s),
        "client_send_timestamp_s": float(client_send_timestamp_s),
        "client_recv_timestamp_s": float(client_recv_timestamp_s),
        "client_roundtrip_ms": float(client_roundtrip_ms),
        "server_recv_timestamp_s": _float_or_none(server_timing.get("server_recv_timestamp_s")),
        "server_response_timestamp_s": _float_or_none(server_timing.get("server_response_timestamp_s")),
        "run_id": str(run_id),
        "task_suite_name": str(task_suite_name),
        "task_id": int(task_id),
        "episode_idx": int(episode_idx),
        "infer_id": int(infer_id),
        "frame_idx": int(frame_idx),
        "env_step": int(env_step),
        "route_type": str(route_type),
        "accepted_prefix_len": int(accepted_prefix_len),
        "chunk_exec_len": int(chunk_exec_len),
        "chunk_actions": np.asarray(chunk_actions, dtype=np.float32).tolist(),
        "sample_actions_ms": _float_or_none(policy_timing.get("sample_actions_ms")),
        "policy_time_ms": policy_time_ms,
        "serve_time_ms": serve_time_ms,
        "ws_unpack_ms": _float_or_none(server_timing.get("ws_unpack_ms")),
        "ws_pack_ms": _float_or_none(server_timing.get("ws_pack_ms")),
        "encoder_ms": _float_or_none(policy_timing.get("encoder_ms")),
        "vlm_prefill_ms": _float_or_none(policy_timing.get("vlm_prefill_ms")),
        "draft_ms": _float_or_none(policy_timing.get("draft_ms")),
        "action_verify_ms": _float_or_none(policy_timing.get("action_verify_ms")),
        "full_fallback_ms": _float_or_none(policy_timing.get("full_fallback_ms")),
        "total_ms": _float_or_none(policy_timing.get("total_ms")),
        "accepted_prefix_len_mean": _float_or_none(policy_timing.get("accepted_prefix_len_mean")),
        "radius_dist_mean": _float_or_none(policy_timing.get("radius_dist_mean")),
        "gripper_override_rate": _float_or_none(policy_timing.get("gripper_override_rate")),
        "did_prefill": int(round(float(policy_timing.get("did_prefill", 0.0)))),
        "is_full_pipeline_round": int(round(float(policy_timing.get("is_full_pipeline_round", 0.0)))),
        "used_full_fallback": int(round(float(policy_timing.get("used_full_fallback", 0.0)))),
        "scheduled_full_fallback": int(round(float(policy_timing.get("scheduled_full_fallback", 0.0)))),
        "verify_mode_random": int(round(float(policy_timing.get("verify_mode_random", 0.0)))),
        "gripper_verify_enabled": int(round(float(policy_timing.get("gripper_verify_enabled", 0.0)))),
    }


def _make_trace_record(
    *,
    run_id: str,
    task_suite_name: str,
    task_id: int,
    episode_idx: int,
    frame_idx: int,
    env_step: int,
    infer_record: Dict[str, Any],
    action: Optional[np.ndarray],
    action_offset_in_chunk: Optional[int],
    infer_start_frame: bool,
    reward: Optional[float],
    done_after_step: bool,
) -> Dict[str, Any]:
    return {
        "run_id": str(run_id),
        "task_suite_name": str(task_suite_name),
        "task_id": int(task_id),
        "episode_idx": int(episode_idx),
        "frame_idx": int(frame_idx),
        "env_step": int(env_step),
        "infer_id": int(infer_record["infer_id"]),
        "infer_start_frame": bool(infer_start_frame),
        "action_offset_in_chunk": None if action_offset_in_chunk is None else int(action_offset_in_chunk),
        "executed_action": None if action is None else np.asarray(action, dtype=np.float32).tolist(),
        "route_type": str(infer_record["route_type"]),
        "accepted_prefix_len": int(infer_record["accepted_prefix_len"]),
        "chunk_exec_len": int(infer_record["chunk_exec_len"]),
        "sample_actions_ms": infer_record["sample_actions_ms"],
        "policy_time_ms": infer_record.get("policy_time_ms"),
        "serve_time_ms": infer_record.get("serve_time_ms"),
        "client_roundtrip_ms": infer_record.get("client_roundtrip_ms"),
        "total_ms": infer_record["total_ms"],
        "radius_dist_mean": infer_record["radius_dist_mean"],
        "did_prefill": int(infer_record["did_prefill"]),
        "is_full_pipeline_round": int(infer_record["is_full_pipeline_round"]),
        "used_full_fallback": int(infer_record["used_full_fallback"]),
        "scheduled_full_fallback": int(infer_record["scheduled_full_fallback"]),
        "reward": reward,
        "done_after_step": bool(done_after_step),
    }


def _count_by_key(records: Iterable[Dict[str, Any]], key: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for record in records:
        value = str(record.get(key, "unknown"))
        counts[value] = counts.get(value, 0) + 1
    return counts


def _ratio_dict(counts: Dict[str, int], *, denom: int) -> Dict[str, float]:
    if denom <= 0:
        return {str(k): 0.0 for k in counts}
    return {str(k): float(v) / float(denom) for k, v in sorted(counts.items())}


def _quat2axisangle(quat):
    """
    Copied from robosuite:
    https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    eval_libero(tyro.cli(Args))
