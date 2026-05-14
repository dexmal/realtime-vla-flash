from __future__ import annotations

import csv
import dataclasses
import json
from pathlib import Path
import re
from typing import Any

import tyro


@dataclasses.dataclass(frozen=True)
class Args:
    sweep_dir: str
    warmup_episodes: int = 3
    include_tasks: bool = False


def _mean(values: list[float | None]) -> float | None:
    filtered = [float(v) for v in values if v is not None]
    if not filtered:
        return None
    return sum(filtered) / float(len(filtered))


def _sum_or_none(values: list[Any]) -> int | None:
    filtered = [int(v) for v in values if v is not None]
    if not filtered:
        return None
    return int(sum(filtered))


def _sum_float_or_none(values: list[Any]) -> float | None:
    filtered = [float(v) for v in values if v is not None]
    if not filtered:
        return None
    return float(sum(filtered))


def _safe_ratio(numerator: int | None, denominator: int | None) -> float | None:
    if numerator is None or denominator is None or int(denominator) <= 0:
        return None
    return float(numerator) / float(denominator)


def _route_action_ratio(records: list[dict[str, Any]], route_type: str) -> float | None:
    weighted_count = 0.0
    total_actions = 0
    for record in records:
        executed_action_count = record.get("executed_action_count")
        if executed_action_count is None or int(executed_action_count) <= 0:
            continue
        count = int(executed_action_count)
        route_ratio_by_action = record.get("route_ratio_by_action")
        if not isinstance(route_ratio_by_action, dict):
            return None
        weighted_count += float(route_ratio_by_action.get(route_type, 0.0)) * float(count)
        total_actions += count
    if total_actions <= 0:
        return None
    return weighted_count / float(total_actions)


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _round_metric_values(payload: Any, *, digits: int = 2) -> Any:
    if isinstance(payload, float):
        return round(payload, int(digits))
    if isinstance(payload, list):
        return [_round_metric_values(item, digits=digits) for item in payload]
    if isinstance(payload, dict):
        return {key: _round_metric_values(value, digits=digits) for key, value in payload.items()}
    return payload


def _percent_metric_values(payload: Any, *, digits: int = 1) -> Any:
    if isinstance(payload, float):
        return round(payload * 100.0, int(digits))
    if isinstance(payload, list):
        return [_percent_metric_values(item, digits=digits) for item in payload]
    if isinstance(payload, dict):
        return {key: _percent_metric_values(value, digits=digits) for key, value in payload.items()}
    return payload


def _is_latency_or_time_key(key: str) -> bool:
    return "latency" in key or "policy_time" in key or "serve_time" in key


def _max_exec_steps_from_run(run_summary: dict[str, Any]) -> int:
    for args_key, value_key in (("server_args", "max_exec_steps"), ("client_args", "replan_steps")):
        value = dict(run_summary.get(args_key, {}) or {}).get(value_key)
        if value is not None and int(value) > 0:
            return int(value)
    return 12


def _format_overall_metrics(overall: dict[str, Any], *, max_exec_steps: int) -> dict[str, Any]:
    formatted: dict[str, Any] = {}
    for key, value in overall.items():
        if key == "accepted_action_len_mean":
            accepted_len = _float_or_none(value)
            formatted["accepted_ratio_mean"] = (
                None if accepted_len is None else round(accepted_len / float(max(1, int(max_exec_steps))) * 100.0, 1)
            )
        elif key == "accepted_action_len_mean_by_suite":
            denominator = float(max(1, int(max_exec_steps)))
            if isinstance(value, dict):
                formatted["acc_ratio_by_suite"] = {
                    suite: None
                    if _float_or_none(accepted_len) is None
                    else round(float(_float_or_none(accepted_len)) / denominator * 100.0, 1)
                    for suite, accepted_len in value.items()
                }
            else:
                formatted["acc_ratio_by_suite"] = value
        elif key.startswith("success_rate") or key.endswith("_ratio") or key.endswith("_ratio_by_suite"):
            formatted[key] = _percent_metric_values(value, digits=1)
        elif _is_latency_or_time_key(key):
            formatted[key] = _round_metric_values(value, digits=1)
        else:
            formatted[key] = _round_metric_values(value)
    return formatted


def _summary_metrics(overall: dict[str, Any]) -> dict[str, Any]:
    summary_keys = [
        "success_rate_mean",
        "avg_latency_per_action_ms",
        "infer_latency_mean_ms",
        "effective_infer_latency_mean_ms",
        "accepted_ratio_mean",
        "draft_ratio",
        "draft_infer_ratio",
        "acc_ratio_by_suite",
        "draft_ratio_by_suite",
        "success_rate_by_suite",
        "avg_latency_per_action_ms_by_suite",
        "infer_latency_mean_ms_by_suite",
        "effective_infer_latency_mean_ms_by_suite",
    ]
    return {key: overall.get(key) for key in summary_keys}


def _round_run_metrics(run_summary: dict[str, Any]) -> dict[str, Any]:
    rounded = dict(run_summary)
    rounded["overall"] = _format_overall_metrics(
        dict(rounded.get("overall", {}) or {}),
        max_exec_steps=_max_exec_steps_from_run(rounded),
    )
    rounded["summary"] = _summary_metrics(dict(rounded.get("overall", {}) or {}))
    rounded["suites"] = [_round_metric_values(suite) for suite in rounded.get("suites", [])]
    ordered: dict[str, Any] = {}
    for key, value in rounded.items():
        if key == "overall":
            ordered["summary"] = rounded["summary"]
        if key != "summary":
            ordered[key] = value
    return ordered


def _effective_infer_chunk_count(record: dict[str, Any], episode_log_dir: Path | None) -> int | None:
    infer_path_value = record.get("infer_path")
    if infer_path_value is None or episode_log_dir is None:
        return None
    infer_path = Path(str(infer_path_value))
    if not infer_path.is_absolute():
        infer_path = episode_log_dir / infer_path
    if not infer_path.exists():
        return None

    count = 0
    with infer_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            infer_record = json.loads(line)
            if int(infer_record.get("chunk_exec_len", 0)) > 0:
                count += 1
    return count


def _effective_infer_latency_totals(
    records: list[dict[str, Any]],
    *,
    episode_log_dir: Path | None,
) -> tuple[float | None, int | None]:
    total_latency = 0.0
    effective_chunks = 0
    for record in records:
        infer_latency_sum_ms = _float_or_none(record.get("infer_latency_sum_ms"))
        chunk_count = _effective_infer_chunk_count(record, episode_log_dir)
        if infer_latency_sum_ms is None or chunk_count is None or chunk_count <= 0:
            continue
        total_latency += float(infer_latency_sum_ms)
        effective_chunks += int(chunk_count)
    if effective_chunks <= 0:
        return None, None
    return total_latency, effective_chunks


def _effective_infer_latency_mean_from_totals(total_latency: Any, effective_chunks: Any) -> float | None:
    total = _float_or_none(total_latency)
    if total is None or effective_chunks is None or int(effective_chunks) <= 0:
        return None
    return float(total) / float(effective_chunks)


def _sum_and_count_from_records(
    records: list[dict[str, Any]],
    *,
    sum_key: str,
    count_key: str,
) -> tuple[float | None, int | None]:
    total = 0.0
    count = 0
    for record in records:
        value = _float_or_none(record.get(sum_key))
        count_value = record.get(count_key)
        if value is None or count_value is None or int(count_value) <= 0:
            continue
        total += float(value)
        count += int(count_value)
    if count <= 0:
        return None, None
    return total, count


def _mean_from_sum_and_count(total: Any, count: Any) -> float | None:
    total_float = _float_or_none(total)
    if total_float is None or count is None or int(count) <= 0:
        return None
    return float(total_float) / float(count)


def _task_metric_values(
    *,
    task_records: list[dict[str, Any]],
    latency_records: list[dict[str, Any]],
    episode_log_dir: Path | None,
) -> dict[str, Any]:
    infer_calls = _sum_or_none([record.get("infer_calls") for record in task_records])
    draft_rounds = _sum_or_none([record.get("draft_rounds") for record in task_records])
    vlm_rounds = _sum_or_none([record.get("vlm_rounds") for record in task_records])
    full_rounds = _sum_or_none([record.get("full_rounds") for record in task_records])
    draft_infer_ratio = _safe_ratio(draft_rounds, infer_calls)
    vlm_infer_ratio = _safe_ratio(vlm_rounds, infer_calls)
    full_infer_ratio = _safe_ratio(full_rounds, infer_calls)
    draft_action_ratio = _route_action_ratio(task_records, "draft")
    vlm_action_ratio = _route_action_ratio(task_records, "vlm")
    full_action_ratio = _route_action_ratio(task_records, "full")
    effective_infer_latency_sum_ms, effective_infer_count = _effective_infer_latency_totals(
        latency_records,
        episode_log_dir=episode_log_dir,
    )
    infer_latency_sum_ms, executed_action_count = _sum_and_count_from_records(
        latency_records,
        sum_key="infer_latency_sum_ms",
        count_key="executed_action_count",
    )
    policy_time_sum_ms, policy_action_count = _sum_and_count_from_records(
        latency_records,
        sum_key="policy_time_sum_ms",
        count_key="executed_action_count",
    )
    serve_time_sum_ms, serve_action_count = _sum_and_count_from_records(
        latency_records,
        sum_key="serve_time_sum_ms",
        count_key="executed_action_count",
    )
    return {
        "infer_calls": infer_calls,
        "draft_rounds": draft_rounds,
        "vlm_rounds": vlm_rounds,
        "full_rounds": full_rounds,
        "infer_latency_mean_ms": _mean(
            [_float_or_none(record.get("infer_latency_mean_ms")) for record in latency_records]
        ),
        "effective_infer_latency_mean_ms": _effective_infer_latency_mean_from_totals(
            effective_infer_latency_sum_ms,
            effective_infer_count,
        ),
        "effective_infer_latency_sum_ms": effective_infer_latency_sum_ms,
        "effective_infer_count": effective_infer_count,
        "avg_latency_per_action_ms": _mean_from_sum_and_count(infer_latency_sum_ms, executed_action_count),
        "infer_latency_sum_ms": infer_latency_sum_ms,
        "executed_action_count": executed_action_count,
        "policy_time_mean_ms": _mean(
            [_float_or_none(record.get("policy_time_mean_ms")) for record in latency_records]
        ),
        "serve_time_mean_ms": _mean(
            [_float_or_none(record.get("serve_time_mean_ms")) for record in latency_records]
        ),
        "avg_policy_time_per_action_ms": _mean_from_sum_and_count(policy_time_sum_ms, policy_action_count),
        "policy_time_sum_ms": policy_time_sum_ms,
        "policy_action_count": policy_action_count,
        "avg_serve_time_per_action_ms": _mean_from_sum_and_count(serve_time_sum_ms, serve_action_count),
        "serve_time_sum_ms": serve_time_sum_ms,
        "serve_action_count": serve_action_count,
        "accepted_action_len_mean": _mean(
            [_float_or_none(record.get("accepted_action_len_mean")) for record in task_records]
        ),
        "draft_ratio": draft_action_ratio if draft_action_ratio is not None else draft_infer_ratio,
        "vlm_ratio": vlm_action_ratio if vlm_action_ratio is not None else vlm_infer_ratio,
        "full_ratio": full_action_ratio if full_action_ratio is not None else full_infer_ratio,
        "draft_infer_ratio": draft_infer_ratio,
        "vlm_infer_ratio": vlm_infer_ratio,
        "full_infer_ratio": full_infer_ratio,
    }


def _task_output_record(task: dict[str, Any]) -> dict[str, Any]:
    output = dict(task)
    all_episode_metrics = dict(output.pop("_all_episode_metrics", {}) or {})
    output["all_episode_metrics"] = all_episode_metrics
    output["metric_scope"] = "successful_episodes"
    return output


def _summarize_task_records(
    *,
    task_id: int,
    records: list[dict[str, Any]],
    warmup_episodes: int,
    episode_log_dir: Path | None,
) -> dict[str, Any]:
    task_records = sorted(records, key=lambda record: int(record.get("episode_idx", 0)))
    latency_records = task_records[max(0, int(warmup_episodes)) :]
    successful_task_records = [record for record in task_records if bool(record.get("success", False))]
    successful_latency_records = [record for record in latency_records if bool(record.get("success", False))]
    success_values = [1.0 if bool(record.get("success", False)) else 0.0 for record in task_records]
    success_metrics = _task_metric_values(
        task_records=successful_task_records,
        latency_records=successful_latency_records,
        episode_log_dir=episode_log_dir,
    )
    all_episode_metrics = _task_metric_values(
        task_records=task_records,
        latency_records=latency_records,
        episode_log_dir=episode_log_dir,
    )

    return {
        "task_id": int(task_id),
        "task_description": str(task_records[0].get("task_description", "")) if task_records else "",
        "total_episodes": int(len(task_records)),
        "successful_episodes": int(len(successful_task_records)),
        "latency_episodes_analyzed": int(len(successful_latency_records)),
        "latency_episodes_analyzed_all": int(len(latency_records)),
        "success_rate": (sum(success_values) / float(len(success_values))) if success_values else None,
        **success_metrics,
        "_all_episode_metrics": all_episode_metrics,
    }


def _summarize_episode_records(
    records: list[dict[str, Any]],
    *,
    warmup_episodes: int,
    include_tasks: bool,
    episode_log_dir: Path | None = None,
) -> dict[str, Any]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for record in records:
        if isinstance(record, dict):
            grouped.setdefault(int(record["task_id"]), []).append(record)
    tasks = [
        _summarize_task_records(
            task_id=task_id,
            records=grouped[task_id],
            warmup_episodes=warmup_episodes,
            episode_log_dir=episode_log_dir,
        )
        for task_id in sorted(grouped)
    ]
    summary = {
        "total_tasks": int(len(tasks)),
        "total_episodes": int(sum(int(task.get("total_episodes", 0)) for task in tasks)),
        "success_rate_mean": _mean([_float_or_none(task.get("success_rate")) for task in tasks]),
        "infer_latency_mean_ms": _mean([_float_or_none(task.get("infer_latency_mean_ms")) for task in tasks]),
        "effective_infer_latency_mean_ms": _effective_infer_latency_mean_from_totals(
            _sum_float_or_none([task.get("effective_infer_latency_sum_ms") for task in tasks]),
            _sum_or_none([task.get("effective_infer_count") for task in tasks]),
        ),
        "effective_infer_latency_sum_ms": _sum_float_or_none(
            [task.get("effective_infer_latency_sum_ms") for task in tasks]
        ),
        "effective_infer_count": _sum_or_none([task.get("effective_infer_count") for task in tasks]),
        "avg_latency_per_action_ms": _mean_from_sum_and_count(
            _sum_float_or_none([task.get("infer_latency_sum_ms") for task in tasks]),
            _sum_or_none([task.get("executed_action_count") for task in tasks]),
        ),
        "infer_latency_sum_ms": _sum_float_or_none([task.get("infer_latency_sum_ms") for task in tasks]),
        "executed_action_count": _sum_or_none([task.get("executed_action_count") for task in tasks]),
        "policy_time_mean_ms": _mean([_float_or_none(task.get("policy_time_mean_ms")) for task in tasks]),
        "serve_time_mean_ms": _mean([_float_or_none(task.get("serve_time_mean_ms")) for task in tasks]),
        "avg_policy_time_per_action_ms": _mean_from_sum_and_count(
            _sum_float_or_none([task.get("policy_time_sum_ms") for task in tasks]),
            _sum_or_none([task.get("policy_action_count") for task in tasks]),
        ),
        "policy_time_sum_ms": _sum_float_or_none([task.get("policy_time_sum_ms") for task in tasks]),
        "policy_action_count": _sum_or_none([task.get("policy_action_count") for task in tasks]),
        "avg_serve_time_per_action_ms": _mean_from_sum_and_count(
            _sum_float_or_none([task.get("serve_time_sum_ms") for task in tasks]),
            _sum_or_none([task.get("serve_action_count") for task in tasks]),
        ),
        "serve_time_sum_ms": _sum_float_or_none([task.get("serve_time_sum_ms") for task in tasks]),
        "serve_action_count": _sum_or_none([task.get("serve_action_count") for task in tasks]),
        "accepted_action_len_mean": _mean([_float_or_none(task.get("accepted_action_len_mean")) for task in tasks]),
        "draft_ratio": _mean([_float_or_none(task.get("draft_infer_ratio")) for task in tasks]),
        "vlm_ratio": _mean([_float_or_none(task.get("vlm_infer_ratio")) for task in tasks]),
        "full_ratio": _mean([_float_or_none(task.get("full_infer_ratio")) for task in tasks]),
        "draft_infer_ratio": _mean([_float_or_none(task.get("draft_infer_ratio")) for task in tasks]),
        "vlm_infer_ratio": _mean([_float_or_none(task.get("vlm_infer_ratio")) for task in tasks]),
        "full_infer_ratio": _mean([_float_or_none(task.get("full_infer_ratio")) for task in tasks]),
    }
    successful_records = [record for record in records if isinstance(record, dict) and bool(record.get("success", False))]
    for route_type, key in (("draft", "draft_ratio"), ("vlm", "vlm_ratio"), ("full", "full_ratio")):
        action_ratio = _route_action_ratio(successful_records, route_type)
        if action_ratio is not None:
            summary[key] = action_ratio
    if include_tasks:
        summary["tasks"] = [_task_output_record(task) for task in tasks]
    return summary


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if isinstance(record, dict):
                records.append(record)
    return records


_EPISODE_DIR_RE = re.compile(r"^task(?P<task_id>\d+)_ep(?P<episode_idx>\d+)_(?P<description>.*)_(?P<status>success|failure)$")


def _ratio_dict_from_counts(counts: dict[str, int], *, denom: int) -> dict[str, float]:
    if denom <= 0:
        return {str(key): 0.0 for key in counts}
    return {str(key): float(value) / float(denom) for key, value in sorted(counts.items())}


def _episode_record_from_dir(episode_dir: Path, *, episode_log_dir: Path) -> dict[str, Any] | None:
    match = _EPISODE_DIR_RE.match(episode_dir.name)
    if match is None:
        return None
    infer_path = episode_dir / "infer.jsonl"
    trace_path = episode_dir / "trace.jsonl"
    if not infer_path.exists() or not trace_path.exists():
        return None

    infer_records = _read_jsonl(infer_path)
    trace_records = _read_jsonl(trace_path)
    first_infer = infer_records[0] if infer_records else {}
    task_id = int(match.group("task_id"))
    episode_idx = int(match.group("episode_idx"))
    success = match.group("status") == "success"
    task_description = match.group("description").replace("_", " ")

    infer_calls = int(len(infer_records))
    route_counts: dict[str, int] = {}
    infer_latency_sum_ms = 0.0
    infer_latency_count = 0
    policy_time_sum_ms = 0.0
    policy_time_count = 0
    serve_time_sum_ms = 0.0
    serve_time_count = 0
    accepted_action_sum = 0.0
    accepted_metric_rounds = 0
    for infer_record in infer_records:
        route_type = str(infer_record.get("route_type", "unknown"))
        route_counts[route_type] = route_counts.get(route_type, 0) + 1
        infer_latency = _float_or_none(infer_record.get("sample_actions_ms"))
        if infer_latency is not None:
            infer_latency_sum_ms += float(infer_latency)
            infer_latency_count += 1
        policy_time = _float_or_none(infer_record.get("policy_time_ms"))
        if policy_time is not None:
            policy_time_sum_ms += float(policy_time)
            policy_time_count += 1
        serve_time = _float_or_none(infer_record.get("serve_time_ms"))
        if serve_time is not None:
            serve_time_sum_ms += float(serve_time)
            serve_time_count += 1
        if route_type != "full":
            accepted = _float_or_none(infer_record.get("accepted_prefix_len"))
            if accepted is not None:
                accepted_action_sum += float(accepted)
                accepted_metric_rounds += 1

    executed_trace_records = [record for record in trace_records if record.get("executed_action") is not None]
    action_route_counts: dict[str, int] = {}
    for trace_record in executed_trace_records:
        route_type = str(trace_record.get("route_type", "unknown"))
        action_route_counts[route_type] = action_route_counts.get(route_type, 0) + 1

    executed_action_count = int(len(executed_trace_records))
    return {
        "run_id": str(first_infer.get("run_id", episode_log_dir.name)),
        "task_suite_name": str(first_infer.get("task_suite_name", episode_log_dir.name)),
        "task_id": task_id,
        "task_description": task_description,
        "episode_idx": episode_idx,
        "success": bool(success),
        "failure_reason": None,
        "env_steps_taken": max([int(record.get("env_step", 0)) for record in trace_records], default=0),
        "infer_calls": infer_calls,
        "draft_rounds": int(route_counts.get("draft", 0)),
        "vlm_rounds": int(route_counts.get("vlm", 0)),
        "full_rounds": int(route_counts.get("full", 0)),
        "executed_action_count": executed_action_count,
        "infer_latency_mean_ms": (infer_latency_sum_ms / float(infer_latency_count)) if infer_latency_count > 0 else None,
        "infer_latency_sum_ms": float(infer_latency_sum_ms),
        "policy_time_mean_ms": (policy_time_sum_ms / float(policy_time_count)) if policy_time_count > 0 else None,
        "policy_time_sum_ms": float(policy_time_sum_ms),
        "policy_time_count": int(policy_time_count),
        "serve_time_mean_ms": (serve_time_sum_ms / float(serve_time_count)) if serve_time_count > 0 else None,
        "serve_time_sum_ms": float(serve_time_sum_ms),
        "serve_time_count": int(serve_time_count),
        "avg_latency_per_action_ms": (infer_latency_sum_ms / float(executed_action_count)) if executed_action_count > 0 else None,
        "avg_policy_time_per_action_ms": (policy_time_sum_ms / float(executed_action_count))
        if policy_time_count > 0 and executed_action_count > 0
        else None,
        "avg_serve_time_per_action_ms": (serve_time_sum_ms / float(executed_action_count))
        if serve_time_count > 0 and executed_action_count > 0
        else None,
        "accepted_action_len_mean": (accepted_action_sum / float(accepted_metric_rounds))
        if accepted_metric_rounds > 0
        else None,
        "route_ratio_by_infer": _ratio_dict_from_counts(route_counts, denom=infer_calls),
        "route_ratio_by_action": _ratio_dict_from_counts(action_route_counts, denom=executed_action_count),
        "trace_path": str(trace_path.relative_to(episode_log_dir)),
        "infer_path": str(infer_path.relative_to(episode_log_dir)),
        "trace_record_count": int(len(trace_records)),
        "infer_record_count": int(len(infer_records)),
    }


def _records_from_episode_dirs(episode_log_dir: Path) -> list[dict[str, Any]]:
    episodes_dir = episode_log_dir / "episodes"
    if not episodes_dir.is_dir():
        return []
    episode_dirs_by_key: dict[tuple[int, int], list[Path]] = {}
    for episode_dir in sorted(path for path in episodes_dir.iterdir() if path.is_dir()):
        match = _EPISODE_DIR_RE.match(episode_dir.name)
        if match is None:
            continue
        key = (int(match.group("task_id")), int(match.group("episode_idx")))
        episode_dirs_by_key.setdefault(key, []).append(episode_dir)
    duplicate_dirs = {key: dirs for key, dirs in episode_dirs_by_key.items() if len(dirs) > 1}
    if duplicate_dirs:
        duplicate_text = ", ".join(
            f"task{task_id:02d}_ep{episode_idx:03d}: {[path.name for path in dirs]}"
            for (task_id, episode_idx), dirs in sorted(duplicate_dirs.items())
        )
        raise ValueError(f"duplicate episode directories under {episodes_dir}: {duplicate_text}")
    records_by_key: dict[tuple[int, int], dict[str, Any]] = {}
    for key, episode_dirs in sorted(episode_dirs_by_key.items()):
        record = _episode_record_from_dir(episode_dirs[0], episode_log_dir=episode_log_dir)
        if record is not None:
            records_by_key[key] = record
    return [records_by_key[key] for key in sorted(records_by_key)]


def _records_for_episode_log(path: Path) -> tuple[list[dict[str, Any]], str]:
    records = _read_json(path)
    if not isinstance(records, list):
        raise ValueError(f"expected list at {path}, got {type(records)!r}")
    episode_dir_records = _records_from_episode_dirs(path.parent)
    if len(episode_dir_records) > len(records):
        return episode_dir_records, "episodes"
    return records, "episode_log"


def _suite_episode_log(run_dir: Path, suite: str) -> Path:
    flat_path = run_dir / "suites" / suite / "episode_log.json"
    if flat_path.exists():
        return flat_path
    return run_dir / "suites" / suite / suite / "episode_log.json"


def _is_run_dir(path: Path) -> bool:
    return (path / "run_spec.json").is_file()


def _is_legacy_client_run_dir(path: Path) -> bool:
    return (path / "manifest.json").is_file() and (path / "episode_log.json").is_file()


def _is_supported_run_dir(path: Path) -> bool:
    return _is_run_dir(path) or _is_legacy_client_run_dir(path)


def _run_dirs_for_input(path: Path) -> list[Path]:
    if _is_supported_run_dir(path):
        return [path]
    runs_root = path / "runs"
    if runs_root.is_dir():
        return sorted(run_dir for run_dir in runs_root.glob("*") if run_dir.is_dir() and _is_supported_run_dir(run_dir))
    return sorted(run_dir for run_dir in path.glob("*") if run_dir.is_dir() and _is_supported_run_dir(run_dir))


def _run_spec_for_dir(run_dir: Path) -> dict[str, Any]:
    run_spec_path = run_dir / "run_spec.json"
    if run_spec_path.exists():
        return _read_json(run_spec_path)
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        manifest = _read_json(manifest_path)
        suite = str(manifest.get("task_suite_name", run_dir.parent.name))
        return {
            "run_id": str(manifest.get("run_id", run_dir.name)),
            "params": {},
            "server_args": {},
            "client_args": dict(manifest),
            "suites": [suite],
            "_legacy_episode_log_path": str(run_dir / "episode_log.json"),
        }
    return {"run_id": run_dir.name, "suites": []}


def _episode_log_for_run_suite(run_dir: Path, run_spec: dict[str, Any], suite: str) -> Path:
    legacy_path = run_spec.get("_legacy_episode_log_path")
    if legacy_path is not None:
        return Path(str(legacy_path))
    return _suite_episode_log(run_dir, suite)


def summarize_sweep(
    *,
    sweep_dir: str | Path,
    warmup_episodes: int = 0,
    include_tasks: bool = False,
) -> dict[str, Any]:
    sweep_path = Path(sweep_dir)
    run_summaries: list[dict[str, Any]] = []
    run_dirs = _run_dirs_for_input(sweep_path)
    for run_dir in run_dirs:
        run_spec = _run_spec_for_dir(run_dir)
        suite_summaries: list[dict[str, Any]] = []
        run_successful_records: list[dict[str, Any]] = []
        for suite in run_spec.get("suites", []):
            episode_log_path = _episode_log_for_run_suite(run_dir, run_spec, str(suite))
            if not episode_log_path.exists():
                suite_summaries.append(
                    {
                        "suite": str(suite),
                        "episode_log_path": str(episode_log_path),
                        "missing": True,
                    }
                )
                continue
            records, records_source = _records_for_episode_log(episode_log_path)
            run_successful_records.extend(
                record for record in records if isinstance(record, dict) and bool(record.get("success", False))
            )
            summary = _summarize_episode_records(
                records,
                warmup_episodes=warmup_episodes,
                include_tasks=include_tasks,
                episode_log_dir=episode_log_path.parent,
            )
            summary.update(
                {
                    "suite": str(suite),
                    "episode_log_path": str(episode_log_path),
                    "records_source": records_source,
                    "missing": False,
                }
            )
            suite_summaries.append(summary)

        present_summaries = [summary for summary in suite_summaries if not bool(summary.get("missing", False))]
        overall = {
            "success_rate_mean": _mean([_float_or_none(summary.get("success_rate_mean")) for summary in present_summaries]),
            "infer_latency_mean_ms": _mean(
                [_float_or_none(summary.get("infer_latency_mean_ms")) for summary in present_summaries]
            ),
            "effective_infer_latency_mean_ms": _effective_infer_latency_mean_from_totals(
                _sum_float_or_none([summary.get("effective_infer_latency_sum_ms") for summary in present_summaries]),
                _sum_or_none([summary.get("effective_infer_count") for summary in present_summaries]),
            ),
            "effective_infer_latency_sum_ms": _sum_float_or_none(
                [summary.get("effective_infer_latency_sum_ms") for summary in present_summaries]
            ),
            "effective_infer_count": _sum_or_none([summary.get("effective_infer_count") for summary in present_summaries]),
            "avg_latency_per_action_ms": _mean_from_sum_and_count(
                _sum_float_or_none([summary.get("infer_latency_sum_ms") for summary in present_summaries]),
                _sum_or_none([summary.get("executed_action_count") for summary in present_summaries]),
            ),
            "infer_latency_sum_ms": _sum_float_or_none(
                [summary.get("infer_latency_sum_ms") for summary in present_summaries]
            ),
            "executed_action_count": _sum_or_none(
                [summary.get("executed_action_count") for summary in present_summaries]
            ),
            "policy_time_mean_ms": _mean(
                [_float_or_none(summary.get("policy_time_mean_ms")) for summary in present_summaries]
            ),
            "serve_time_mean_ms": _mean(
                [_float_or_none(summary.get("serve_time_mean_ms")) for summary in present_summaries]
            ),
            "avg_policy_time_per_action_ms": _mean_from_sum_and_count(
                _sum_float_or_none([summary.get("policy_time_sum_ms") for summary in present_summaries]),
                _sum_or_none([summary.get("policy_action_count") for summary in present_summaries]),
            ),
            "policy_time_sum_ms": _sum_float_or_none(
                [summary.get("policy_time_sum_ms") for summary in present_summaries]
            ),
            "policy_action_count": _sum_or_none(
                [summary.get("policy_action_count") for summary in present_summaries]
            ),
            "avg_serve_time_per_action_ms": _mean_from_sum_and_count(
                _sum_float_or_none([summary.get("serve_time_sum_ms") for summary in present_summaries]),
                _sum_or_none([summary.get("serve_action_count") for summary in present_summaries]),
            ),
            "serve_time_sum_ms": _sum_float_or_none([summary.get("serve_time_sum_ms") for summary in present_summaries]),
            "serve_action_count": _sum_or_none([summary.get("serve_action_count") for summary in present_summaries]),
            "accepted_action_len_mean": _mean(
                [_float_or_none(summary.get("accepted_action_len_mean")) for summary in present_summaries]
            ),
            "draft_ratio": _mean([_float_or_none(summary.get("draft_infer_ratio")) for summary in present_summaries]),
            "vlm_ratio": _mean([_float_or_none(summary.get("vlm_infer_ratio")) for summary in present_summaries]),
            "full_ratio": _mean([_float_or_none(summary.get("full_infer_ratio")) for summary in present_summaries]),
            "draft_infer_ratio": _mean(
                [_float_or_none(summary.get("draft_infer_ratio")) for summary in present_summaries]
            ),
            "vlm_infer_ratio": _mean(
                [_float_or_none(summary.get("vlm_infer_ratio")) for summary in present_summaries]
            ),
            "full_infer_ratio": _mean(
                [_float_or_none(summary.get("full_infer_ratio")) for summary in present_summaries]
            ),
            "success_rate_by_suite": {
                str(summary.get("suite")): _float_or_none(summary.get("success_rate_mean"))
                for summary in suite_summaries
                if summary.get("suite") is not None
            },
            "accepted_action_len_mean_by_suite": {
                str(summary.get("suite")): _float_or_none(summary.get("accepted_action_len_mean"))
                for summary in suite_summaries
                if summary.get("suite") is not None
            },
            "draft_ratio_by_suite": {
                str(summary.get("suite")): _float_or_none(summary.get("draft_ratio"))
                for summary in suite_summaries
                if summary.get("suite") is not None
            },
            "avg_latency_per_action_ms_by_suite": {
                str(summary.get("suite")): _float_or_none(summary.get("avg_latency_per_action_ms"))
                for summary in suite_summaries
                if summary.get("suite") is not None
            },
            "infer_latency_mean_ms_by_suite": {
                str(summary.get("suite")): _float_or_none(summary.get("infer_latency_mean_ms"))
                for summary in suite_summaries
                if summary.get("suite") is not None
            },
            "effective_infer_latency_mean_ms_by_suite": {
                str(summary.get("suite")): _float_or_none(summary.get("effective_infer_latency_mean_ms"))
                for summary in suite_summaries
                if summary.get("suite") is not None
            },
        }
        for route_type, key in (("draft", "draft_ratio"), ("vlm", "vlm_ratio"), ("full", "full_ratio")):
            action_ratio = _route_action_ratio(run_successful_records, route_type)
            if action_ratio is not None:
                overall[key] = action_ratio
        run_summaries.append(
            _round_run_metrics(
                {
                    "run_id": str(run_spec.get("run_id", run_dir.name)),
                    "params": dict(run_spec.get("params", {}) or {}),
                    "server_args": dict(run_spec.get("server_args", {}) or {}),
                    "client_args": dict(run_spec.get("client_args", {}) or {}),
                    "overall": overall,
                    "suites": suite_summaries,
                }
            )
        )

    return {
        "sweep_dir": str(sweep_path),
        "analysis_scope": "run" if _is_supported_run_dir(sweep_path) else "sweep",
        "warmup_episodes": int(warmup_episodes),
        "include_tasks": bool(include_tasks),
        "runs": run_summaries,
    }


def _csv_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in summary.get("runs", []):
        common = {
            "run_id": run.get("run_id"),
            "params_json": json.dumps(run.get("params", {}), sort_keys=True, ensure_ascii=False),
        }
        for key, value in dict(run.get("params", {}) or {}).items():
            common[f"param_{key}"] = json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value
        for suite_summary in run.get("suites", []):
            rows.append(
                {
                    **common,
                    "row_type": "suite",
                    "suite": suite_summary.get("suite"),
                    "missing": bool(suite_summary.get("missing", False)),
                    **{key: value for key, value in suite_summary.items() if key not in {"tasks"}},
                }
            )
        rows.append(
            {
                **common,
                "row_type": "overall",
                "suite": "__overall__",
                "missing": False,
                **dict(run.get("overall", {}) or {}),
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, sort_keys=True, ensure_ascii=False)
                    if isinstance(value, (dict, list))
                    else value
                    for key, value in row.items()
                }
            )


def write_summary(
    *,
    sweep_dir: str | Path,
    warmup_episodes: int = 0,
    include_tasks: bool = False,
) -> dict[str, Any]:
    sweep_path = Path(sweep_dir)
    summary = summarize_sweep(
        sweep_dir=sweep_path,
        warmup_episodes=warmup_episodes,
        include_tasks=include_tasks,
    )
    (sweep_path / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_csv(sweep_path / "summary.csv", _csv_rows(summary))
    return summary


def main(args: Args) -> None:
    summary = write_summary(
        sweep_dir=args.sweep_dir,
        warmup_episodes=args.warmup_episodes,
        include_tasks=args.include_tasks,
    )
    print(f"runs={len(summary.get('runs', []))} summary={Path(args.sweep_dir) / 'summary.json'}")


if __name__ == "__main__":
    main(tyro.cli(Args))
