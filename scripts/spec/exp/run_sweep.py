from __future__ import annotations

import dataclasses
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import time
from typing import Any
import urllib.error
import urllib.request

import tyro


DEFAULT_SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
DEFAULT_CLIENT_ACTIVATE = "source examples/libero/.venv/bin/activate"


def _prefer_repo_root(script_path: str | Path, sys_path: list[str] | None = None) -> Path | None:
    path_list = sys.path if sys_path is None else sys_path
    repo_root = Path(script_path).resolve().parents[3]
    if not (repo_root / "scripts" / "spec").is_dir():
        return None

    repo_root = repo_root.resolve()
    path_list[:] = [p for p in path_list if Path(p).resolve() != repo_root]
    path_list.insert(0, str(repo_root))
    return repo_root


_prefer_repo_root(__file__)


@dataclasses.dataclass(frozen=True)
class Args:
    spec_path: str = "scripts/spec/exp/config/main_exp.json"
    dry_run: bool = False
    start_timeout_s: float = 600.0
    health_interval_s: float = 2.0
    server_shutdown_timeout_s: float = 30.0
    continue_on_error: bool = False
    analyze_after: bool = True
    analyze_warmup_episodes: int = 0
    analyze_include_tasks: bool = False
    show_progress: bool = True
    server_python: str = sys.executable
    client_python: str = "python"
    client_activate: str = DEFAULT_CLIENT_ACTIVATE


def _load_spec(spec_path: str | Path) -> dict[str, Any]:
    path = Path(spec_path)
    spec = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(spec, dict):
        raise ValueError(f"sweep spec must be a JSON object, got {type(spec)!r}")
    return spec


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    return [value]


def _expand_grid(grid: dict[str, Any]) -> list[dict[str, Any]]:
    if not grid:
        return [{}]
    keys = list(grid)
    value_lists = [_as_list(grid[key]) for key in keys]
    runs: list[dict[str, Any]] = []

    def _recurse(index: int, current: dict[str, Any]) -> None:
        if index >= len(keys):
            runs.append(dict(current))
            return
        key = keys[index]
        for value in value_lists[index]:
            current[key] = value
            _recurse(index + 1, current)
        current.pop(key, None)

    _recurse(0, {})
    return runs


def _merge_dicts(*items: dict[str, Any] | None) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for item in items:
        if item:
            merged.update(item)
    return merged


def _stable_hash(payload: Any) -> str:
    data = json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha1(data.encode("utf-8")).hexdigest()[:8]


def _format_tag_value(value: Any) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (list, tuple)):
        return "-".join(_format_tag_value(v) for v in value)
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _sanitize_tag(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip())
    value = value.strip("-")
    return value or "run"


def _auto_run_id(params: dict[str, Any], *, index: int) -> str:
    preferred = (
        "tau_radius",
        "t_list",
        "periodic_full_every_n_draft_rounds",
        "enable_gripper_verify",
        "enable_gripper_post_verify",
        "gripper_full_window",
        "full_fallback",
        "force_full_each_round",
    )
    short_names = {
        "tau_radius": "tau",
        "t_list": "t",
        "periodic_full_every_n_draft_rounds": "pf",
        "enable_gripper_verify": "gv",
        "enable_gripper_post_verify": "gpv",
        "gripper_full_window": "gfw",
        "full_fallback": "ff",
        "force_full_each_round": "forcefull",
    }
    parts: list[str] = []
    for key in preferred:
        if key in params:
            parts.append(f"{short_names[key]}{_format_tag_value(params[key])}")
    if not parts:
        parts.append(f"run{index:03d}")
    parts.append(_stable_hash(params))
    return _sanitize_tag("_".join(parts))


def _unique_run_id(base_id: str, used: set[str]) -> str:
    run_id = _sanitize_tag(base_id)
    if run_id not in used:
        used.add(run_id)
        return run_id
    suffix = 2
    while f"{run_id}-{suffix}" in used:
        suffix += 1
    unique = f"{run_id}-{suffix}"
    used.add(unique)
    return unique


def _iter_run_specs(spec: dict[str, Any]) -> list[dict[str, Any]]:
    server_base = dict(spec.get("server_base_args", {}) or {})
    client_base = dict(spec.get("client_base_args", {}) or {})
    suite_server_base = dict(spec.get("suite_server_args", {}) or {})
    default_suites = tuple(spec.get("suites", DEFAULT_SUITES) or DEFAULT_SUITES)
    used_ids: set[str] = set()
    run_specs: list[dict[str, Any]] = []

    explicit_runs = spec.get("runs")
    if explicit_runs is not None:
        if not isinstance(explicit_runs, list):
            raise ValueError("runs must be a list when provided")
        for index, item in enumerate(explicit_runs):
            if not isinstance(item, dict):
                raise ValueError(f"runs[{index}] must be an object")
            server_delta = dict(item.get("server_args", {}) or {})
            client_delta = dict(item.get("client_args", {}) or {})
            suite_server_delta = dict(item.get("suite_server_args", {}) or {})
            params = dict(item.get("params", server_delta) or {})
            base_id = str(item.get("name") or item.get("run_id") or _auto_run_id(params, index=index))
            run_specs.append(
                {
                    "run_id": _unique_run_id(base_id, used_ids),
                    "params": params,
                    "server_args": _merge_dicts(server_base, server_delta),
                    "client_args": _merge_dicts(client_base, client_delta),
                    "suite_server_args": {
                        suite: _merge_dicts(
                            dict(suite_server_base.get(suite, {}) or {}),
                            dict(suite_server_delta.get(suite, {}) or {}),
                        )
                        for suite in set(suite_server_base) | set(suite_server_delta)
                    },
                    "suites": list(item.get("suites", default_suites) or default_suites),
                }
            )
        return run_specs

    grid = dict(spec.get("grid", {}) or {})
    for index, params in enumerate(_expand_grid(grid)):
        run_specs.append(
            {
                "run_id": _unique_run_id(_auto_run_id(params, index=index), used_ids),
                "params": dict(params),
                "server_args": _merge_dicts(server_base, params),
                "client_args": dict(client_base),
                "suite_server_args": suite_server_base,
                "suites": list(default_suites),
            }
        )
    return run_specs


def _cli_key(key: str) -> str:
    return "--" + str(key).replace("_", "-")


def _args_to_cli(args: dict[str, Any]) -> list[str]:
    cli: list[str] = []
    for key in sorted(args):
        value = args[key]
        if value is None:
            continue
        flag = _cli_key(key)
        if isinstance(value, bool):
            cli.append(flag if value else "--no-" + str(key).replace("_", "-"))
        elif isinstance(value, (list, tuple)):
            cli.append(flag)
            cli.extend(str(v) for v in value)
        else:
            cli.extend([flag, str(value)])
    return cli


def _command_text(command: list[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in command)


def _server_command(server_args: dict[str, Any], *, server_python: str) -> list[str]:
    return [server_python, "scripts/spec/spec_serve_policy.py", *_args_to_cli(server_args)]


def _server_args_for_suite(
    run_spec: dict[str, Any],
    *,
    suite: str,
    run_dir: Path | None = None,
) -> dict[str, Any]:
    suite_overrides = dict(run_spec.get("suite_server_args", {}).get(str(suite), {}) or {})
    server_args = _merge_dicts(dict(run_spec["server_args"]), suite_overrides)
    server_args.setdefault("task_suite_name", str(suite))
    if "draft_checkpoint" in server_args and "draft_triton_path" not in server_args and run_dir is not None:
        server_args["draft_triton_path"] = str(run_dir / "suites" / str(suite) / "draft_triton.pkl")
    return server_args


def _client_host_from_server(server_args: dict[str, Any]) -> str:
    host = str(server_args.get("host", "0.0.0.0"))
    return "127.0.0.1" if host in {"0.0.0.0", "::"} else host


def _client_command(
    run_spec: dict[str, Any],
    *,
    suite: str,
    suite_output_root: Path,
    server_args: dict[str, Any],
    client_python: str,
) -> list[str]:
    client_args = dict(run_spec["client_args"])
    client_args.setdefault("host", _client_host_from_server(server_args))
    client_args.setdefault("port", server_args.get("port", 8000))
    client_args["task_suite_name"] = str(suite)
    client_args["video_out_path"] = str(suite_output_root)
    client_args["run_name"] = str(suite)
    return [client_python, "scripts/spec/spec_client_libero.py", *_args_to_cli(client_args)]


def _client_shell_command(client_command: list[str], *, client_activate: str) -> list[str]:
    return ["bash", "-lc", f"{client_activate} && {_command_text(client_command)}"]


def _server_health_url(server_args: dict[str, Any]) -> str:
    host = _client_host_from_server(server_args)
    port = int(server_args.get("port", 8000))
    return f"http://{host}:{port}/healthz"


def _wait_for_server(
    *,
    process: subprocess.Popen,
    health_url: str,
    timeout_s: float,
    interval_s: float,
) -> None:
    deadline = time.monotonic() + float(timeout_s)
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"server exited before health check passed with code {process.returncode}")
        try:
            with urllib.request.urlopen(health_url, timeout=5.0) as response:
                if int(response.status) == 200:
                    return
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = exc
        time.sleep(float(interval_s))
    raise TimeoutError(f"server did not become healthy at {health_url}: {last_error}")


def _terminate_process(process: subprocess.Popen, *, timeout_s: float) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    except Exception:
        process.terminate()
    try:
        process.wait(timeout=float(timeout_s))
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    except Exception:
        process.kill()
    process.wait(timeout=float(timeout_s))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _sweep_dir_for_spec(spec: dict[str, Any]) -> Path:
    name = str(spec.get("name") or dt.datetime.utcnow().strftime("sweep_%Y%m%d_%H%M%S"))
    output_root = Path(str(spec.get("output_root", "data/spec_libero/sweeps")))
    return output_root / _sanitize_tag(name)


def _progress(args: Args, message: str) -> None:
    if not args.show_progress:
        return
    timestamp = dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    print(f"[{timestamp}] {message}", flush=True)


def _elapsed_s(start_time: float) -> str:
    return f"{time.monotonic() - start_time:.1f}s"


def _run_one_suite(
    *,
    run_spec: dict[str, Any],
    suite: str,
    suite_index: int,
    suite_total: int,
    run_dir: Path,
    args: Args,
    client_python: str,
    client_activate: str,
    dry_run: bool,
) -> int:
    run_id = str(run_spec["run_id"])
    suite_output_root = run_dir / "suites"
    suite_dir = suite_output_root / str(suite)
    suite_dir.mkdir(parents=True, exist_ok=True)
    server_args = _server_args_for_suite(run_spec, suite=str(suite), run_dir=run_dir)
    server_command = _server_command(server_args, server_python=args.server_python)
    server_cmd_path = suite_dir / "server_cmd.txt"
    client_cmd_path = suite_dir / "client_cmd.txt"
    server_log_path = suite_dir / "server.log"
    client_log_path = suite_dir / "client.log"
    suite_label = f"run={run_id} suite={suite_index}/{suite_total} {suite}"
    server_cmd_path.write_text(_command_text(server_command) + "\n", encoding="utf-8")
    client_command = _client_command(
        run_spec,
        suite=str(suite),
        suite_output_root=suite_output_root,
        server_args=server_args,
        client_python=client_python,
    )
    shell_command = _client_shell_command(client_command, client_activate=client_activate)
    client_cmd_path.write_text(_command_text(shell_command) + "\n", encoding="utf-8")
    if dry_run:
        _progress(args, f"{suite_label}: dry-run commands written to {suite_dir}")
        return 0
    server_process: subprocess.Popen | None = None
    server_log = None
    suite_start = time.monotonic()
    try:
        _progress(args, f"{suite_label}: starting server (log={server_log_path})")
        server_log = server_log_path.open("w", encoding="utf-8")
        server_process = subprocess.Popen(
            server_command,
            cwd=Path.cwd(),
            stdout=server_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        health_url = _server_health_url(server_args)
        _progress(args, f"{suite_label}: waiting for server health at {health_url}")
        _wait_for_server(
            process=server_process,
            health_url=health_url,
            timeout_s=args.start_timeout_s,
            interval_s=args.health_interval_s,
        )
        _progress(args, f"{suite_label}: server healthy after {_elapsed_s(suite_start)}; starting client")
        client_start = time.monotonic()
        with client_log_path.open("w", encoding="utf-8") as log_file:
            completed = subprocess.run(shell_command, cwd=Path.cwd(), stdout=log_file, stderr=subprocess.STDOUT)
        returncode = int(completed.returncode)
        if returncode == 0:
            _progress(args, f"{suite_label}: client completed in {_elapsed_s(client_start)}")
        else:
            _progress(args, f"{suite_label}: client failed with returncode={returncode} (log={client_log_path})")
        return returncode
    except Exception:
        _progress(args, f"{suite_label}: failed after {_elapsed_s(suite_start)} (server_log={server_log_path})")
        raise
    finally:
        if server_process is not None:
            _progress(args, f"{suite_label}: stopping server")
            _terminate_process(server_process, timeout_s=args.server_shutdown_timeout_s)
        if server_log is not None:
            server_log.close()


def _run_one_param_group(
    *,
    run_spec: dict[str, Any],
    run_index: int,
    run_total: int,
    sweep_dir: Path,
    args: Args,
) -> dict[str, Any]:
    run_dir = sweep_dir / "runs" / str(run_spec["run_id"])
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(run_dir / "run_spec.json", run_spec)

    status: dict[str, Any] = {
        "run_id": run_spec["run_id"],
        "started_at": dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "status": "dry_run" if args.dry_run else "running",
        "suite_status": {},
    }
    _write_json(run_dir / "status.json", status)

    try:
        params_text = json.dumps(run_spec.get("params", {}), ensure_ascii=False, sort_keys=True)
        _progress(args, f"run {run_index}/{run_total} {run_spec['run_id']}: started params={params_text}")
        suites = list(run_spec["suites"])
        for suite_index, suite in enumerate(suites, start=1):
            returncode = _run_one_suite(
                run_spec=run_spec,
                suite=str(suite),
                suite_index=suite_index,
                suite_total=len(suites),
                run_dir=run_dir,
                args=args,
                client_python=args.client_python,
                client_activate=args.client_activate,
                dry_run=args.dry_run,
            )
            status["suite_status"][str(suite)] = {"returncode": int(returncode)}
            _write_json(run_dir / "status.json", status)
            if returncode != 0:
                raise RuntimeError(f"client failed for suite={suite} with returncode={returncode}")

        status["status"] = "dry_run" if args.dry_run else "completed"
        _progress(args, f"run {run_index}/{run_total} {run_spec['run_id']}: {status['status']}")
        return status
    except Exception as exc:
        status["status"] = "failed"
        status["error"] = str(exc)
        _progress(args, f"run {run_index}/{run_total} {run_spec['run_id']}: failed: {exc}")
        if not args.continue_on_error:
            raise
        return status
    finally:
        status["finished_at"] = dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
        _write_json(run_dir / "status.json", status)


def run_sweep(args: Args) -> Path:
    spec = _load_spec(args.spec_path)
    sweep_dir = _sweep_dir_for_spec(spec)
    sweep_dir.mkdir(parents=True, exist_ok=True)
    _write_json(sweep_dir / "sweep_spec.json", spec)

    run_specs = _iter_run_specs(spec)
    _write_json(sweep_dir / "expanded_runs.json", run_specs)

    suite_jobs = sum(len(run_spec.get("suites", [])) for run_spec in run_specs)
    _progress(
        args,
        f"sweep started: dir={sweep_dir} runs={len(run_specs)} suite_jobs={suite_jobs} dry_run={args.dry_run}",
    )

    statuses = []
    for run_index, run_spec in enumerate(run_specs, start=1):
        statuses.append(
            _run_one_param_group(
                run_spec=run_spec,
                run_index=run_index,
                run_total=len(run_specs),
                sweep_dir=sweep_dir,
                args=args,
            )
        )
    _write_json(sweep_dir / "status.json", {"runs": statuses})

    if args.analyze_after and not args.dry_run:
        from scripts.spec.exp import analyze_sweep

        for run_spec in run_specs:
            run_dir = sweep_dir / "runs" / str(run_spec["run_id"])
            _progress(args, f"analyzing run results: dir={run_dir} warmup_episodes={args.analyze_warmup_episodes}")
            analyze_sweep.write_summary(
                sweep_dir=run_dir,
                warmup_episodes=args.analyze_warmup_episodes,
                include_tasks=args.analyze_include_tasks,
            )
        _progress(args, f"analyzing sweep results: dir={sweep_dir} warmup_episodes={args.analyze_warmup_episodes}")
        analyze_sweep.write_summary(
            sweep_dir=sweep_dir,
            warmup_episodes=args.analyze_warmup_episodes,
            include_tasks=args.analyze_include_tasks,
        )
    _progress(args, f"sweep finished: dir={sweep_dir}")
    return sweep_dir


def main(args: Args) -> None:
    sweep_dir = run_sweep(args)
    print(f"sweep_dir={sweep_dir}")


if __name__ == "__main__":
    main(tyro.cli(Args))
