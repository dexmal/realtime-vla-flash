from __future__ import annotations

import argparse
import csv
import dataclasses
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DTYPE_BYTES = 2
BINARY_MB = 2**20


def _apply_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 11.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 1.1,
            "axes.labelsize": 12.0,
            "axes.titlesize": 13.0,
            "legend.frameon": False,
            "legend.fontsize": 10.0,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


@dataclasses.dataclass(frozen=True)
class ModelSpec:
    hidden_size: int
    intermediate_size: int
    num_ffi: int
    num_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int = 0


@dataclasses.dataclass(frozen=True)
class GenZWorkload:
    flops: float
    bytes_: float

    @property
    def oi_flop_per_byte(self) -> float:
        return self.flops / self.bytes_

    @property
    def vla_perf_oi(self) -> float:
        # VLA-perf evaluates OI as Num ops (MFLOP) / Total Data (MB), where
        # MFLOP is decimal and MB is binary.
        return (self.flops / 1e6) / (self.bytes_ / BINARY_MB)


@dataclasses.dataclass(frozen=True)
class Stage:
    name: str
    label: str
    measured_ms: float
    workload: GenZWorkload
    reported_latency_multiplier: int = 1
    published_vla_perf_oi: float | None = None
    note: str = ""


@dataclasses.dataclass(frozen=True)
class StagePoint:
    stage: Stage
    oi_flop_per_byte: float
    vla_perf_oi: float
    static_flops: float
    static_bytes: float
    executed_flops: float
    executed_bytes: float
    roofline_tflops_at_oi: float
    roofline_ms: float
    measured_tflops: float
    executed_tflops: float
    vla_perf_equiv_bandwidth_gbps: float
    executed_bandwidth_gbps: float
    bound: str


def _sum_workloads(*items: GenZWorkload) -> GenZWorkload:
    return GenZWorkload(sum(item.flops for item in items), sum(item.bytes_ for item in items))


def _scale_workload(workload: GenZWorkload, multiplier: int) -> GenZWorkload:
    return GenZWorkload(workload.flops * multiplier, workload.bytes_ * multiplier)


def _gemm(batch: int, m: int, n: int, k: int) -> GenZWorkload:
    # GenZ GEMM dimensions are [B, M, N, K]. get_num_ops() returns MACs;
    # get_roofline() multiplies by 2 to convert MACs to FLOPs.
    macs = batch * m * n * k
    elements = batch * k * n + m * k + batch * m * n
    return GenZWorkload(flops=2 * macs, bytes_=DTYPE_BYTES * elements)


def _logit(batch: int, heads: int, m: int, n: int, head_dim: int, kv_heads: int) -> GenZWorkload:
    macs = batch * heads * m * n * head_dim
    elements = batch * heads * m * head_dim + batch * kv_heads * n * head_dim + batch * heads * m * n
    return GenZWorkload(flops=2 * macs, bytes_=DTYPE_BYTES * elements)


def _attend(batch: int, heads: int, m: int, n: int, head_dim: int, kv_heads: int) -> GenZWorkload:
    macs = batch * heads * m * n * head_dim
    elements = batch * heads * m * n + batch * kv_heads * n * head_dim + batch * heads * m * head_dim
    return GenZWorkload(flops=2 * macs, bytes_=DTYPE_BYTES * elements)


def _prefill_workload(spec: ModelSpec, *, sequence_length: int, batch_size: int) -> GenZWorkload:
    layer = _sum_workloads(
        _gemm(
            batch_size,
            spec.num_attention_heads * spec.head_dim + 2 * spec.num_key_value_heads * spec.head_dim,
            sequence_length,
            spec.hidden_size,
        ),
        _logit(
            batch_size,
            spec.num_attention_heads,
            sequence_length,
            sequence_length,
            spec.head_dim,
            spec.num_key_value_heads,
        ),
        _attend(
            batch_size,
            spec.num_attention_heads,
            sequence_length,
            sequence_length,
            spec.head_dim,
            spec.num_key_value_heads,
        ),
        _gemm(batch_size, spec.hidden_size, sequence_length, spec.num_attention_heads * spec.head_dim),
        _gemm(batch_size, spec.intermediate_size * spec.num_ffi, sequence_length, spec.hidden_size),
        _gemm(batch_size, spec.hidden_size, sequence_length, spec.intermediate_size),
    )
    embedding = _gemm(batch_size, spec.hidden_size, sequence_length, spec.vocab_size)
    return _sum_workloads(embedding, _scale_workload(layer, spec.num_layers))


def _parallel_decode_workload(
    spec: ModelSpec,
    *,
    context_length: int,
    output_tokens_parallel: int,
    batch_size: int = 1,
    self_attention: bool = True,
) -> GenZWorkload:
    attn_projection = _gemm(
        batch_size,
        spec.num_attention_heads * spec.head_dim + 2 * spec.num_key_value_heads * spec.head_dim,
        output_tokens_parallel,
        spec.hidden_size,
    )
    ops = [
        attn_projection,
        _logit(
            batch_size,
            spec.num_attention_heads,
            output_tokens_parallel,
            context_length,
            spec.head_dim,
            spec.num_key_value_heads,
        ),
        _attend(
            batch_size,
            spec.num_attention_heads,
            output_tokens_parallel,
            context_length,
            spec.head_dim,
            spec.num_key_value_heads,
        ),
    ]
    if self_attention and output_tokens_parallel > 1:
        ops.extend(
            [
                attn_projection,
                _logit(
                    batch_size,
                    spec.num_attention_heads,
                    output_tokens_parallel,
                    output_tokens_parallel,
                    spec.head_dim,
                    spec.num_key_value_heads,
                ),
                _attend(
                    batch_size,
                    spec.num_attention_heads,
                    output_tokens_parallel,
                    output_tokens_parallel,
                    spec.head_dim,
                    spec.num_key_value_heads,
                ),
            ]
        )
    ops.extend(
        [
            _gemm(batch_size, spec.hidden_size, output_tokens_parallel, spec.num_attention_heads * spec.head_dim),
            _gemm(batch_size, spec.intermediate_size * spec.num_ffi, output_tokens_parallel, spec.hidden_size),
            _gemm(batch_size, spec.hidden_size, output_tokens_parallel, spec.intermediate_size),
        ]
    )
    return _scale_workload(_sum_workloads(*ops), spec.num_layers)


PI0_VISION_TOKENS = 256
PI0_VISION_FRAMES = 3
PI0_LANGUAGE_TOKENS = 32
PI0_VLM_SEQUENCE_LENGTH = PI0_VISION_TOKENS * PI0_VISION_FRAMES + PI0_LANGUAGE_TOKENS
PI0_ACTION_CHUNK_SIZE = 50
PI0_DENOISING_STEPS = 10

PI0_VISION = ModelSpec(
    hidden_size=1152,
    intermediate_size=4304,
    num_ffi=1,
    num_layers=27,
    num_attention_heads=16,
    num_key_value_heads=16,
    head_dim=72,
)
PI0_VLM = ModelSpec(
    hidden_size=2048,
    intermediate_size=16384,
    num_ffi=2,
    num_layers=18,
    num_attention_heads=8,
    num_key_value_heads=1,
    head_dim=256,
)
PI0_ACTION_EXPERT = ModelSpec(
    hidden_size=1024,
    intermediate_size=4096,
    num_ffi=2,
    num_layers=18,
    num_attention_heads=8,
    num_key_value_heads=1,
    head_dim=256,
)

STAGES = (
    Stage(
        name="encoder",
        label="Encoder / Vision",
        measured_ms=11.0,
        workload=_prefill_workload(PI0_VISION, sequence_length=PI0_VISION_TOKENS, batch_size=PI0_VISION_FRAMES),
        published_vla_perf_oi=321.4,
        note="GenZ full prefill for pi0-vision with batch_size_multiplier=3.",
    ),
    Stage(
        name="vlm",
        label="VLM Prefill",
        measured_ms=25.0,
        workload=_prefill_workload(PI0_VLM, sequence_length=PI0_VLM_SEQUENCE_LENGTH, batch_size=1),
        published_vla_perf_oi=542.8,
        note="GenZ full prefill for pi0-vlm with 768 vision tokens + 32 language tokens.",
    ),
    Stage(
        name="ae",
        label="Action Expert",
        measured_ms=20.0,
        workload=_parallel_decode_workload(
            PI0_ACTION_EXPERT,
            context_length=PI0_VLM_SEQUENCE_LENGTH,
            output_tokens_parallel=PI0_ACTION_CHUNK_SIZE,
            self_attention=True,
        ),
        reported_latency_multiplier=PI0_DENOISING_STEPS,
        published_vla_perf_oi=54.0,
        note="GenZ parallel decode for one denoising pass; VLA-perf multiplies latency by denoising steps.",
    ),
)


def build_points(
    *,
    peak_tflops: float,
    bandwidth_gbps: float,
    throughput_mode: str,
) -> list[StagePoint]:
    balance_oi = peak_tflops * 1000.0 / bandwidth_gbps
    points: list[StagePoint] = []
    for stage in STAGES:
        workload = stage.workload
        executed = _scale_workload(workload, stage.reported_latency_multiplier)
        plotted_workload = executed if throughput_mode == "executed" else workload
        measured_seconds = stage.measured_ms / 1e3
        roofline_tflops_at_oi = min(peak_tflops, workload.oi_flop_per_byte * bandwidth_gbps / 1000.0)
        roofline_workload = executed if throughput_mode == "executed" else workload
        compute_ms = roofline_workload.flops / (peak_tflops * 1e12) * 1e3
        memory_ms = roofline_workload.bytes_ / (bandwidth_gbps * 1e9) * 1e3
        points.append(
            StagePoint(
                stage=stage,
                oi_flop_per_byte=workload.oi_flop_per_byte,
                vla_perf_oi=workload.vla_perf_oi,
                static_flops=workload.flops,
                static_bytes=workload.bytes_,
                executed_flops=executed.flops,
                executed_bytes=executed.bytes_,
                roofline_tflops_at_oi=roofline_tflops_at_oi,
                roofline_ms=max(compute_ms, memory_ms),
                measured_tflops=plotted_workload.flops / measured_seconds / 1e12,
                executed_tflops=executed.flops / measured_seconds / 1e12,
                vla_perf_equiv_bandwidth_gbps=plotted_workload.bytes_ / measured_seconds / 1e9,
                executed_bandwidth_gbps=executed.bytes_ / measured_seconds / 1e9,
                bound="memory" if workload.oi_flop_per_byte < balance_oi else "compute",
            )
        )
    return points


def write_csv(points: list[StagePoint], csv_path: Path, *, balance_oi: float, throughput_mode: str) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "stage",
                "label",
                "oi_flop_per_byte_decimal",
                "vla_perf_oi_mflop_per_binary_mb",
                "published_vla_perf_oi",
                "bound_4090d",
                "balance_oi_4090d",
                "throughput_mode",
                "genz_static_tflops",
                "genz_static_gbytes",
                "executed_tflops_with_denoising",
                "executed_gbytes_with_denoising",
                "roofline_tflops_at_oi",
                "roofline_ms",
                "measured_ms",
                "measured_tflops_plotted",
                "executed_tflops",
                "measured_bandwidth_gbps_plotted",
                "executed_bandwidth_gbps",
                "reported_latency_multiplier",
                "note",
            ]
        )
        for point in points:
            writer.writerow(
                [
                    point.stage.name,
                    point.stage.label,
                    point.oi_flop_per_byte,
                    point.vla_perf_oi,
                    point.stage.published_vla_perf_oi,
                    point.bound,
                    balance_oi,
                    throughput_mode,
                    point.static_flops / 1e12,
                    point.static_bytes / 1e9,
                    point.executed_flops / 1e12,
                    point.executed_bytes / 1e9,
                    point.roofline_tflops_at_oi,
                    point.roofline_ms,
                    point.stage.measured_ms,
                    point.measured_tflops,
                    point.executed_tflops,
                    point.vla_perf_equiv_bandwidth_gbps,
                    point.executed_bandwidth_gbps,
                    point.stage.reported_latency_multiplier,
                    point.stage.note,
                ]
            )


def plot_roofline(
    points: list[StagePoint],
    *,
    peak_tflops: float,
    bandwidth_gbps: float,
    throughput_mode: str,
    png_path: Path,
    pdf_path: Path,
) -> None:
    _apply_plot_style()

    balance_oi = peak_tflops * 1000.0 / bandwidth_gbps
    min_oi = min(point.oi_flop_per_byte for point in points) / 3.0
    max_oi = max(point.oi_flop_per_byte for point in points) * 3.0
    xs = np.logspace(np.log10(min_oi), np.log10(max_oi), 512)
    ys = np.minimum(peak_tflops, xs * bandwidth_gbps / 1000.0)

    colors = {"encoder": "#2d6cdf", "vlm": "#2fa36b", "ae": "#e85d4c"}
    markers = {"encoder": "o", "vlm": "s", "ae": "^"}

    fig, ax = plt.subplots(figsize=(7.6, 5.0), layout="constrained")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.plot(
        xs,
        ys,
        color="#1a1a1a",
        linewidth=2.0,
        solid_capstyle="round",
        label="4090D roofline",
        zorder=1,
    )
    ax.axvline(
        balance_oi,
        color="#9ca3af",
        linewidth=1.2,
        linestyle=(0, (4, 4)),
        zorder=0,
        label=f"Balance OI = {balance_oi:.2f}",
    )

    for point in points:
        ax.scatter(
            point.oi_flop_per_byte,
            point.measured_tflops,
            s=500,
            marker=markers[point.stage.name],
            color=colors[point.stage.name],
            edgecolor="white",
            linewidth=1.35,
            zorder=3,
            label=point.stage.label,
        )

    ax.set_xlim(min_oi, max_oi)
    ax.set_ylim(1.0, peak_tflops * 1.75)
    subtitle = "VLA-perf-style single-pass workload" if throughput_mode == "vla-perf" else "executed workload"
    ax.set_title(rf"$\pi_0$ static roofline (RTX 4090D, {subtitle})")
    ax.set_xlabel("Operational intensity (FLOP/byte)")
    ax.set_ylabel("Measured performance (TFLOP/s)")
    ax.grid(True, which="major", linestyle="-", linewidth=0.55, color="#e5e7eb", alpha=0.95)
    ax.grid(True, which="minor", linestyle="-", linewidth=0.35, color="#f3f4f6", alpha=0.9)
    ax.set_axisbelow(True)

    handles, labels = ax.get_legend_handles_labels()
    seen: set[str] = set()
    uniq_handles: list = []
    uniq_labels: list[str] = []
    for h, lab in zip(handles, labels, strict=True):
        if lab not in seen:
            seen.add(lab)
            uniq_handles.append(h)
            uniq_labels.append(lab)
    ax.legend(
        uniq_handles,
        uniq_labels,
        loc="lower right",
        handlelength=2.4,
        labelspacing=0.55,
        markerscale=0.55,
        scatterpoints=1,
    )

    ax.set_facecolor("#ffffff")
    fig.patch.set_facecolor("#ffffff")

    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=240, facecolor="white")
    fig.savefig(pdf_path, facecolor="white")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot GenZ-style static pi0 roofline for RTX 4090D.")
    parser.add_argument(
        "--peak-tflops",
        type=float,
        default=146.95632,
        help="Default matches balance OI 145.79 at 1008 GB/s, approximately 147 TFLOP/s.",
    )
    parser.add_argument("--bandwidth-gbps", type=float, default=1008.0)
    parser.add_argument(
        "--balance-oi",
        type=float,
        default=None,
        help="Optional balance OI override. If omitted, peak_tflops * 1000 / bandwidth_gbps is used.",
    )
    parser.add_argument(
        "--throughput-mode",
        choices=("vla-perf", "executed"),
        default="executed",
        help=(
            "executed scales AE workload by denoising steps before dividing by total measured latency; "
            "vla-perf uses AE single-pass GenZ workload divided by total measured AE latency."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("scripts/figures"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.balance_oi is not None:
        args.peak_tflops = args.balance_oi * args.bandwidth_gbps / 1000.0
    balance_oi = args.peak_tflops * 1000.0 / args.bandwidth_gbps
    points = build_points(
        peak_tflops=args.peak_tflops,
        bandwidth_gbps=args.bandwidth_gbps,
        throughput_mode=args.throughput_mode,
    )

    output_dir = Path(args.output_dir)
    write_csv(
        points,
        output_dir / "pi0_roofline_4090d.csv",
        balance_oi=balance_oi,
        throughput_mode=args.throughput_mode,
    )
    plot_roofline(
        points,
        peak_tflops=args.peak_tflops,
        bandwidth_gbps=args.bandwidth_gbps,
        throughput_mode=args.throughput_mode,
        png_path=output_dir / "pi0_roofline_4090d.png",
        pdf_path=output_dir / "pi0_roofline_4090d.pdf",
    )

    print(f"4090D balance OI: {balance_oi:.2f} FLOP/Byte")
    print(f"throughput mode: {args.throughput_mode}")
    print(
        "stage\tbound\tOI\tVLA-perf OI\troofline_ms\tmeasured_ms\t"
        "plotted_TFLOP/s\texecuted_TFLOP/s\tplotted_GB/s"
    )
    for point in points:
        print(
            f"{point.stage.name}\t{point.bound}\t{point.oi_flop_per_byte:.1f}\t"
            f"{point.vla_perf_oi:.1f}\t{point.roofline_ms:.2f}\t"
            f"{point.stage.measured_ms:.2f}\t{point.measured_tflops:.2f}\t"
            f"{point.executed_tflops:.2f}\t{point.vla_perf_equiv_bandwidth_gbps:.1f}"
        )
    print(f"wrote {output_dir / 'pi0_roofline_4090d.png'}")
    print(f"wrote {output_dir / 'pi0_roofline_4090d.pdf'}")
    print(f"wrote {output_dir / 'pi0_roofline_4090d.csv'}")


if __name__ == "__main__":
    main()
