"""SmartPhotoFilter CLI and GUI entry points."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from src.config import RuntimeConfig, Thresholds
from src.pipeline.exporter import ExportMode, ResultExporter
from src.pipeline.scanner import scan_inputs
from src.pipeline.scorer import Decision, process_batch
from src.utils.platform_utils import detect_runtime_capabilities


def cli_main() -> None:
    main()


def main(argv: list[str] | None = None) -> None:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    raw_args = [arg for arg in raw_args if not arg.startswith("-psn_")]
    if getattr(sys, "frozen", False) and not raw_args:
        _run_gui()
        return

    parser = _build_parser()
    args = parser.parse_args(raw_args)
    if args.gui:
        _run_gui()
        return
    if not args.input:
        parser.print_help()
        return
    _run_cli(args)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SmartPhotoFilter 智能照片筛选")
    parser.add_argument("--gui", action="store_true", help="启动 PyQt6 桌面界面")
    parser.add_argument("--input", "-i", nargs="+", help="输入照片或文件夹")
    parser.add_argument("--output", "-o", help="分类照片和报告的输出文件夹")
    parser.add_argument("--export-mode", choices=[mode.value for mode in ExportMode], default="copy")
    parser.add_argument("--sharpness", type=float, default=100.0, help="Laplacian 模糊阈值")
    parser.add_argument("--aesthetic", type=float, default=6.0, help="0 到 10 分制的保留阈值")
    parser.add_argument("--workers", type=int, default=None, help="最大工作线程数")
    parser.add_argument("--no-ml", action="store_true", help="停用可选 Core ML 运行时")
    return parser


def _run_cli(args: argparse.Namespace) -> None:
    caps = detect_runtime_capabilities()
    assets = scan_inputs(args.input)
    if not assets:
        print("未找到支持的图片。")
        return

    thresholds = Thresholds(
        sharpness_variance=args.sharpness,
        aesthetic_keep_score=args.aesthetic,
    )
    config = RuntimeConfig(
        thresholds=thresholds,
        max_workers=args.workers,
        use_optional_ml=not args.no_ml,
    )

    print(f"运行环境: {caps.system} {caps.machine} | {caps.acceleration_label}")
    print(f"待分析: {len(assets)} 张图片")
    started = time.perf_counter()

    def progress(done: int, total: int, _result) -> None:
        print(f"\r正在分析 {done}/{total}", end="", flush=True)

    results = process_batch(assets, config=config, progress=progress)
    elapsed = time.perf_counter() - started
    print()
    _print_summary(results, elapsed)

    if args.output:
        summary = ResultExporter().export(
            results,
            args.output,
            mode=ExportMode(args.export_mode),
        )
        print(f"报告已写入: {summary.output_dir}")
        if args.export_mode != ExportMode.REPORT_ONLY.value:
            print("分类目录: keep/, review/, discard/")


def _print_summary(results, elapsed: float) -> None:
    counts = {
        decision: sum(1 for result in results if result.decision == decision)
        for decision in Decision
    }
    speed = elapsed / len(results) if results else 0.0
    print(
        "结果: "
        f"{counts[Decision.KEEP]} 保留, "
        f"{counts[Decision.REVIEW]} 待审查, "
        f"{counts[Decision.DISCARD]} 丢弃 "
        f"({elapsed:.2f}s, {speed:.3f}s/张)"
    )
    for result in results:
        print(
            f"{_decision_label(result.decision):<7} "
            f"{result.final_score:5.2f} "
            f"{Path(result.path).name} "
            f"[{'; '.join(result.reasons)}]"
        )


def _decision_label(decision: Decision) -> str:
    return {
        Decision.KEEP: "保留",
        Decision.REVIEW: "待审查",
        Decision.DISCARD: "丢弃",
    }[decision]


def _run_gui() -> None:
    try:
        from src.gui.app import create_app
    except ImportError as exc:
        print("未安装 PyQt6。请运行: pip install -e '.[gui]'", file=sys.stderr)
        raise SystemExit(2) from exc

    app, window = create_app(sys.argv)
    window.show()
    raise SystemExit(app.exec())


if __name__ == "__main__":
    main()
