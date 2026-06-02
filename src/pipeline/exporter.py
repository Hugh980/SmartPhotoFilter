"""Export sorted photos and reports."""

from __future__ import annotations

import csv
import json
import shutil
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterable

from src.pipeline.scorer import Decision, PhotoScore


class ExportMode(str, Enum):
    COPY = "copy"
    MOVE = "move"
    REPORT_ONLY = "report-only"


@dataclass(frozen=True)
class ExportSummary:
    output_dir: str
    total: int
    keep: int
    review: int
    discard: int
    report_json: str
    report_csv: str
    summary_md: str


class ResultExporter:
    """Copy or move photos into keep/review/discard folders and write reports."""

    def export(
        self,
        results: Iterable[PhotoScore],
        output_dir: str | Path,
        mode: ExportMode = ExportMode.COPY,
    ) -> ExportSummary:
        result_list = list(results)
        destination = Path(output_dir).expanduser().resolve()
        destination.mkdir(parents=True, exist_ok=True)

        buckets = {
            Decision.KEEP: destination / "keep",
            Decision.REVIEW: destination / "review",
            Decision.DISCARD: destination / "discard",
        }
        if mode != ExportMode.REPORT_ONLY:
            for bucket in buckets.values():
                bucket.mkdir(parents=True, exist_ok=True)
            self._export_files(result_list, buckets, mode)

        report_json = destination / "report.json"
        report_csv = destination / "report.csv"
        summary_md = destination / "summary.md"
        self._write_json(report_json, result_list)
        self._write_csv(report_csv, result_list)
        self._write_summary(summary_md, result_list, mode)

        counts = _counts(result_list)
        return ExportSummary(
            output_dir=str(destination),
            total=len(result_list),
            keep=counts[Decision.KEEP],
            review=counts[Decision.REVIEW],
            discard=counts[Decision.DISCARD],
            report_json=str(report_json),
            report_csv=str(report_csv),
            summary_md=str(summary_md),
        )

    @staticmethod
    def _export_files(
        results: list[PhotoScore],
        buckets: dict[Decision, Path],
        mode: ExportMode,
    ) -> None:
        for result in results:
            source = Path(result.path)
            if not source.exists():
                continue
            target = _unique_target(buckets[result.decision], source.name)
            if mode == ExportMode.MOVE:
                shutil.move(str(source), str(target))
            else:
                shutil.copy2(source, target)

    @staticmethod
    def _write_json(path: Path, results: list[PhotoScore]) -> None:
        with path.open("w", encoding="utf-8") as fh:
            json.dump([result.to_dict() for result in results], fh, indent=2, ensure_ascii=False)

    @staticmethod
    def _write_csv(path: Path, results: list[PhotoScore]) -> None:
        with path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=[
                    "path",
                    "decision",
                    "final_score",
                    "laplacian_variance",
                    "blur_score",
                    "face_count",
                    "expression_score",
                    "aesthetic_score",
                    "reasons",
                ],
            )
            writer.writeheader()
            for result in results:
                writer.writerow(
                    {
                        "path": result.path,
                        "decision": result.decision.value,
                        "final_score": result.final_score,
                        "laplacian_variance": f"{result.sharpness.laplacian_variance:.3f}",
                        "blur_score": f"{result.sharpness.blur_score:.3f}",
                        "face_count": result.face.face_count,
                        "expression_score": f"{result.expression.score:.3f}",
                        "aesthetic_score": f"{result.aesthetic.final_score:.3f}",
                        "reasons": "; ".join(result.reasons),
                    }
                )

    @staticmethod
    def _write_summary(path: Path, results: list[PhotoScore], mode: ExportMode) -> None:
        counts = _counts(results)
        lines = [
            "# SmartPhotoFilter Summary",
            "",
            f"- Export mode: `{mode.value}`",
            f"- Total: {len(results)}",
            f"- Keep: {counts[Decision.KEEP]}",
            f"- Review: {counts[Decision.REVIEW]}",
            f"- Discard: {counts[Decision.DISCARD]}",
            "",
            "## Results",
            "",
            "| Decision | Score | File | Reasons |",
            "|---|---:|---|---|",
        ]
        for result in results:
            lines.append(
                f"| {result.decision.value} | {result.final_score:.2f} | "
                f"`{Path(result.path).name}` | {'; '.join(result.reasons)} |"
            )
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _counts(results: list[PhotoScore]) -> dict[Decision, int]:
    return {decision: sum(1 for result in results if result.decision == decision) for decision in Decision}


def _unique_target(directory: Path, filename: str) -> Path:
    target = directory / filename
    if not target.exists():
        return target
    stem = target.stem
    suffix = target.suffix
    counter = 2
    while True:
        candidate = directory / f"{stem}-{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1
