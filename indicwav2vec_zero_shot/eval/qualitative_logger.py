"""
eval/qualitative_logger.py
--------------------------
Per-sample JSONL logging and summary table generation.

This logger deliberately does NOT hide failures with aggregated averages.
Every sample is written individually so that an engineer can inspect
per-sample behaviour.

Output files (per language):
  {output_dir}/{language}_log.jsonl   — one JSON object per line

Summary table:
  Printed to stdout in Markdown table format.
  Also written to {output_dir}/{language}_summary.txt
"""

import json
import os
import time
from typing import List, Optional


class QualitativeLogger:
    """
    Per-language qualitative evaluation logger.

    Parameters
    ----------
    language   : str
    output_dir : str  Directory where JSONL logs are written.
    """

    def __init__(self, language: str, output_dir: str):
        self.language = language
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self._log_path = os.path.join(output_dir, f"{language}_log.jsonl")
        self._records: List[dict] = []

        # Open in append mode so partial runs can be recovered
        self._fh = open(self._log_path, "a", encoding="utf-8")
        print(f"[logger] Writing log to: {self._log_path}")

    # ------------------------------------------------------------------
    # Record a single sample
    # ------------------------------------------------------------------

    def log_sample(
        self,
        audio_id: str,
        hypothesis: str,
        duration_sec: float,
        reference: Optional[str] = None,
        cer: Optional[float] = None,
        wer: Optional[float] = None,
    ) -> None:
        """
        Write one sample record to the JSONL log.

        Parameters
        ----------
        audio_id     : str
        hypothesis   : str   Decoded text from the CTC head
        duration_sec : float Audio duration
        reference    : str   Ground-truth text (optional — may not be available)
        cer          : float Character error rate (optional)
        wer          : float Word error rate (optional)
        """
        record = {
            "audio_id": audio_id,
            "language": self.language,
            "duration_sec": round(duration_sec, 3),
            "hypothesis": hypothesis,
            "reference": reference,
            "cer": round(cer, 4) if cer is not None else None,
            "wer": round(wer, 4) if wer is not None else None,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        self._records.append(record)
        self._fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._fh.flush()

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def write_summary(self) -> str:
        """
        Compute per-language summary statistics and write to stdout + file.

        Returns
        -------
        str  Summary text (Markdown formatted)
        """
        n = len(self._records)
        if n == 0:
            summary = f"[logger] No samples logged for language '{self.language}'."
            print(summary)
            return summary

        avg_duration = sum(r["duration_sec"] for r in self._records) / n

        cer_values = [r["cer"] for r in self._records if r["cer"] is not None]
        wer_values = [r["wer"] for r in self._records if r["wer"] is not None]

        avg_cer = sum(cer_values) / len(cer_values) if cer_values else None
        avg_wer = sum(wer_values) / len(wer_values) if wer_values else None

        # --- Markdown table ---
        lines = [
            "",
            f"## Zero-Shot ASR Summary — {self.language.title()}",
            "",
            "| Field              | Value                     |",
            "|--------------------|---------------------------|",
            f"| Language           | {self.language}           |",
            f"| Samples evaluated  | {n}                       |",
            f"| Avg duration (sec) | {avg_duration:.2f}        |",
            f"| Avg CER            | {_fmt(avg_cer)}           |",
            f"| Avg WER            | {_fmt(avg_wer)}           |",
            f"| Log file           | {self._log_path}          |",
            "",
            "> **Note:** CER > 1.0 is expected and acceptable in zero-shot evaluation.",
            "> The CTC projection head is randomly initialized.",
            "",
        ]

        # Sample-level view (first 5)
        lines.append("### Per-Sample Results (first 5)")
        lines.append("")
        lines.append("| Audio ID | Duration | Reference | Hypothesis | CER | WER |")
        lines.append("|----------|----------|-----------|------------|-----|-----|")
        for r in self._records[:5]:
            ref_disp = _truncate(r["reference"] or "(none)", 30)
            hyp_disp = _truncate(r["hypothesis"], 30)
            lines.append(
                f"| {r['audio_id']} "
                f"| {r['duration_sec']:.2f}s "
                f"| {ref_disp} "
                f"| {hyp_disp} "
                f"| {_fmt(r['cer'])} "
                f"| {_fmt(r['wer'])} |"
            )

        summary = "\n".join(lines)
        print(summary)

        # Write to file
        summary_path = os.path.join(self.output_dir, f"{self.language}_summary.txt")
        with open(summary_path, "w", encoding="utf-8") as fh:
            fh.write(summary)
        print(f"[logger] Summary written to: {summary_path}")

        return summary

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self):
        """Flush and close the JSONL file handle."""
        self._fh.flush()
        self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False  # Do not suppress exceptions


# ---------------------------------------------------------------------------
# Module-level helper: load references from a TSV/plain text file
# ---------------------------------------------------------------------------

def load_references(ref_file: str) -> dict:
    """
    Load reference transcriptions from a plain text file.

    Expected format (tab-separated):
        audio_id<TAB>reference text

    Parameters
    ----------
    ref_file : str  Path to the reference file.

    Returns
    -------
    dict  audio_id → reference text
    """
    if not os.path.isfile(ref_file):
        raise FileNotFoundError(f"Reference file not found: {ref_file}")

    refs = {}
    with open(ref_file, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t", maxsplit=1)
            if len(parts) != 2:
                print(
                    f"[logger] Warning: line {lineno} in '{ref_file}' "
                    "does not match 'audio_id<TAB>reference' format — skipping."
                )
                continue
            audio_id, reference = parts
            refs[audio_id.strip()] = reference.strip()

    return refs


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _fmt(value) -> str:
    if value is None:
        return "N/A"
    if value == float("inf"):
        return "∞"
    return f"{value:.4f}"


def _truncate(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "…"
