"""Validate and split data/raw_curated.jsonl for the BiasharaAssist capstone.

Runs LOCALLY (no GPU needed). Reads the human-curated, provenance-tagged
records in data/raw_curated.jsonl, rejects anything that is not verified and
sourced, reports token-length and duplicate warnings, then writes the clean
chat/messages-format train/val/test splits the training and evaluation
scripts consume.

Usage:
    python data_prep.py
    python data_prep.py --input data/raw_curated.jsonl --output-dir data \
        --seed 42 --max-seq-length 1024

Exit code is non-zero if any record fails structural/provenance validation.
The graded dataset must validate with zero errors (Deliverable 1).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ALLOWED_AREAS = {
    "loan_eligibility",
    "business_registration",
    "tax_obligations",
    "mobile_money",
}

REQUIRED_FIELDS = (
    "id",
    "area",
    "instruction",
    "response",
    "source_doc",
    "source_url",
    "verified",
)

DISCLAIMER = (
    "BiasharaAssist provides general business information only — not "
    "financial, tax, or legal advice. It cannot guarantee loan approval or "
    "any financial outcome. Please confirm details with KRA, the Business "
    "Registration Service, or a licensed advisor; all lending decisions "
    "rest with the institution."
)

SYSTEM_PROMPT = f"""You are BiasharaAssist, a non-advisory information assistant for Kenyan \
small and medium-sized businesses (SMEs). You help with routine questions \
about business registration, tax obligations, loan eligibility, and mobile \
money (M-Pesa / Daraja) integration.

Rules:
- Give general, informational answers only. You are not a licensed \
financial, tax, or legal advisor.
- Never promise loan approval, guarantee a financial outcome, or issue a \
binding tax or legal ruling.
- If asked whether an outcome (such as a loan) is guaranteed, say plainly \
that no outcome can be guaranteed and explain what actually determines it.
- End every answer with this exact disclaimer:

"{DISCLAIMER}\""""

DEFAULT_INPUT = Path("data/raw_curated.jsonl")
DEFAULT_OUTPUT_DIR = Path("data")
DEFAULT_MAX_SEQ_LENGTH = 1024  # matches fine_tune.py's max_seq_length
MIN_TOKEN_LENGTH = 24
TRAIN_FRACTION = 0.8
VAL_FRACTION = 0.1
# TEST_FRACTION is whatever remains (0.1), computed as a remainder below so
# rounding never drops a record.


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--report", type=Path, default=Path("validation_report.md")
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-seq-length", type=int, default=DEFAULT_MAX_SEQ_LENGTH)
    parser.add_argument(
        "--base-model",
        default=os.environ.get("BASE_MODEL", "meta-llama/Meta-Llama-3.1-8B-Instruct"),
        help="Tokenizer used for the token-length check. Defaults to $BASE_MODEL.",
    )
    parser.add_argument(
        "--skip-token-check",
        action="store_true",
        help="Skip loading the tokenizer (useful without HF_TOKEN/network access).",
    )
    return parser.parse_args()


def load_records(path: Path) -> tuple[list[dict], list[str]]:
    """Read raw_curated.jsonl. Returns (records, parse_errors)."""
    records: list[dict] = []
    errors: list[str] = []
    if not path.exists():
        errors.append(f"Input file not found: {path}")
        return records, errors
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"Line {line_no}: invalid JSON ({exc})")
                continue
            record["_line_no"] = line_no
            records.append(record)
    return records, errors


def validate_records(records: list[dict]) -> tuple[list[dict], list[str]]:
    """Validate structure and provenance. Returns (valid_records, errors)."""
    valid: list[dict] = []
    errors: list[str] = []
    seen_ids: set[str] = set()

    for record in records:
        line_no = record.get("_line_no", "?")
        prefix = f"Line {line_no} (id={record.get('id', '?')})"
        record_errors = []

        for field in REQUIRED_FIELDS:
            if field not in record:
                record_errors.append(f"{prefix}: missing required field '{field}'")

        if record_errors:
            errors.extend(record_errors)
            continue

        rid = record["id"]
        if rid in seen_ids:
            record_errors.append(f"{prefix}: duplicate id '{rid}'")
        seen_ids.add(rid)

        if record["area"] not in ALLOWED_AREAS:
            record_errors.append(
                f"{prefix}: area '{record['area']}' not in {sorted(ALLOWED_AREAS)}"
            )

        if not str(record["instruction"]).strip():
            record_errors.append(f"{prefix}: instruction is empty")

        if not str(record["response"]).strip():
            record_errors.append(f"{prefix}: response is empty")

        if record["verified"] is not True:
            record_errors.append(
                f"{prefix}: verified is not true (found {record['verified']!r}) "
                "— unverified records cannot enter the graded split"
            )

        if not str(record.get("source_doc", "")).strip():
            record_errors.append(f"{prefix}: source_doc is empty")

        if not str(record.get("source_url", "")).strip():
            record_errors.append(f"{prefix}: source_url is empty")

        if record_errors:
            errors.extend(record_errors)
        else:
            valid.append(record)

    return valid, errors


def find_duplicate_instructions(records: list[dict]) -> list[tuple[str, list[str]]]:
    """Group ids by normalised instruction text; return groups with >1 id."""
    normalised = defaultdict(list)
    for record in records:
        key = " ".join(str(record["instruction"]).lower().split())
        normalised[key].append(record["id"])
    return [(text, ids) for text, ids in normalised.items() if len(ids) > 1]


def build_messages(record: dict) -> dict:
    return {
        "id": record["id"],
        "area": record["area"],
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": record["instruction"]},
            {"role": "assistant", "content": record["response"]},
        ],
    }


def compute_token_stats(
    records: list[dict], base_model: str, max_seq_length: int
) -> tuple[dict, list[str]]:
    """Tokenise each composed (chat-templated) example. Returns (stats, warnings)."""
    from transformers import AutoTokenizer  # lazy: only needed for this check

    tokenizer = AutoTokenizer.from_pretrained(
        base_model, token=os.environ.get("HF_TOKEN")
    )
    lengths = []
    warnings = []
    for record in records:
        clean = build_messages(record)
        token_ids = tokenizer.apply_chat_template(
            clean["messages"], tokenize=True, add_generation_prompt=False
        )
        length = len(token_ids)
        lengths.append(length)
        if length < MIN_TOKEN_LENGTH:
            warnings.append(
                f"id={record['id']}: {length} tokens, below the {MIN_TOKEN_LENGTH}-token "
                "floor — likely too terse to teach useful behaviour"
            )
        elif length > max_seq_length:
            warnings.append(
                f"id={record['id']}: {length} tokens, exceeds max_seq_length="
                f"{max_seq_length} — will be truncated during training"
            )
    stats = {
        "count": len(lengths),
        "min": min(lengths) if lengths else 0,
        "median": statistics.median(lengths) if lengths else 0,
        "max": max(lengths) if lengths else 0,
    }
    return stats, warnings


def stratified_split(
    records: list[dict], seed: int, train_frac: float, val_frac: float
) -> tuple[list[dict], list[dict], list[dict]]:
    """Split records 80/10/10 stratified by area, with a fixed seed."""
    rng = random.Random(seed)
    by_area: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_area[record["area"]].append(record)

    train: list[dict] = []
    val: list[dict] = []
    test: list[dict] = []
    for area in sorted(by_area):
        group = by_area[area][:]
        rng.shuffle(group)
        n = len(group)
        n_train = round(n * train_frac)
        n_train = min(n_train, n)
        n_val = round(n * val_frac)
        n_val = min(n_val, n - n_train)
        n_test = n - n_train - n_val
        train.extend(group[:n_train])
        val.extend(group[n_train : n_train + n_val])
        test.extend(group[n_train + n_val :])

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)

    train_ids = {r["id"] for r in train}
    val_ids = {r["id"] for r in val}
    test_ids = {r["id"] for r in test}
    assert not (train_ids & val_ids), "id leakage between train and val"
    assert not (train_ids & test_ids), "id leakage between train and test"
    assert not (val_ids & test_ids), "id leakage between val and test"

    return train, val, test


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(build_messages(record), ensure_ascii=False) + "\n")


def write_report(
    report_path: Path,
    *,
    total_input: int,
    valid_count: int,
    errors: list[str],
    duplicates: list[tuple[str, list[str]]],
    token_stats: dict | None,
    token_warnings: list[str],
    area_counts: Counter,
    split_sizes: dict[str, int],
) -> None:
    lines = ["# BiasharaAssist Dataset Validation Report", ""]

    if errors:
        lines.append(f"## Result: FAILED — {len(errors)} error(s)")
        lines.append("")
        lines.append(
            f"{total_input} record(s) read, {valid_count} passed validation, "
            f"{len(errors)} error(s) found. Unverified/unsourced records are "
            "correctly rejected and cannot reach the graded split."
        )
        lines.append("")
        lines.append("## Errors")
        lines.extend(f"- {e}" for e in errors)
        lines.append("")
        lines.append(
            "No train/val/test splits were written because validation did "
            "not pass with zero errors."
        )
        report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    lines.append("## Result: PASSED — zero errors")
    lines.append("")
    lines.append(f"{total_input} record(s) read, all {valid_count} passed validation.")
    lines.append("")

    lines.append("## Counts per area (raw curated set)")
    for area in sorted(ALLOWED_AREAS):
        lines.append(f"- {area}: {area_counts.get(area, 0)}")
    lines.append("")

    lines.append("## Split sizes")
    for name in ("train", "val", "test"):
        lines.append(f"- {name}: {split_sizes.get(name, 0)}")
    lines.append("")

    lines.append("## Token length (composed chat-template example)")
    if token_stats is not None:
        lines.append(f"- min: {token_stats['min']}")
        lines.append(f"- median: {token_stats['median']}")
        lines.append(f"- max: {token_stats['max']}")
    else:
        lines.append("- skipped (--skip-token-check)")
    lines.append("")

    if token_warnings:
        lines.append("## Token-length warnings")
        lines.extend(f"- {w}" for w in token_warnings)
        lines.append("")

    if duplicates:
        lines.append("## Possible duplicate instructions")
        for text, ids in duplicates:
            lines.append(f"- {ids}: \"{text[:100]}\"")
        lines.append("")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()

    records, parse_errors = load_records(args.input)
    valid_records, validation_errors = validate_records(records)
    errors = parse_errors + validation_errors

    duplicates = find_duplicate_instructions(valid_records) if valid_records else []

    if errors:
        write_report(
            args.report,
            total_input=len(records),
            valid_count=len(valid_records),
            errors=errors,
            duplicates=duplicates,
            token_stats=None,
            token_warnings=[],
            area_counts=Counter(),
            split_sizes={},
        )
        print(f"VALIDATION FAILED: {len(errors)} error(s). See {args.report}.", file=sys.stderr)
        for e in errors[:20]:
            print(f"  - {e}", file=sys.stderr)
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more", file=sys.stderr)
        return 1

    area_counts = Counter(r["area"] for r in valid_records)

    token_stats = None
    token_warnings: list[str] = []
    if not args.skip_token_check:
        try:
            token_stats, token_warnings = compute_token_stats(
                valid_records, args.base_model, args.max_seq_length
            )
        except Exception as exc:  # noqa: BLE001 - report and continue without token stats
            print(
                f"WARNING: token-length check skipped — could not load tokenizer "
                f"for '{args.base_model}': {exc}",
                file=sys.stderr,
            )

    train, val, test = stratified_split(
        valid_records, args.seed, TRAIN_FRACTION, VAL_FRACTION
    )
    write_jsonl(args.output_dir / "train.jsonl", train)
    write_jsonl(args.output_dir / "val.jsonl", val)
    write_jsonl(args.output_dir / "test.jsonl", test)

    split_sizes = {"train": len(train), "val": len(val), "test": len(test)}
    write_report(
        args.report,
        total_input=len(records),
        valid_count=len(valid_records),
        errors=[],
        duplicates=duplicates,
        token_stats=token_stats,
        token_warnings=token_warnings,
        area_counts=area_counts,
        split_sizes=split_sizes,
    )

    print(f"VALIDATION PASSED: {len(valid_records)} record(s), zero errors.")
    print(f"Splits written: {split_sizes}")
    print(f"Report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
