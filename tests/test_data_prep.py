"""Unit tests for data_prep.py — no GPU, no network, no HF_TOKEN needed."""

from __future__ import annotations

import json

import data_prep


# --------------------------- load_records ---------------------------
def test_load_records_reports_missing_file(tmp_path):
    records, errors = data_prep.load_records(tmp_path / "missing.jsonl")
    assert records == []
    assert "not found" in errors[0]


def test_load_records_reports_invalid_json(tmp_path):
    path = tmp_path / "raw.jsonl"
    path.write_text('{"id": "a"}\nnot json\n', encoding="utf-8")
    records, errors = data_prep.load_records(path)
    assert len(records) == 1
    assert len(errors) == 1
    assert "Line 2" in errors[0]


def test_load_records_skips_blank_lines(tmp_path):
    path = tmp_path / "raw.jsonl"
    path.write_text('{"id": "a"}\n\n{"id": "b"}\n', encoding="utf-8")
    records, errors = data_prep.load_records(path)
    assert len(records) == 2
    assert errors == []


# --------------------------- validate_records ---------------------------
def _valid_record(**overrides) -> dict:
    record = {
        "_line_no": 1,
        "id": "reg-001",
        "area": "business_registration",
        "instruction": "How do I register a business name?",
        "response": "Register through eCitizen. " + data_prep.DISCLAIMER,
        "source_doc": "BRS Guide",
        "source_url": "https://brs.go.ke/guide",
        "verified": True,
    }
    record.update(overrides)
    return record


def test_validate_records_accepts_a_well_formed_record():
    valid, errors = data_prep.validate_records([_valid_record()])
    assert errors == []
    assert len(valid) == 1


def test_validate_records_rejects_missing_field():
    record = _valid_record()
    del record["source_url"]
    valid, errors = data_prep.validate_records([record])
    assert valid == []
    assert any("source_url" in e for e in errors)


def test_validate_records_rejects_unverified():
    valid, errors = data_prep.validate_records([_valid_record(verified=False)])
    assert valid == []
    assert any("verified is not true" in e for e in errors)


def test_validate_records_rejects_bad_area():
    valid, errors = data_prep.validate_records([_valid_record(area="not_a_real_area")])
    assert valid == []
    assert any("not in" in e for e in errors)


def test_validate_records_rejects_duplicate_id():
    a = _valid_record(id="reg-001")
    b = _valid_record(id="reg-001", instruction="A different question?")
    valid, errors = data_prep.validate_records([a, b])
    assert any("duplicate id" in e for e in errors)


def test_validate_records_rejects_empty_instruction():
    valid, errors = data_prep.validate_records([_valid_record(instruction="   ")])
    assert valid == []
    assert any("instruction is empty" in e for e in errors)


# ----------------------- find_duplicate_instructions -----------------------
def test_find_duplicate_instructions_normalises_whitespace_and_case():
    a = _valid_record(id="a", instruction="How do I   register a Business?")
    b = _valid_record(id="b", instruction="how do i register a business?")
    dupes = data_prep.find_duplicate_instructions([a, b])
    assert len(dupes) == 1
    assert set(dupes[0][1]) == {"a", "b"}


def test_find_duplicate_instructions_ignores_unique_questions():
    a = _valid_record(id="a", instruction="How do I register a business?")
    b = _valid_record(id="b", instruction="What is Turnover Tax?")
    assert data_prep.find_duplicate_instructions([a, b]) == []


# ------------------------------ build_messages ------------------------------
def test_build_messages_produces_the_chat_format():
    record = _valid_record()
    out = data_prep.build_messages(record)
    assert out["id"] == record["id"]
    assert out["area"] == record["area"]
    roles = [m["role"] for m in out["messages"]]
    assert roles == ["system", "user", "assistant"]
    assert out["messages"][0]["content"] == data_prep.SYSTEM_PROMPT
    assert out["messages"][1]["content"] == record["instruction"]
    assert out["messages"][2]["content"] == record["response"]


# ------------------------------ stratified_split ------------------------------
def test_stratified_split_partitions_every_record_exactly_once():
    records = [
        _valid_record(id=f"reg-{i:03d}", area="business_registration")
        for i in range(10)
    ] + [
        _valid_record(id=f"tax-{i:03d}", area="tax_obligations")
        for i in range(10)
    ]
    train, val, test = data_prep.stratified_split(records, seed=42, train_frac=0.8, val_frac=0.1)

    assert len(train) + len(val) + len(test) == len(records)
    train_ids = {r["id"] for r in train}
    val_ids = {r["id"] for r in val}
    test_ids = {r["id"] for r in test}
    assert train_ids | val_ids | test_ids == {r["id"] for r in records}
    assert not (train_ids & val_ids)
    assert not (train_ids & test_ids)
    assert not (val_ids & test_ids)


def test_stratified_split_is_deterministic_for_a_fixed_seed():
    records = [_valid_record(id=f"reg-{i:03d}") for i in range(20)]
    split_a = data_prep.stratified_split(records, seed=42, train_frac=0.8, val_frac=0.1)
    split_b = data_prep.stratified_split(records, seed=42, train_frac=0.8, val_frac=0.1)
    ids_a = [[r["id"] for r in group] for group in split_a]
    ids_b = [[r["id"] for r in group] for group in split_b]
    assert ids_a == ids_b


def test_stratified_split_stratifies_each_area_independently():
    records = [
        _valid_record(id=f"reg-{i:03d}", area="business_registration")
        for i in range(50)
    ] + [
        _valid_record(id=f"tax-{i:03d}", area="tax_obligations")
        for i in range(10)
    ]
    train, val, test = data_prep.stratified_split(records, seed=42, train_frac=0.8, val_frac=0.1)

    reg_train = [r for r in train if r["area"] == "business_registration"]
    tax_train = [r for r in train if r["area"] == "tax_obligations"]
    # 80% of 50 == 40, 80% of 10 == 8 -- each area split independently,
    # not just the pooled total, so a small area isn't starved by a big one.
    assert len(reg_train) == 40
    assert len(tax_train) == 8


# -------------------------------- write_jsonl --------------------------------
def test_write_jsonl_round_trips_through_build_messages(tmp_path):
    records = [_valid_record(id="a"), _valid_record(id="b", area="tax_obligations")]
    out_path = tmp_path / "train.jsonl"
    data_prep.write_jsonl(out_path, records)

    lines = out_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    parsed = [json.loads(line) for line in lines]
    assert [p["id"] for p in parsed] == ["a", "b"]
    assert all("_line_no" not in p for p in parsed)  # internal bookkeeping field must not leak


# --------------------------------- main() ---------------------------------
def test_main_end_to_end_passes_on_a_small_valid_dataset(tmp_path, monkeypatch, capsys):
    records = [
        _valid_record(id=f"reg-{i:03d}", area="business_registration") for i in range(5)
    ] + [
        _valid_record(id=f"tax-{i:03d}", area="tax_obligations") for i in range(5)
    ]
    raw_path = tmp_path / "raw_curated.jsonl"
    raw_path.write_text(
        "\n".join(json.dumps({k: v for k, v in r.items() if k != "_line_no"}) for r in records),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "sys.argv",
        [
            "data_prep.py",
            "--input", str(raw_path),
            "--output-dir", str(tmp_path),
            "--report", str(tmp_path / "validation_report.md"),
            "--skip-token-check",
        ],
    )
    exit_code = data_prep.main()
    assert exit_code == 0
    assert (tmp_path / "train.jsonl").exists()
    assert (tmp_path / "val.jsonl").exists()
    assert (tmp_path / "test.jsonl").exists()
    out = capsys.readouterr().out
    assert "VALIDATION PASSED: 10 record(s), zero errors." in out


def test_main_exits_nonzero_and_writes_no_splits_on_invalid_data(tmp_path, monkeypatch):
    raw_path = tmp_path / "raw_curated.jsonl"
    raw_path.write_text(json.dumps({"id": "bad-1"}), encoding="utf-8")  # missing required fields

    monkeypatch.setattr(
        "sys.argv",
        [
            "data_prep.py",
            "--input", str(raw_path),
            "--output-dir", str(tmp_path),
            "--report", str(tmp_path / "validation_report.md"),
            "--skip-token-check",
        ],
    )
    exit_code = data_prep.main()
    assert exit_code == 1
    assert not (tmp_path / "train.jsonl").exists()
