#!/usr/bin/env python3
"""Fast, read-only PLELog semantic-text preflight.

Mirrors the text-selection and `[a-z]+` tokenization rules in
PLELog_FIXED_TOKENSAFE.py without importing PLELog, loading GloVe vectors,
rebuilding windows, or training/evaluating any model.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd

TOKEN_RE = re.compile(r"[a-z]+")
UNUSABLE = {"nan", "none", "null", "<na>"}


def nonempty_text(value):
    if not isinstance(value, str):
        return None
    value = value.strip()
    if not value or value.lower() in UNUSABLE:
        return None
    return value


def tokenize(text):
    return TOKEN_RE.findall(text.lower())


def unique_mapping(frame, ids, field):
    """Match the existing runner: one unique nonblank text per EventId."""
    selected = frame.loc[frame["EventId"].isin(ids), ["EventId", field]].drop_duplicates().copy()
    selected[field] = selected[field].map(nonempty_text)
    selected = selected.dropna(subset=[field]).drop_duplicates()
    conflicts = selected.groupby("EventId", sort=False)[field].nunique()
    conflicts = conflicts[conflicts > 1]
    mapping = dict(zip(selected["EventId"], selected[field]))
    return mapping, {str(k): int(v) for k, v in conflicts.head(12).items()}


def scan_glove_vocab(glove_path, required):
    """Check lexical coverage without loading 300-d vectors into memory."""
    path = Path(glove_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"GloVe file does not exist: {path}")
    matched = set()
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            parts = line.rstrip("\n").split(" ")
            if len(parts) == 301 and parts[0] in required:
                matched.add(parts[0])
                if len(matched) == len(required):
                    break
    return matched


def preflight(train_pkl, semantic_column, glove_file=None):
    train_pkl = Path(train_pkl).expanduser()
    if not train_pkl.is_file():
        raise FileNotFoundError(f"Training PKL not found: {train_pkl}")
    frame = pd.read_pickle(train_pkl)
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"Expected event-row pandas DataFrame; got {type(frame).__name__}")
    for col in ("EventId", semantic_column):
        if col not in frame:
            raise ValueError(f"Missing required column {col!r}; available: {list(frame.columns)}")
    if frame["EventId"].isna().any():
        raise ValueError(f"Found {int(frame['EventId'].isna().sum())} training rows with null EventId")
    known = set(frame["EventId"].unique().tolist())
    original_available = "EventTemplate" in frame.columns and semantic_column != "EventTemplate"
    report = {
        "source": str(train_pkl.resolve()),
        "read_only": True,
        "splits_used": ["train"],
        "rows": int(len(frame)),
        "unique_training_event_ids": len(known),
        "preferred_column": semantic_column,
        "fallback_column": "EventTemplate" if original_available else None,
        "training_rows_with_null_preferred_text": int(frame[semantic_column].isna().sum()),
        "tokenizer": "[a-z]+",
        "no_sequence_regrouping": True,
    }
    def fail(reason, examples=None):
        report["status"] = "FAIL"
        report["failure_reason"] = reason
        if examples:
            report["examples"] = examples
        return report, []

    preferred_map, conflicts = unique_mapping(frame, known, semantic_column)
    if conflicts:
        return fail("Conflicting preferred template text for an EventId", conflicts)
    missing_preferred = known - set(preferred_map)
    fallback_map = {}
    if missing_preferred:
        if not original_available:
            return fail("Missing preferred text and EventTemplate fallback not available",
                        [str(x) for x in list(missing_preferred)[:12]])
        fallback_map, conflicts = unique_mapping(frame, missing_preferred, "EventTemplate")
        if conflicts:
            return fail("Conflicting original template text for missing preferred EventIds", conflicts)
    mapping = {eid: preferred_map[eid] for eid in known if eid in preferred_map}
    mapping.update({eid: fallback_map[eid] for eid in missing_preferred if eid in fallback_map})
    missing_both = known - set(mapping)
    report["event_ids_missing_text_in_both_columns"] = len(missing_both)
    report["missing_text_event_id_examples"] = [str(eid) for eid in sorted(missing_both, key=str)[:12]]
    if missing_both:
        return fail("No nonblank processed or original text for EventId",
                    report["missing_text_event_id_examples"])

    docs = {eid: tokenize(txt) for eid, txt in mapping.items()}
    tokenless_before = [eid for eid, words in docs.items() if not words]
    recovered_tokenless = set()
    if tokenless_before and original_available:
        original_map, conflicts = unique_mapping(frame, set(tokenless_before), "EventTemplate")
        if conflicts:
            return fail("Conflicting original template text for tokenless EventIds", conflicts)
        for eid in tokenless_before:
            if eid in original_map:
                raw_words = tokenize(original_map[eid])
                if raw_words:
                    docs[eid] = raw_words
                    recovered_tokenless.add(eid)

    fallback_used = set(fallback_map) | recovered_tokenless
    no_tokens = sorted([eid for eid, words in docs.items() if not words], key=str)
    report.update({
        "event_ids_using_preferred_text": len(known - fallback_used),
        "event_ids_using_original_template_fallback": len(fallback_used),
        "event_ids_fallback_due_to_missing_preferred_text": len(fallback_map),
        "event_ids_recovered_from_tokenless_processed_text": len(recovered_tokenless),
        "event_ids_with_empty_tokens_in_both_sources": len(no_tokens),
        "tokenless_event_id_examples": [str(x) for x in no_tokens[:12]],
        "fallback_event_id_examples": [str(x) for x in sorted(fallback_used, key=str)[:12]],
        "event_ids_with_at_least_one_token": len(known) - len(no_tokens),
        "status": "WARN_ZERO_TOKENS" if no_tokens else "PASS",
    })
    if glove_file:
        required = {word for words in docs.values() for word in words}
        matched = scan_glove_vocab(glove_file, required)
        no_covered = sorted([eid for eid, words in docs.items() if not any(w in matched for w in words)], key=str)
        report["glove_check"] = {
            "glove_file": str(Path(glove_file).expanduser().resolve()),
            "distinct_template_words": len(required),
            "words_in_glove": len(matched),
            "vocabulary_coverage": len(matched) / len(required) if required else 0.0,
            "event_ids_without_any_glove_word": len(no_covered),
            "examples_without_glove_word": [str(x) for x in no_covered[:12]],
            "note": "Lexical coverage only; does not calculate exact TF-IDF weighted vectors",
        }
        if not matched:
            report["status"] = "FAIL"
            report["failure_reason"] = "No training-template words found in GloVe"
        elif no_covered and report["status"] == "PASS":
            report["status"] = "WARN_GLOVE_COVERAGE"

    rows = []
    no_tokens_set = set(no_tokens)
    for eid in sorted(known, key=str):
        if eid in no_tokens_set:
            outcome = "ZERO_TOKENS"
        elif eid in recovered_tokenless:
            outcome = "ORIGINAL_FALLBACK_TOKENLESS"
        elif eid in fallback_map:
            outcome = "ORIGINAL_FALLBACK_MISSING"
        else:
            outcome = "PROCESSED"
        rows.append({"EventId": str(eid), "semantic_source": outcome, "token_count": len(docs[eid])})
    return report, rows


def main():
    ap = argparse.ArgumentParser(description="Check PLELog template fallback before training (PKL read-only)")
    ap.add_argument("--dataset", default="BGL")
    ap.add_argument("--data_root", default="/storage/home/roqaya/LWADLS/datasets")
    ap.add_argument("--train-pkl", default=None)
    ap.add_argument("--semantic_column", default="processed_EventTemplate")
    ap.add_argument("--glove_file", default=None, help="Optional coverage check; scans words, not embeddings")
    ap.add_argument("--strict", action="store_true", help="Treat zero-token or uncovered-GloVe cases as failure")
    ap.add_argument("--output_dir", default=".")
    args = ap.parse_args()
    pkl = args.train_pkl or str(Path(args.data_root) / args.dataset / f"1_{args.dataset}_Splitted_Datasets" / "train_df.pkl")
    report, rows = preflight(pkl, args.semantic_column, args.glove_file)
    dest = Path(args.output_dir)
    dest.mkdir(parents=True, exist_ok=True)
    summary_path = dest / f"PLELog_Semantic_Preflight_{args.dataset}.json"
    summary_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    detail_path = dest / f"PLELog_Semantic_Preflight_{args.dataset}_eventids.csv"
    if rows:
        with detail_path.open("w", newline="", encoding="utf-8") as out:
            writer = csv.DictWriter(out, fieldnames=("EventId", "semantic_source", "token_count"))
            writer.writeheader()
            writer.writerows(rows)
    print("\n[PLELOG SEMANTIC PREFLIGHT]")
    for label in ("rows", "unique_training_event_ids", "training_rows_with_null_preferred_text",
                  "event_ids_using_preferred_text", "event_ids_using_original_template_fallback",
                  "event_ids_fallback_due_to_missing_preferred_text",
                  "event_ids_recovered_from_tokenless_processed_text",
                  "event_ids_with_empty_tokens_in_both_sources", "event_ids_missing_text_in_both_columns"):
        if label in report:
            print(f"  {label}: {report[label]}")
    if "glove_check" in report:
        print("  GloVe coverage:", report["glove_check"])
    if report.get("failure_reason"):
        print("  Failure reason:", report["failure_reason"])
        print("  Examples:", report.get("examples", []))
    if report.get("tokenless_event_id_examples"):
        print("  Tokenless EventIds:", report["tokenless_event_id_examples"])
    print("  STATUS:", report["status"])
    print("  Audit:", summary_path.resolve())
    if rows:
        print("  Per-EventId detail:", detail_path.resolve())
    if report["status"] == "FAIL":
        return 2
    if args.strict and report["status"].startswith("WARN"):
        print("  --strict: stopping because semantic coverage is incomplete")
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
