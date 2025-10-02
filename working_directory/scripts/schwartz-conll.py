#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Create a CoNLL BIO dataset (token <TAB> tag) suitable for NLTK-style
ClassifierBasedTagger BIO training.

Input: JSONL with fields:
  - file_name : str
  - text      : str
  - n_sections: int (metadata; not used)

Output (default): token \t BIO
Sentences separated by a blank line. Optional -DOCSTART- per doc.

Usage:
  python make_conll_for_classifierbasedtagger.py \
      --input ./papers.jsonl \
      --out ./output.conll
"""

import argparse
import json
import re
from typing import List, Tuple, Iterable, Dict

# ----------------------------
# Regex heuristics
# ----------------------------

ACRONYM_RE = re.compile(r"^[A-Z][A-Z0-9\-]{1,14}$")          # e.g., NHS, HCP-DS
PAREN_PAIR_RE = re.compile(r"\(([^)]+)\)")
SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")
TOKEN_SPLIT_RE = re.compile(r"(\w+|[^\w\s])", re.UNICODE)

# Capitalized sequence (up to 6 tokens)
CAPSEQ = r"(?:[A-Z][A-Za-z0-9\-]+(?:\s+[A-Z][A-Za-z0-9\-]+){0,5})"
DATA_TAIL = r"(?:dataset|data set|database|corpus|survey|study|initiative|registry|archive|repository)"
PATTERN_RIGHT = re.compile(rf"\b{CAPSEQ}\s+{DATA_TAIL}\b")
PATTERN_LEFT  = re.compile(rf"\b{DATA_TAIL}\s+{CAPSEQ}\b", re.IGNORECASE)

# Source priority when resolving overlaps
SOURCE_PRIORITY = {"SH": 2, "PAT": 1}

# ----------------------------
# Basic text utilities
# ----------------------------

def sent_tokenize(text: str) -> List[str]:
    text = text.strip()
    if not text:
        return []
    parts = SENT_SPLIT_RE.split(text)
    return [p.strip() for p in parts if p.strip()]

def tokenize(sent: str) -> List[str]:
    return TOKEN_SPLIT_RE.findall(sent)

def char_spans(tokens: List[str], sent: str) -> List[Tuple[int,int]]:
    spans = []
    idx = 0
    for tok in tokens:
        start = sent.find(tok, idx)
        if start == -1:
            start = idx
        end = start + len(tok)
        spans.append((start, end))
        idx = end
    return spans

def dedup_overlapping(spans: List[Tuple[int,int,str]], priority: Dict[str,int]) -> List[Tuple[int,int,str]]:
    """
    Keep highest-priority source; tie-break by longer span.
    spans: list of (start, end, source_tag)
    """
    spans_sorted = sorted(spans, key=lambda x: (x[0], -(x[1]-x[0])))
    kept = []
    for s, e, src in spans_sorted:
        drop = False
        for i,(ks,ke,ksrc) in enumerate(kept):
            if not (e <= ks or s >= ke):  # overlap
                pri_new = priority.get(src, 0)
                pri_old = priority.get(ksrc, 0)
                len_new = e - s
                len_old = ke - ks
                keep_new = (pri_new > pri_old) or (pri_new == pri_old and len_new > len_old)
                if keep_new:
                    kept[i] = (s, e, src)
                drop = not keep_new
                break
        if not drop:
            kept.append((s,e,src))
    return sorted(kept, key=lambda x: (x[0], x[1]))

# ----------------------------
# Step 1: Schwartz–Hearst (practical)
# ----------------------------

def sh_extract_pairs(sentence: str) -> List[Tuple[Tuple[int,int], Tuple[int,int]]]:
    """
    Return list of ((full_start,full_end),(acr_start,acr_end)) spans.
    Detects "Full (ACR)" and "ACR (Full)" patterns.
    """
    pairs = []
    for m in PAREN_PAIR_RE.finditer(sentence):
        inner = m.group(1).strip()
        par_start, par_end = m.span()

        if ACRONYM_RE.match(inner):
            # Full (ACR)
            acr_span = (par_start+1, par_end-1)
            left_ctx = sentence[:par_start].rstrip()
            cands = list(re.finditer(rf"{CAPSEQ}$", left_ctx))
            fm = cands[-1] if cands else re.search(
                r"([A-Z][A-Za-z0-9\-]+(?:\s+[A-Z][A-Za-z0-9\-]+){0,3})\s*$", left_ctx
            )
            if fm:
                pairs.append(((fm.start(), fm.end()), acr_span))
        else:
            # ACR (Full)
            before = sentence[:par_start].rstrip()
            m2 = re.search(r"([A-Z][A-Z0-9\-]{1,14})\s*$", before)
            if m2 and re.match(rf"^{CAPSEQ}$", inner):
                pairs.append(((par_start+1, par_end-1), (m2.start(1), m2.end(1))))
    return pairs

def sh_spans(sentence: str) -> List[Tuple[int,int]]:
    spans = []
    for (fs, fe), (as_, ae) in sh_extract_pairs(sentence):
        spans.append((fs, fe))   # full name
        spans.append((as_, ae))  # acronym
    spans = dedup_overlapping([(s,e,"SH") for (s,e) in spans], SOURCE_PRIORITY)
    return [(s,e) for (s,e,_) in spans]

# ----------------------------
# Step 2: Pattern-based candidates
# ----------------------------

def pattern_spans(sentence: str) -> List[Tuple[int,int]]:
    spans = []
    for m in PATTERN_RIGHT.finditer(sentence):
        spans.append((m.start(), m.end()))
    for m in PATTERN_LEFT.finditer(sentence):
        spans.append((m.start(), m.end()))
    # "X data/records" variant
    m3 = re.compile(rf"\b{CAPSEQ}\s+(?:data|records)\b", re.IGNORECASE)
    for x in m3.finditer(sentence):
        spans.append((x.start(), x.end()))
    spans = dedup_overlapping([(s,e,"PAT") for (s,e) in spans], SOURCE_PRIORITY)
    return [(s,e) for (s,e,_) in spans]

# ----------------------------
# Step 3: BIO tagging (token -> BIO)
# ----------------------------

def to_bio_for_sentence(sentence: str, mention_spans: List[Tuple[int,int]]) -> List[Tuple[str,str]]:
    tokens = tokenize(sentence)
    spans = char_spans(tokens, sentence)
    labels = ["O"] * len(tokens)

    for (ms, me) in mention_spans:
        first = True
        for i,(ts,te) in enumerate(spans):
            if te <= ms or ts >= me:
                continue
            labels[i] = "B-DATASET" if first else "I-DATASET"
            first = False

    return list(zip(tokens, labels))

# ----------------------------
# JSONL IO + pipeline
# ----------------------------

def iter_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

def process_document(text: str):
    conll_doc = []
    for sent in sent_tokenize(text):
        sh = sh_spans(sent)
        pat = pattern_spans(sent)
        combined = [(s,e,"SH") for (s,e) in sh] + [(s,e,"PAT") for (s,e) in pat]
        combined = dedup_overlapping(combined, SOURCE_PRIORITY)
        spans = [(s,e) for (s,e,_) in combined]
        conll_sent = to_bio_for_sentence(sent, spans)
        conll_doc.append(conll_sent)
    return conll_doc

def write_conll(jsonl_path: str, out_path: str, add_docstart: bool = True, include_comments: bool = True):
    with open(out_path, "w", encoding="utf-8") as w:
        for row in iter_jsonl(jsonl_path):
            file_name = row.get("file_name", "UNKNOWN")
            text = row.get("text", "") or ""

            if add_docstart:
                w.write("-DOCSTART- O\n\n")
            if include_comments:
                w.write(f"# file_name: {file_name}\n\n")

            doc = process_document(text)
            for sent in doc:
                for tok, lab in sent:
                    # Minimal columns for NLTK ClassifierBasedTagger BIO: token \t tag
                    w.write(f"{tok}\t{lab}\n")
                w.write("\n")  # sentence boundary

# ----------------------------
# CLI
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to input JSONL (file_name,text,n_sections).")
    ap.add_argument("--out", default="output.conll", help="Path to write CoNLL.")
    ap.add_argument("--no_docstart", action="store_true", help="Disable -DOCSTART- per document.")
    ap.add_argument("--no_comments", action="store_true", help="Disable comment lines like # file_name.")
    args = ap.parse_args()

    write_conll(
        jsonl_path=args.input,
        out_path=args.out,
        add_docstart=not args.no_docstart,
        include_comments=not args.no_comments
    )
    print(f"Wrote CoNLL to: {args.out}")

if __name__ == "__main__":
    main()
