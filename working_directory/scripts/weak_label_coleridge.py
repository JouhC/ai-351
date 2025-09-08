#!/usr/bin/env python3
"""
weak_label_coleridge.py
-----------------------
Create a weakly-labeled BIO CoNLL file (train_weak.conll) for the Coleridge dataset.

It scans each paper JSON, reads all dataset titles for that Id from train.csv,
builds robust surface-form candidates (original, no-parentheses, inside-parentheses,
acronyms, normalized variants), finds char spans with regex (case-insensitive, punctuation-tolerant),
then aligns to tokens and emits BIO labels.

Dependencies: only Python stdlib.

USAGE
=====
python weak_label_coleridge.py \
  --train_csv /path/to/train.csv \
  --json_dir  /path/to/train_jsons \
  --out_conll /path/to/train_weak.conll \
  [--min_tokens 2] [--max_tokens 8] [--fuzzy_threshold 0] \
  [--context_cues]

Notes
-----
- Supports either 'dataset_title' or 'cleaned_label' column in train.csv.
- JSON files must be named '{Id}.json' and contain a list of sections with 'text' fields.
- Tokenization & sentence splitting are regex-based for portability.
- "Fuzzy" here means allowing punctuation/spacing variation between words (not edit-distance).
- If --context_cues is set, spans near left/right cue words are preferred; generic single-token spans may be filtered.
"""

import argparse, csv, json, re, string
from pathlib import Path

# --------------------------- Tokenization & Sentence Split ---------------------------

ABBREVS = ["Mr.", "Mrs.", "Ms.", "Dr.", "Prof.", "Sr.", "Jr.", "St.", "vs.", "etc.", "e.g.", "i.e.", "U.S.", "U.K.", "U.N."]
DOT_TOKEN = "§DOT§"

def protect_abbrevs(text: str):
    for ab in ABBREVS:
        text = text.replace(ab, ab.replace(".", DOT_TOKEN))
    return text

def unprotect_abbrevs(text: str):
    return text.replace(DOT_TOKEN, ".")

def sent_tokenize(text: str):
    t = re.sub(r'\s+', ' ', text).strip()
    if not t: return []
    t = protect_abbrevs(t)
    t = re.sub(r'([.!?])\s+(?=[A-Z0-9(])', r'\1<SPLIT>', t)
    parts = [unprotect_abbrevs(p).strip() for p in t.split("<SPLIT>")]
    return [p for p in parts if p]

_TOK_RE = re.compile(
    r"""
    (?:[A-Za-z]\.){2,}                # U.S.A.
    |[A-Za-z]+(?:[-'][A-Za-z]+)*\.?   # words incl. hyphen/apostrophe (optional final .)
    |\d{1,3}(?:,\d{3})*(?:\.\d+)?     # numbers
    |[$€£]\d+(?:,\d{3})*(?:\.\d+)?    # money
    |\([^\s)]*\)                      # (ERS)
    |[%&@#]                           # symbols
    |[;:.,!?/"“”‘’()\[\]-]            # punctuation
    """,
    re.VERBOSE
)

def tokenize(text: str):
    return _TOK_RE.findall(text)

# --------------------------- Candidate Generation ---------------------------

def normalize_space_punct(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[\s"+re.escape(string.punctuation)+"]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def acronym(s: str) -> str:
    words = re.findall(r"[A-Za-z]+", s)
    if not words: return ""
    return "".join(w[0].upper() for w in words if w)

def gen_candidates(title: str):
    """
    Build a set of surface-forms:
      - original title
      - title without parenthetical
      - inside parentheses pieces
      - acronym (>=3 letters)
      - normalized variants (punct/space collapsed, lowercased)
    """
    cands = set()
    t = (title or "").strip()
    if not t: return []
    cands.add(t)

    # inside parentheses
    ins = re.findall(r"\(([^)]+)\)", t)
    for p in ins:
        p = p.strip()
        if len(p) >= 2:
            cands.add(p)

    # title without parentheses
    no_paren = re.sub(r"\s*\([^)]*\)\s*", " ", t).strip()
    if no_paren and no_paren != t:
        cands.add(no_paren)

    # acronym
    acro = acronym(no_paren or t)
    if len(acro) >= 3:
        cands.add(acro)

    # normalized variants
    cands.update({normalize_space_punct(x) for x in list(cands)})
    return [c for c in cands if c]

# --------------------------- Span Finding ---------------------------

def build_patterns(cand: str, min_tokens=2, max_tokens=8, fuzzy_threshold=0):
    """
    Create regex patterns tolerant to punctuation/spacing.
    If cand is already normalized (spaces+lowercase), allow punctuation between parts.
    Always include a strict word-boundary exact pattern.
    """
    pats = []
    # flexible pattern if normalized
    if cand == normalize_space_punct(cand):
        parts = [re.escape(p) for p in cand.split()]
        if 1 <= len(parts) <= max_tokens:
            # Allow up to 3 non-word chars and optional spaces between parts
            flex = r"\b" + r"[^\w]{0,3}\s*".join(parts) + r"\b"
            pats.append(flex)
    # exact raw
    pats.append(r"\b" + re.escape(cand) + r"\b")
    return pats

def find_spans(text: str, candidates, min_tokens=2, max_tokens=8):
    spans = []
    for cand in candidates:
        for pat in build_patterns(cand, min_tokens, max_tokens):
            for m in re.finditer(pat, text, flags=re.IGNORECASE):
                spans.append((m.start(), m.end(), m.group(0)))
    # Deduplicate & merge: sort by start, prefer longer spans on overlaps
    spans = sorted(spans, key=lambda x: (x[0], -(x[1]-x[0])))
    merged, taken = [], [False]*len(spans)
    for i,(s,e,txt) in enumerate(spans):
        if taken[i]: continue
        best = i
        for j in range(i+1, len(spans)):
            s2,e2,_ = spans[j]
            if s2 < e:
                taken[j] = True
                if (e2 - s2) > (spans[best][1] - spans[best][0]):
                    best = j
            else:
                break
        taken[best] = True
        merged.append(spans[best])
    return merged

# Context cues to keep/reject ambiguous single-word hits
LEFT_CUES  = {"using","from","based","obtained","via","with","leveraged","drawn"}
RIGHT_CUES = {"data","dataset","records","microdata","survey","corpus"}

def apply_context_filter(text: str, span, window=5):
    s,e,_ = span
    # Build a spaCy-free token list via regex to locate index positions
    toks = tokenize(text)
    offs = []
    idx = 0
    for t in toks:
        j = text.find(t, idx)
        if j == -1: j = idx
        offs.append((j, j+len(t)))
        idx = j + len(t)
    # find token index range covering span
    left_i = next((i for i,(ts,te) in enumerate(offs) if ts <= s < te or (s <= ts and te <= e)), None)
    right_i = next((i for i,(ts,te) in reversed(list(enumerate(offs))) if ts < e <= te or (s <= ts and te <= e)), None)
    if left_i is None or right_i is None: 
        return True  # can't decide, keep
    L = [t.lower() for t in toks[max(0, left_i-window):left_i]]
    R = [t.lower() for t in toks[right_i+1:min(len(toks), right_i+1+window)]]
    return (any(w in LEFT_CUES for w in L) or any(w in RIGHT_CUES for w in R))

# --------------------------- BIO Alignment ---------------------------

def to_bio(text: str, spans):
    tokens = []
    offs = []
    i = 0
    # sentence-by-sentence to preserve sentence breaks
    for sent in sent_tokenize(text):
        for tok in tokenize(sent):
            j = text.find(tok, i)
            if j == -1: j = i
            tokens.append(tok)
            offs.append((j, j+len(tok)))
            i = j + len(tok)
        tokens.append("")  # sentence break
        offs.append((-1,-1))

    labels = []
    for k,(ts,te) in enumerate(offs):
        if ts == -1:
            labels.append("")
            continue
        lab = "O"
        for (ss,se,_) in spans:
            if te <= ss or ts >= se:
                continue
            if k>0 and labels[k-1] in {"B-DATASET","I-DATASET"}:
                # if previous token also overlaps the same span, continue I-
                prev_ts, prev_te = offs[k-1]
                if not (prev_te <= ss or prev_ts >= se):
                    lab = "I-DATASET"
                else:
                    lab = "B-DATASET"
            else:
                lab = "B-DATASET"
            break
        labels.append(lab)
    return list(zip(tokens, labels))

# --------------------------- Main Pipeline ---------------------------

def main(train_csv: Path, json_dir: Path, out_conll: Path, min_tokens=2, max_tokens=8, use_context=False):
    # Read mapping Id -> titles
    id2titles = {}
    with open(train_csv, encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        fields = [c.strip() for c in rdr.fieldnames]
        title_key = "dataset_title" if "dataset_title" in fields else ("cleaned_label" if "cleaned_label" in fields else None)
        if not title_key:
            raise ValueError(f"Expected 'dataset_title' or 'cleaned_label' in {fields}")
        for row in rdr:
            pid = row["Id"].strip()
            ttl = row[title_key].strip()
            if not pid or not ttl: 
                continue
            id2titles.setdefault(pid, []).append(ttl)

    n_docs = 0
    n_spans = 0

    with open(out_conll, "w", encoding="utf-8") as out:
        for json_path in sorted(json_dir.glob("*.json")):
            pid = json_path.stem
            if pid not in id2titles:
                continue
            # Load paper text
            data = json.loads(json_path.read_text(encoding="utf-8"))
            secs = data if isinstance(data, list) else []
            text = " ".join([sec.get("text","") for sec in secs])
            if not text.strip():
                continue

            # Build candidates
            cand_set = set()
            for ttl in id2titles[pid]:
                for c in gen_candidates(ttl):
                    cand_set.add(c)

            # Find spans
            raw_spans = find_spans(text, cand_set, min_tokens=min_tokens, max_tokens=max_tokens)

            # Optional context filtering for single-token generic hits
            spans = []
            for s in raw_spans:
                span_text = s[2]
                # if span is single token and looks generic, require context cues when enabled
                if use_context and len(span_text.split()) == 1 and span_text.lower() in {"survey","dataset","program","census"}:
                    if apply_context_filter(text, s):
                        spans.append(s)
                else:
                    spans.append(s)

            if not spans:
                continue

            # Emit BIO
            bio = to_bio(text, spans)
            out.write(f"-DOCSTART- ({pid})\n")
            for tok, lab in bio:
                if tok == "":
                    out.write("\n")
                else:
                    out.write(f"{tok}\t{lab}\n")
            out.write("\n")

            n_docs += 1
            n_spans += len(spans)

    print(f"[OK] wrote {out_conll} | docs labeled: {n_docs} | spans found: {n_spans}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", required=True, type=Path)
    ap.add_argument("--json_dir", required=True, type=Path)
    ap.add_argument("--out_conll", required=True, type=Path)
    ap.add_argument("--min_tokens", type=int, default=2, help="min tokens in flexible pattern (normalized)")
    ap.add_argument("--max_tokens", type=int, default=8, help="max tokens in flexible pattern (normalized)")
    ap.add_argument("--context_cues", action="store_true", help="enable context filtering for generic 1-token spans")
    args = ap.parse_args()
    main(args.train_csv, args.json_dir, args.out_conll, args.min_tokens, args.max_tokens, args.context_cues)
