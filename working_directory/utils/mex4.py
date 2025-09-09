import random
import re, sys, gzip
from typing import Iterator, Tuple, List, Dict
from pathlib import Path

def _shape(w):
    return "".join(
        "X" if c.isupper() else
        "x" if c.islower() else
        "d" if c.isdigit() else
        "-" if c in "-_/" else "."
        for c in w
    )

def _basic_feats(tok):
    return {
        "w": tok,
        "w.lower": tok.lower(),
        "is_title": tok.istitle(),
        "is_upper": tok.isupper(),
        "is_digit": tok.isdigit(),
        "shape": _shape(tok),
        "pref3": tok[:3].lower(),
        "suf3": tok[-3:].lower(),
        "bias": 1.0,
    }

def read_conll_any(path):
    """
    Flexible reader:
      - If a line has *two* tab-separated cols -> treat as token \t label (your format),
        and immediately build per-token features.
      - Else: split on whitespace; first col = token, last col = label.
    Sentence split on blank line. '-DOCSTART-' is skipped.
    Returns:
      X: list[list[dict]]  (per-token feature dicts)
      Y: list[list[str]]   (BIO/BILOU tags)
    """
    X, Y, curX, curY = [], [], [], []
    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                if curX:
                    X.append(curX); Y.append(curY)
                    curX, curY = [], []
                continue
            if line.startswith("-DOCSTART-"):
                continue

            # Try your strict "token<TAB>label" first
            if "\t" in line:
                tok, lab = line.split("\t", 1)
                feats = _basic_feats(tok)
                curX.append(feats); curY.append(lab)
            else:
                cols = re.split(r"\s+", line)
                if len(cols) < 2:
                    raise ValueError(f"Bad line (need ≥2 cols): {line}")
                tok, lab = cols[0], cols[-1]
                feats = _basic_feats(tok)
                curX.append(feats); curY.append(lab)

    if curX:
        X.append(curX); Y.append(curY)
    return X, Y

def add_context(X):
    """
    Add simple ±1 context using already-built token features.
    Assumes each token feat has at least 'w' / 'w.lower' / casing flags.
    """
    out = []
    for sent in X:
        s = []
        for i, feats in enumerate(sent):
            f = dict(feats)  # copy
            if i == 0: f["BOS"] = True
            if i == len(sent)-1: f["EOS"] = True
            if i > 0:
                p = sent[i-1]
                f["-1.lower"]   = p.get("w.lower")
                f["-1.istitle"] = p.get("is_title", False)
                f["-1.isupper"] = p.get("is_upper", False)
                f["-1.shape"]   = p.get("shape")
            if i < len(sent)-1:
                n = sent[i+1]
                f["+1.lower"]   = n.get("w.lower")
                f["+1.istitle"] = n.get("is_title", False)
                f["+1.isupper"] = n.get("is_upper", False)
                f["+1.shape"]   = n.get("shape")
            s.append(f)
        out.append(s)
    return out

def simple_split(X, Y, dev_ratio=0.1):
    idx = list(range(len(X)))
    random.shuffle(idx)
    cut = max(1, int(len(idx) * (1 - dev_ratio)))
    tr, dv = idx[:cut], idx[cut:]
    Xtr = [X[i] for i in tr]; Ytr = [Y[i] for i in tr]
    Xdv = [X[i] for i in dv]; Ydv = [Y[i] for i in dv]
    return Xtr, Ytr, Xdv, Ydv


def _basic_feats(tok: str) -> Dict[str, str]:
    tl = sys.intern(tok.lower())
    return {
        "w": sys.intern(tok),
        "w.lower": tl,
        "is_title": "1" if tok.istitle() else "0",
        "is_upper": "1" if tok.isupper() else "0",
        "is_digit": "1" if tok.isdigit() else "0",
        "pref3": sys.intern(tok[:3]),
        "suf3": sys.intern(tok[-3:]),
    }

def read_conll_stream(path: Path) -> Iterator[Tuple[List[Dict[str, str]], List[str]]]:
    """Yield (X_sent, y_sent) without holding the whole file in memory."""
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "rt", encoding="utf-8", newline="") as f:
        curX, curY = [], []
        for raw in f:
            line = raw.strip()
            if not line:
                if curX:
                    yield curX, curY
                    curX, curY = [], []
                continue
            if line.startswith("-DOCSTART-"):
                continue
            if "\t" in line:
                tok, lab = line.split("\t", 1)
            else:
                cols = re.split(r"\s+", line)
                if len(cols) < 2:
                    continue
                tok, lab = cols[0], cols[-1]
            curX.append(_basic_feats(tok))
            curY.append(sys.intern(lab))
        if curX:
            yield curX, curY

def add_context_inplace(X_sent: List[Dict[str, str]], left: int = 1, right: int = 1) -> None:
    n = len(X_sent)
    for i in range(n):
        feats = X_sent[i]
        # left context
        for k in range(1, left + 1):
            if i - k >= 0:
                feats[f"-{k}:w.lower"] = X_sent[i - k]["w.lower"]
            else:
                feats[f"-{k}:BOS"] = "1"
        # right context
        for k in range(1, right + 1):
            if i + k < n:
                feats[f"+{k}:w.lower"] = X_sent[i + k]["w.lower"]
            else:
                feats[f"+{k}:EOS"] = "1"