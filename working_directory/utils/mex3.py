import re
from collections import Counter
from typing import List, Tuple
import random
from .trigram_lm import InterpTrigramLM
import math
from pathlib import Path
import json
import unicodedata
import html

# Regex patterns

_SENT_SPLIT = re.compile(r'(?<=[.!?])\s+')
_WORD_TOK = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?|[^\w\s]")
URL_RE   = re.compile(r"(https?://\S+|www\.\S+)")
EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
NUM_RE   = re.compile(r"[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?")

def read_file_to_docs(filepath: Path) -> List[str]:
    """
    Read a JSONL file (one JSON object per line with a 'text' field),
    apply cleaning, and return a list of cleaned documents.
    """
    docs = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            doc = record["text"]

            # ---- Cleaning ----
            doc = unicodedata.normalize("NFKC", doc)   # normalize unicode
            doc = html.unescape(doc)                   # decode &amp; etc.
            doc = URL_RE.sub("<url>", doc)             # replace URLs
            doc = EMAIL_RE.sub("<email>", doc)         # replace emails
            doc = NUM_RE.sub("<num>", doc)             # replace numbers
            doc = re.sub(r"(''|``)", '', doc)          # remove double quotes
            doc = re.sub(r"(\!\s*)+", '!', doc)        # collapse multiple !
            doc = re.sub(r"(\?\s*)+", '?', doc)        # collapse multiple ?
            doc = re.sub(r'\s+', ' ', doc)             # collapse whitespace

            docs.append(doc.strip().lower())           # lowercase + strip
    return docs

def to_token_sents(texts: List[str]) -> List[List[str]]:
    sents = []
    for t in texts:
        if not t:
            continue
        t = re.sub(r"\s+", " ", t.strip())
        for s in _SENT_SPLIT.split(t):
            toks = _WORD_TOK.findall(s.lower())
            if toks:
                sents.append(toks)
    return sents

def get_vocabulary(texts, pattern=r"\w+|\(|\)|\.|\,", min_freq=10):
    """
    Returns vocabulary as a list
    """
    # Join docs into one string
    corpus = " ".join(texts)

    # Tokenize
    tokens = re.findall(pattern, corpus)

    # Lowercase
    tokens = [t.lower() for t in tokens]

    # Frequency counts
    count = Counter(tokens)

    # Apply cutoff (strictly greater than min_freq)
    vocab = [w for w, c in count.items() if c > min_freq]

    return vocab

def split_train_dev(sents: List[List[str]], dev_ratio=0.1, seed=42):
    rnd = random.Random(seed)
    indices = list(range(len(sents)))
    rnd.shuffle(indices)
    cut = int(len(indices) * (1 - dev_ratio))
    tr = [sents[i] for i in indices[:cut]]
    dv = [sents[i] for i in indices[cut:]]
    return tr, dv

def add_bounds(token_sents: List[List[str]]) -> List[List[str]]:
    return [["<s>", "<s>", *s, "</s>"] for s in token_sents]

def count_ngrams(bounded_sents: List[List[str]]):
    uni, bi, tri = Counter(), Counter(), Counter()
    for s in bounded_sents:
        for w in s:
            uni[w] += 1
        for i in range(1, len(s)):
            bi[(s[i-1], s[i])] += 1
        for i in range(2, len(s)):
            tri[(s[i-2], s[i-1], s[i])] += 1
    return uni, bi, tri

def generate(lm: InterpTrigramLM, lambdas:Tuple[float,float,float], max_tokens=30, seed=42) -> str:
    rng = random.Random(seed)
    w1, w2 = "<s>", "<s>"
    out = []
    vocab = [w for w, _ in lm.uni.most_common()]  # candidate list
    for _ in range(max_tokens):
        probs = [lm.p(w, w1, w2, lambdas) for w in vocab]
        Z = sum(probs)
        r, acc = rng.random(), 0.0
        sample = vocab[-1]
        for w, p in zip(vocab, probs):
            acc += p / Z
            if r <= acc:
                sample = w
                break
        if sample == "</s>":
            break
        out.append(sample)
        w1, w2 = w2, sample
    return " ".join(out)


def perplexity(lm: InterpTrigramLM, sents: List[List[str]], lambdas:Tuple[float,float,float]) -> float:
    neg_log = 0.0
    N = 0
    for s in sents:
        s = ["<s>", "<s>", *s, "</s>"]
        for i in range(2, len(s)):
            p = lm.p(s[i], s[i-2], s[i-1], lambdas)
            neg_log += -math.log(p)
            N += 1
    return math.exp(neg_log / max(N, 1))

def split_train_dev(sents: List[List[str]], dev_ratio=0.1, seed=42):
    rnd = random.Random(seed)
    indices = list(range(len(sents)))
    rnd.shuffle(indices)
    cut = int(len(indices) * (1 - dev_ratio))
    tr = [sents[i] for i in indices[:cut]]
    dv = [sents[i] for i in indices[cut:]]
    return tr, dv

def tune_lambdas(lm: InterpTrigramLM, dev_sents: List[List[str]], step=0.1) -> Tuple[Tuple[float,float,float], float]:
    best_pp = float("inf")
    best = (0.1, 0.2, 0.7)
    # e.g., step=0.1 → grid {0.0,0.1,...,1.0}
    grid = [round(i * step, 10) for i in range(int(1/step) + 1)]
    for l1 in grid:
        for l2 in grid:
            l3 = 1.0 - l1 - l2
            if l3 < 0:
                continue
            pp = perplexity(lm, dev_sents, (l1, l2, l3))
            if pp < best_pp:
                best_pp, best = pp, (l1, l2, l3)
    return best, best_pp