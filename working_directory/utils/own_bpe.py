from collections import Counter, defaultdict
import re

EOW = "</w>"  # end-of-word marker so merges stay inside words

def words(text):
    return re.findall(r"\w+|\S", text.lower())

def train_bpe(texts, k=1000, min_freq=2):
    """
    Learn up to k merges from a small corpus.
    Returns list of merges in order (left_symbol, right_symbol).
    """
    vocab = Counter()
    for t in texts:
        for w in words(t):
            if w.strip():
                vocab[tuple(list(w) + [EOW])] += 1

    merges = []
    for _ in range(k):
        pair_counts = defaultdict(int)
        for sym_seq, freq in vocab.items():
            for i in range(len(sym_seq) - 1):
                pair_counts[(sym_seq[i], sym_seq[i+1])] += freq
        if not pair_counts:
            break

        best_pair, best_cnt = max(pair_counts.items(), key=lambda kv: kv[1])
        if best_cnt < min_freq:
            break

        merges.append(best_pair)
        L, R = best_pair
        merged = L + R

        # replace L R -> merged in every word sequence
        new_vocab = Counter()
        for sym_seq, freq in vocab.items():
            i, out = 0, []
            while i < len(sym_seq):
                if i < len(sym_seq) - 1 and sym_seq[i] == L and sym_seq[i+1] == R:
                    out.append(merged); i += 2
                else:
                    out.append(sym_seq[i]); i += 1
            new_vocab[tuple(out)] += freq
        vocab = new_vocab

    return merges

def bpe_tokenize(text, merges):
    """
    Tokenize a string using learned merges.
    """
    # quick lookup: pair -> rank (earlier merge = higher priority)
    ranks = {p: i for i, p in enumerate(merges)}

    def encode_word(w):
        symbols = list(w) + [EOW]
        # repeatedly apply best available merge
        while True:
            # find best-ranked pair present
            best, best_rank = None, None
            for i in range(len(symbols) - 1):
                p = (symbols[i], symbols[i+1])
                if p in ranks and (best_rank is None or ranks[p] < best_rank):
                    best, best_rank = p, ranks[p]
            if best is None:
                break
            L, R = best
            merged = L + R
            # do one pass replacing that pair
            i, out = 0, []
            while i < len(symbols):
                if i < len(symbols) - 1 and symbols[i] == L and symbols[i+1] == R:
                    out.append(merged); i += 2
                else:
                    out.append(symbols[i]); i += 1
            symbols = out
        # drop EOW
        return [s for s in symbols if s != EOW]

    toks = []
    for w in words(text):
        if w.strip():
            toks.extend(encode_word(w))
    return toks