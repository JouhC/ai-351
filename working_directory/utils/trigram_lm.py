from collections import Counter
from typing import Tuple

class InterpTrigramLM:
    def __init__(self, uni: Counter, bi: Counter, tri: Counter, vocab_size: int, k_floor=1e-12):
        self.uni, self.bi, self.tri = uni, bi, tri
        self.V = vocab_size
        self.total_uni = sum(uni.values())
        self.k = k_floor
        # Denominators
        self.bi_left = Counter()
        self.tri_left = Counter()
        for (w1, w2), c in bi.items():
            self.bi_left[w1] += c
        for (w1, w2, w3), c in tri.items():
            self.tri_left[(w1, w2)] += c

    def p_uni(self, w:str) -> float:
        return max(self.uni.get(w, 0) / self.total_uni, self.k)

    def p_bi(self, w2:str, w1:str) -> float:
        denom = self.bi_left.get(w1, 0)
        if denom == 0:
            return self.p_uni(w2)
        return max(self.bi.get((w1, w2), 0) / denom, self.k)

    def p_tri(self, w3:str, w1:str, w2:str) -> float:
        denom = self.tri_left.get((w1, w2), 0)
        if denom == 0:
            return self.p_bi(w3, w2)
        return max(self.tri.get((w1, w2, w3), 0) / denom, self.k)

    def p(self, w3:str, w1:str, w2:str, lambdas:Tuple[float,float,float]) -> float:
        l1, l2, l3 = lambdas
        return l1 * self.p_uni(w3) + l2 * self.p_bi(w3, w2) + l3 * self.p_tri(w3, w1, w2)