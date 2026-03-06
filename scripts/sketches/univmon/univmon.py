import hashlib
import math
import os
import struct
import heapq
from typing import Callable, Iterable, List, Tuple, Any


# --------------------- hashing helpers ---------------------
def _u32_sha256(data: bytes) -> int:
    return struct.unpack("<I", hashlib.sha256(data).digest()[:4])[0]

def _bit_sha256(seed_u32: int, key_b: bytes) -> int:
    # low bit of sha256(seed||key)
    return _u32_sha256(seed_u32.to_bytes(4, "little") + key_b) & 1

def _to_bytes(x: Any) -> bytes:
    if isinstance(x, (bytes, bytearray)):
        return bytes(x)
    if isinstance(x, str):
        return x.encode("utf-8")
    if isinstance(x, int):
        n = max(1, (x.bit_length() + 7) // 8)
        return x.to_bytes(n, "little", signed=False)
    return repr(x).encode("utf-8")


# --------------------- minimal CountSketch ---------------------
class CountSketch:
    """
    CountSketch with d rows and w columns.
    API:
        CountSketch(num_of_counters=d, length_of_counter=w)
        add(elem, delta=1)
        frequency(elem) -> int   # median-of-rows estimate
    """
    def __init__(self, num_of_counters: int = 5, length_of_counter: int = 2000) -> None:
        assert num_of_counters > 0 and length_of_counter > 0
        self.d = int(num_of_counters)
        self.w = int(length_of_counter)
        self.table: List[List[int]] = [[0] * self.w for _ in range(self.d)]
        rnd = os.urandom
        self.index_seeds = [int.from_bytes(rnd(4), "little") for _ in range(self.d)]
        self.sign_seeds  = [int.from_bytes(rnd(4), "little") for _ in range(self.d)]

    def _col(self, r: int, key_b: bytes) -> int:
        return _u32_sha256(self.index_seeds[r].to_bytes(4, "little") + key_b) % self.w

    def _sign(self, r: int, key_b: bytes) -> int:
        return 1 if (_u32_sha256(self.sign_seeds[r].to_bytes(4, "little") + key_b) & 1) else -1

    def add(self, elem: Any, delta: int = 1) -> None:
        key_b = _to_bytes(elem)
        dlt = int(delta)
        for r in range(self.d):
            j = self._col(r, key_b)
            s = self._sign(r, key_b)
            self.table[r][j] += s * dlt

    def frequency(self, elem: Any) -> int:
        key_b = _to_bytes(elem)
        vals = []
        for r in range(self.d):
            j = self._col(r, key_b)
            s = self._sign(r, key_b)
            vals.append(s * self.table[r][j])
        vals.sort()
        m = len(vals) // 2
        if len(vals) % 2:
            return int(vals[m])
        return int((vals[m - 1] + vals[m]) // 2)


# --------------------- UnivMon (domain-driven) ---------------------
class UnivMon:
    """
    UnivMon with one CountSketch per level.
    - Insert cascades down levels using sha256(seed||key) low bit.
    - All queries operate over an explicit 'domain' iterable of keys.
    - k controls how many top domain keys (by estimate) we include per level in g_sum.
    """
    def __init__(self, mem_in_bytes: int, level: int = 14, rows: int = 5, k: int = 1000) -> None:
        assert mem_in_bytes > 0 and level > 0 and rows > 0 and k > 0
        self.mem_in_bytes = int(mem_in_bytes)
        self.level = int(level)
        self.rows = int(rows)   # CountSketch num_of_counters (d)
        self.k = int(k)         # per-level top-k from the provided domain
        self.element_num = 0

        # Size each level's CountSketch: approximately rows * width * 4 bytes
        per_level_bytes = max(1, self.mem_in_bytes // self.level)
        width = max(32, per_level_bytes // (self.rows * 4))

        self.sketches: List[CountSketch] = [
            CountSketch(num_of_counters=self.rows, length_of_counter=width)
            for _ in range(self.level)
        ]
        self.polar_seeds: List[int] = [int.from_bytes(os.urandom(4), "little") for _ in range(self.level)]

    # ------------- stream update -------------
    def insert(self, key: Any) -> None:
        key_b = _to_bytes(key)
        self.element_num += 1
        # Level 0 always
        self.sketches[0].add(key)
        # Cascade while bit == 1
        for i in range(1, self.level):
            if _bit_sha256(self.polar_seeds[i], key_b) == 1:
                self.sketches[i].add(key)
            else:
                break

    # ------------- internals -------------
    def _level_topk_over_domain(self, level_idx: int, domain: Iterable[Any], k: int) -> List[Tuple[Any, int]]:
        cs = self.sketches[level_idx]
        ests: List[Tuple[Any, int]] = []
        for key in domain:
            try:
                est = int(cs.frequency(key))
            except Exception:
                try:
                    est = int(cs.frequency(_to_bytes(key)))
                except Exception:
                    continue
            if est > 0:
                ests.append((key, est))
        if not ests:
            return []
        return heapq.nlargest(k, ests, key=lambda kv: kv[1])

    # ------------- UnivMon estimators (domain required) -------------
    def g_sum(self, g: Callable[[float], float], domain: Iterable[Any]) -> float:
        Y = [0.0] * self.level
        for i in range(self.level - 1, -1, -1):
            Y[i] = 0.0 if i == self.level - 1 else 2.0 * Y[i + 1]
            for key, est in self._level_topk_over_domain(i, domain, self.k):
                if est <= 0:
                    continue
                if i == self.level - 1:
                    coe = 1
                else:
                    coe = 1 - 2 * (_bit_sha256(self.polar_seeds[i + 1], _to_bytes(key)))
                Y[i] += coe * g(float(est))
        return Y[0]

    def get_cardinality(self, domain: Iterable[Any]) -> float:
        return float(self.g_sum(lambda _: 1.0, domain))

    def get_entropy(self, domain: Iterable[Any]) -> float:
        if self.element_num == 0:
            return 0.0
        sum_term = self.g_sum(lambda x: 0.0 if x == 0 else x * math.log2(x), domain)
        return math.log2(self.element_num) - (sum_term / self.element_num)

    def get_heavy_hitters(self, threshold: int, domain: Iterable[Any]) -> List[Tuple[str, int]]:
        """
        For each key in domain, take the max CountSketch estimate across all levels.
        Return keys with estimate >= threshold, sorted by (count desc, key asc).
        """
        out: List[Tuple[str, int]] = []
        for key in domain:
            best = 0
            for i in range(self.level):
                try:
                    v = int(self.sketches[i].frequency(key))
                except Exception:
                    try:
                        v = int(self.sketches[i].frequency(_to_bytes(key)))
                    except Exception:
                        v = 0
                if v > best:
                    best = v
            if best >= threshold:
                out.append((str(key) if isinstance(key, (str, int)) else _to_bytes(key).hex(), best))
        out.sort(key=lambda x: (-x[1], x[0]))
        return out


# --------------------- quick demo ---------------------
if __name__ == "__main__":
    import random
    from collections import Counter
    random.seed(0)

    um = UnivMon(mem_in_bytes=1_000_000, level=12, rows=5, k=1000) # 1MB

    heavy = [f"hot{i}" for i in range(10)]
    tail  = [f"id{i}" for i in range(10, 50000)]
    domain = heavy + tail

    N = 10000
    gold = []
    for _ in range(N):
        item = random.choice(heavy) if random.random() < 0.8 else random.choice(tail)
        um.insert(item)
        gold.append(item)

    # ----- gold status -----
    counter = Counter(gold)
    gold_cardinality = len(counter)
    probs = [c / N for c in counter.values()]
    gold_entropy = -sum(p * math.log2(p) for p in probs)
    gold_HH = [(k, v) for k, v in counter.items() if v >= 300]
    gold_HH.sort(key=lambda x: (-x[1], x[0]))

    print("=== Gold status ===")
    print("cardinality:", gold_cardinality)
    print("entropy:", gold_entropy)
    print("HH sample:", gold_HH[:10])

    # ----- UnivMon estimates -----
    print("\n=== UnivMon estimates ===")
    print("cardinality(domain):", um.get_cardinality(domain))
    print("entropy(domain):", um.get_entropy(domain))
    print("HH sample:", um.get_heavy_hitters(threshold=300, domain=domain)[:10])