import mmap
import os
import struct
import time

import numpy as np

from adaptive_ngram_index import AdaptiveNgramIndex
from fixed_ngram_prefix_slab import FixedNgramPrefixSlab
from lisa_ip_sa_index import LisaIndex
from packed_ngram_table import PackedRangeTable


class SuffixArrayQueryIndex:
    """Loads and queries a single-shard infini-gram index."""

    def __init__(
        self, index_dir, token_width=2, build_lisa=False, lisa_k=2, lisa_leaf_alpha=16.0
    ):
        self.token_width = token_width
        self.index_dir = index_dir
        self._perf_active = False
        self._perf_sa_time = 0.0

        ds_path = os.path.join(index_dir, "tokenized.0")
        self.ds_size = os.path.getsize(ds_path)
        self.tok_cnt = self.ds_size // token_width
        self.ds_file = open(ds_path, "rb")
        self.ds_mmap = mmap.mmap(self.ds_file.fileno(), 0, access=mmap.ACCESS_READ)

        sa_path = os.path.join(index_dir, "table.0")
        sa_size = os.path.getsize(sa_path)
        self.ptr_size = sa_size // self.tok_cnt
        self.sa_file = open(sa_path, "rb")
        self.sa_mmap = mmap.mmap(self.sa_file.fileno(), 0, access=mmap.ACCESS_READ)

        self.prefix_slab = None

        bg_path = os.path.join(index_dir, "bigram.0")
        tg_path = os.path.join(index_dir, "trigram.0")
        qg_path = os.path.join(index_dir, "quadgram.0")
        self.bigram_cache = None
        self.trigram_cache = None
        self.quadgram_cache = None
        if os.path.exists(bg_path):
            if self.token_width in (1, 2):
                self.prefix_slab = self._load_prefix_slab(bg_path, tg_path, qg_path)
                self.bigram_cache = self.prefix_slab.bigram_level
                self.trigram_cache = self.prefix_slab.trigram_level
                self.quadgram_cache = self.prefix_slab.quadgram_level
            else:
                self.bigram_cache = self._load_bigram_cache(bg_path)
                if os.path.exists(tg_path):
                    self.trigram_cache = self._load_ngram_cache(tg_path)
                if os.path.exists(qg_path):
                    self.quadgram_cache = self._load_ngram_cache(qg_path)

        adaptive_path = os.path.join(index_dir, "adaptive_ngram.0")
        self.adaptive_cache = None
        if os.path.exists(adaptive_path):
            self.adaptive_cache = self._load_adaptive_ngram(adaptive_path)

        if token_width == 2:
            self.dtype = np.uint16
        elif token_width == 1:
            self.dtype = np.uint8
        elif token_width == 4:
            self.dtype = np.uint32
        else:
            raise ValueError(f"Unsupported token width: {token_width}")

        self.unigram_cum = None
        self.unigram_cache = self._build_unigram_cache()
        if self.unigram_cache is not None:
            self.unigram_cum = self._build_unigram_cum()

        self.lisa = None
        if build_lisa:
            self.lisa = LisaIndex(self, k_tokens=lisa_k, leaf_alpha=lisa_leaf_alpha)

    def _perf_begin_call(self):
        self._perf_sa_time = 0.0
        self._perf_active = True

    def _perf_end_call(self):
        sa_time = self._perf_sa_time
        self._perf_active = False
        return sa_time

    def _perf_add_sa_time(self, elapsed):
        if self._perf_active:
            self._perf_sa_time += elapsed

    def _load_bigram_cache(self, path):
        with open(path, "rb") as f:
            (num_entries,) = struct.unpack("<Q", f.read(8))
            (tw,) = struct.unpack("<I", f.read(4))
            (_padding,) = struct.unpack("<I", f.read(4))
            assert tw == self.token_width
            data_offset = f.tell()
        return PackedRangeTable(
            path, num_entries, data_offset, ngram_n=2, sort_order="lex_bytes"
        )

    def _load_ngram_cache(self, path):
        with open(path, "rb") as f:
            (num_entries,) = struct.unpack("<Q", f.read(8))
            (tw,) = struct.unpack("<I", f.read(4))
            (ngram_n,) = struct.unpack("<I", f.read(4))
            assert tw == self.token_width
            data_offset = f.tell()
        return PackedRangeTable(
            path, num_entries, data_offset, ngram_n=ngram_n, sort_order="lex_bytes"
        )

    def _load_prefix_slab(self, bigram_path, trigram_path, quadgram_path):
        trigram_in = trigram_path if os.path.exists(trigram_path) else None
        quadgram_in = quadgram_path if os.path.exists(quadgram_path) else None
        return FixedNgramPrefixSlab(
            bigram_path,
            trigram_in,
            quadgram_in,
            token_width=self.token_width,
            tok_cnt=self.tok_cnt,
        )

    def _load_adaptive_ngram(self, path):
        return AdaptiveNgramIndex(path, token_width=self.token_width)

    def _build_unigram_cache(self):
        if self.bigram_cache is None:
            return None
        if self.prefix_slab is not None:
            return self.prefix_slab.build_unigram_cache()

        cache = {}
        mask = (1 << (self.token_width * 8)) - 1
        for key, lo, hi in self.bigram_cache.iter_ranges():
            t1 = key & mask
            if t1 in cache:
                old_lo, old_hi = cache[t1]
                cache[t1] = (min(old_lo, lo), max(old_hi, hi))
            else:
                cache[t1] = (lo, hi)
        return cache

    def _build_unigram_cum(self):
        vocab_size = 1 << (self.token_width * 8)
        deltas = np.zeros(vocab_size + 1, dtype=np.int64)
        for token_id, (lo, hi) in self.unigram_cache.items():
            if 0 <= token_id < vocab_size:
                deltas[token_id + 1] = hi - lo
        return np.cumsum(deltas)

    def _make_bigram_key(self, token_ids):
        return self._make_ngram_key(token_ids, 2)

    def _make_ngram_key(self, token_ids, ngram_n):
        key_bytes = b""
        for i in range(ngram_n):
            key_bytes += int(token_ids[i]).to_bytes(self.token_width, "little")
        key_bytes += b"\x00" * (8 - ngram_n * self.token_width)
        return struct.unpack("<Q", key_bytes)[0]

    def _read_sa_ptr(self, rank):
        offset = rank * self.ptr_size
        raw = self.sa_mmap[offset : offset + self.ptr_size]
        padded = raw + b"\x00" * (8 - self.ptr_size)
        return struct.unpack("<Q", padded)[0]

    def _read_suffix_bytes(self, ptr, num_bytes):
        end = min(ptr + num_bytes, self.ds_size)
        return self.ds_mmap[ptr:end]

    def _compare_suffix(self, rank, query_bytes):
        ptr = self._read_sa_ptr(rank)
        suffix = self._read_suffix_bytes(ptr, len(query_bytes))
        if suffix < query_bytes:
            return -1
        if suffix > query_bytes:
            return 1
        return 0

    def _token_ids_to_bytes(self, token_ids):
        arr = np.array(token_ids, dtype=self.dtype)
        return arr.tobytes()

    def binary_search(self, token_ids, lo=None, hi=None):
        start_t = time.perf_counter() if self._perf_active else None
        try:
            if lo is None:
                lo = 0
            if hi is None:
                hi = self.tok_cnt

            query_bytes = self._token_ids_to_bytes(token_ids)
            num_bytes = len(query_bytes)
            comparisons = 0

            if num_bytes == 0:
                return lo, hi, comparisons

            orig_lo, orig_hi = lo, hi
            match_idx = lo
            while lo < hi:
                match_idx = (lo + hi - 1) >> 1
                cmp_result = self._compare_suffix(match_idx, query_bytes)
                comparisons += 1
                if cmp_result < 0:
                    lo = match_idx + 1
                elif cmp_result > 0:
                    hi = match_idx
                else:
                    break

            if lo == hi:
                return lo, lo, comparisons

            left_lo, left_hi = orig_lo - 1, match_idx
            while left_hi - left_lo > 1:
                mid = (left_lo + left_hi) >> 1
                ptr = self._read_sa_ptr(mid)
                suffix = self._read_suffix_bytes(ptr, num_bytes)
                comparisons += 1
                if suffix < query_bytes:
                    left_lo = mid
                else:
                    left_hi = mid
            left = left_hi

            right_lo, right_hi = match_idx, orig_hi
            while right_hi - right_lo > 1:
                mid = (right_lo + right_hi) >> 1
                ptr = self._read_sa_ptr(mid)
                suffix = self._read_suffix_bytes(ptr, num_bytes)
                comparisons += 1
                if query_bytes < suffix:
                    right_hi = mid
                else:
                    right_lo = mid
            right = right_hi
            return left, right, comparisons
        finally:
            if start_t is not None:
                self._perf_add_sa_time(time.perf_counter() - start_t)

    def _exp_search_boundary(self, query_bytes, est, lo_bnd, hi_bnd, strict):
        start_t = time.perf_counter() if self._perf_active else None
        try:
            if lo_bnd >= hi_bnd:
                return lo_bnd, 0

            num_bytes = len(query_bytes)
            comparisons = [0]

            def is_left(rank):
                comparisons[0] += 1
                ptr = self._read_sa_ptr(rank)
                suffix = self._read_suffix_bytes(ptr, num_bytes)
                if strict:
                    return not (query_bytes < suffix)
                return suffix < query_bytes

            if est < lo_bnd:
                est = lo_bnd
            if est >= hi_bnd:
                est = hi_bnd - 1

            if is_left(est):
                lo = est
                step = 1
                hi = hi_bnd
                while True:
                    if est + step >= hi_bnd:
                        hi = hi_bnd
                        break
                    cand = est + step
                    if is_left(cand):
                        lo = cand
                        step <<= 1
                    else:
                        hi = cand
                        break
            else:
                hi = est
                step = 1
                found_lo = False
                lo = 0
                while True:
                    cand = lo_bnd if step > est - lo_bnd else est - step
                    if is_left(cand):
                        lo = cand
                        found_lo = True
                        break
                    hi = cand
                    if cand == lo_bnd:
                        break
                    step <<= 1
                if not found_lo:
                    return lo_bnd, comparisons[0]

            while hi - lo > 1:
                mid = (lo + hi) >> 1
                if is_left(mid):
                    lo = mid
                else:
                    hi = mid
            return hi, comparisons[0]
        finally:
            if start_t is not None:
                self._perf_add_sa_time(time.perf_counter() - start_t)

    def search_baseline(self, token_ids):
        return self.binary_search(token_ids)

    def search_with_unigram(self, token_ids):
        if self.unigram_cache is None or len(token_ids) < 1:
            return self.binary_search(token_ids)

        first = int(token_ids[0])
        if first in self.unigram_cache:
            lo, hi = self.unigram_cache[first]
            return self.binary_search(token_ids, lo, hi)
        return 0, 0, 0

    def search_with_bigram(self, token_ids):
        if self.bigram_cache is None or len(token_ids) < 2:
            return self.binary_search(token_ids)

        if self.prefix_slab is not None:
            entry = self.prefix_slab.lookup_bigram_tokens(token_ids)
        else:
            entry = self.bigram_cache.lookup(self._make_bigram_key(token_ids))
        if entry is None:
            return 0, 0, 0
        lo, hi = entry[:2]
        return self.binary_search(token_ids, lo, hi)

    def search_with_trigram(self, token_ids):
        if self.trigram_cache is not None and len(token_ids) >= 3:
            if self.prefix_slab is not None:
                entry = self.prefix_slab.lookup_trigram_tokens(token_ids)
            else:
                entry = self.trigram_cache.lookup(self._make_ngram_key(token_ids, 3))
            if entry is None:
                return 0, 0, 0
            lo, hi = entry[:2]
            return self.binary_search(token_ids, lo, hi)
        return self.search_with_bigram(token_ids)

    def search_with_quadgram(self, token_ids):
        if self.quadgram_cache is not None and len(token_ids) >= 4:
            if self.prefix_slab is not None:
                entry = self.prefix_slab.lookup_quadgram_tokens(token_ids)
            else:
                entry = self.quadgram_cache.lookup(self._make_ngram_key(token_ids, 4))
            if entry is None:
                return 0, 0, 0
            lo, hi = entry[:2]
            return self.binary_search(token_ids, lo, hi)
        return self.search_with_trigram(token_ids)

    def search_with_1approx2(self, token_ids):
        if (
            len(token_ids) != 2
            or self.unigram_cache is None
            or self.unigram_cum is None
        ):
            return self.binary_search(token_ids)

        a, b = int(token_ids[0]), int(token_ids[1])
        if a not in self.unigram_cache:
            return 0, 0, 0

        uni_a_lo, uni_a_hi = self.unigram_cache[a]
        if uni_a_lo >= uni_a_hi:
            return uni_a_lo, uni_a_lo, 0

        total = self.tok_cnt
        range_a = uni_a_hi - uni_a_lo
        cum_b_lo = int(self.unigram_cum[b])
        cum_b_hi = int(self.unigram_cum[b + 1])
        est_lo = uni_a_lo + (range_a * cum_b_lo) // total
        est_hi = uni_a_lo + (range_a * cum_b_hi) // total
        est_lo = max(uni_a_lo, min(est_lo, uni_a_hi))
        est_hi = max(uni_a_lo, min(est_hi, uni_a_hi))

        query_bytes = self._token_ids_to_bytes([a, b])
        left, c1 = self._exp_search_boundary(
            query_bytes, est_lo, uni_a_lo, uni_a_hi, strict=False
        )
        right, c2 = self._exp_search_boundary(
            query_bytes, est_hi, uni_a_lo, uni_a_hi, strict=True
        )
        if right < left:
            right = left
        return left, right, c1 + c2

    def search_with_2approx3(self, token_ids):
        if len(token_ids) != 3 or self.bigram_cache is None or self.unigram_cum is None:
            return self.binary_search(token_ids)

        a, b, c = int(token_ids[0]), int(token_ids[1]), int(token_ids[2])
        if self.prefix_slab is not None:
            bi_entry = self.prefix_slab.lookup_bigram_tokens((a, b))
        else:
            bi_entry = self.bigram_cache.lookup(self._make_ngram_key([a, b], 2))
        if bi_entry is None:
            return 0, 0, 0

        bi_lo, bi_hi = bi_entry[:2]
        if bi_lo >= bi_hi:
            return bi_lo, bi_lo, 0

        total = self.tok_cnt
        range_ab = bi_hi - bi_lo
        cum_c_lo = int(self.unigram_cum[c])
        cum_c_hi = int(self.unigram_cum[c + 1])
        est_lo = bi_lo + (range_ab * cum_c_lo) // total
        est_hi = bi_lo + (range_ab * cum_c_hi) // total
        est_lo = max(bi_lo, min(est_lo, bi_hi))
        est_hi = max(bi_lo, min(est_hi, bi_hi))

        query_bytes = self._token_ids_to_bytes([a, b, c])
        left, c1 = self._exp_search_boundary(
            query_bytes, est_lo, bi_lo, bi_hi, strict=False
        )
        right, c2 = self._exp_search_boundary(
            query_bytes, est_hi, bi_lo, bi_hi, strict=True
        )
        if right < left:
            right = left
        return left, right, c1 + c2

    def search_with_adaptive(self, token_ids):
        if self.adaptive_cache is None:
            return self.binary_search(token_ids)

        entry = self.adaptive_cache.lookup(token_ids, self._make_ngram_key)
        if entry is None:
            if len(token_ids) >= 2:
                return 0, 0, 0
            return self.binary_search(token_ids)
        lo, hi = entry[:2]
        return self.binary_search(token_ids, lo, hi)

    def search_with_lisa_binary(self, token_ids):
        if self.lisa is None:
            return self.binary_search(token_ids)
        return self.lisa.search(token_ids, use_rmi=False)

    def search_with_lisa_rmi(self, token_ids):
        if self.lisa is None:
            return self.binary_search(token_ids)
        return self.lisa.search(token_ids, use_rmi=True)

    def close(self):
        for cache in (self.bigram_cache, self.trigram_cache, self.quadgram_cache):
            if cache is not None and hasattr(cache, "close"):
                cache.close()
        if self.adaptive_cache is not None:
            self.adaptive_cache.close()
        self.ds_mmap.close()
        self.ds_file.close()
        self.sa_mmap.close()
        self.sa_file.close()


SuffixArrayIndex = SuffixArrayQueryIndex
