import bisect
import os
import struct

import numpy as np

from packed_ngram_table import PackedRangeTable


class PrefixSlabLevel:
    """Thin metadata wrapper around one level of the fixed prefix slab."""

    def __init__(self, slab, ngram_n, num_entries, mem_bytes_estimate):
        self.slab = slab
        self.ngram_n = int(ngram_n)
        self.num_entries = int(num_entries)
        self.mem_bytes_estimate = int(mem_bytes_estimate)

    def __len__(self):
        return self.num_entries

    def close(self):
        return None


class FixedNgramPrefixSlab:
    """Packed in-memory trie over the fixed 2/3/4-gram cache files."""

    RECORD_DTYPE = PackedRangeTable.RECORD_DTYPE

    def __init__(self, bigram_path, trigram_path, quadgram_path, token_width, tok_cnt):
        if token_width not in (1, 2):
            raise ValueError(
                f"Prefix slab is only implemented for token_width 1 or 2, got {token_width}"
            )

        self.token_width = int(token_width)
        self.token_bits = self.token_width * 8
        self.token_mask = (1 << self.token_bits) - 1
        self.token_dtype = np.uint8 if self.token_width == 1 else np.uint16
        self.range_dtype = np.uint32 if tok_cnt <= np.iinfo(np.uint32).max else np.uint64
        self.offset_dtype = np.uint32
        self.vocab_size = 1 << self.token_bits

        self.root_start = np.zeros(self.vocab_size, dtype=self.offset_dtype)
        self.root_len = np.zeros(self.vocab_size, dtype=self.offset_dtype)

        bigram_records = self._open_ngram_records(bigram_path, expected_n=2)
        (self.bigram_tokens,
         self.bigram_lo,
         self.bigram_hi,
         bigram_prefix_keys) = self._build_bigram_level(bigram_records)
        self.bigram_count = len(self.bigram_tokens)
        self.bigram_mem_bytes = (
            self.root_start.nbytes
            + self.root_len.nbytes
            + self.bigram_tokens.nbytes
            + self.bigram_lo.nbytes
            + self.bigram_hi.nbytes
        )
        self.bigram_level = PrefixSlabLevel(self, 2, self.bigram_count, self.bigram_mem_bytes)
        self._close_memmap(bigram_records)

        self.trigram_tokens = None
        self.trigram_lo = None
        self.trigram_hi = None
        self.bigram_trigram_start = None
        self.bigram_trigram_len = None
        self.trigram_prefix_keys = None
        self.trigram_count = 0
        self.trigram_mem_bytes = 0
        self.trigram_level = None

        if trigram_path and os.path.exists(trigram_path):
            trigram_records = self._open_ngram_records(trigram_path, expected_n=3)
            (self.trigram_tokens,
             self.trigram_lo,
             self.trigram_hi,
             self.bigram_trigram_start,
             self.bigram_trigram_len,
             self.trigram_prefix_keys) = self._build_trigram_level(
                trigram_records, bigram_prefix_keys
            )
            self.trigram_count = len(self.trigram_tokens)
            self.trigram_mem_bytes = (
                self.bigram_trigram_start.nbytes
                + self.bigram_trigram_len.nbytes
                + self.trigram_tokens.nbytes
                + self.trigram_lo.nbytes
                + self.trigram_hi.nbytes
            )
            self.trigram_level = PrefixSlabLevel(self, 3, self.trigram_count, self.trigram_mem_bytes)
            self._close_memmap(trigram_records)

        self.quadgram_tokens = None
        self.quadgram_lo = None
        self.quadgram_hi = None
        self.trigram_quad_start = None
        self.trigram_quad_len = None
        self.quadgram_count = 0
        self.quadgram_mem_bytes = 0
        self.quadgram_level = None

        if quadgram_path and os.path.exists(quadgram_path):
            if self.trigram_prefix_keys is None:
                raise ValueError("Quadgram slab requires trigram slab metadata")
            quadgram_records = self._open_ngram_records(quadgram_path, expected_n=4)
            (self.quadgram_tokens,
             self.quadgram_lo,
             self.quadgram_hi,
             self.trigram_quad_start,
             self.trigram_quad_len) = self._build_quadgram_level(
                quadgram_records, self.trigram_prefix_keys
            )
            self.quadgram_count = len(self.quadgram_tokens)
            self.quadgram_mem_bytes = (
                self.trigram_quad_start.nbytes
                + self.trigram_quad_len.nbytes
                + self.quadgram_tokens.nbytes
                + self.quadgram_lo.nbytes
                + self.quadgram_hi.nbytes
            )
            self.quadgram_level = PrefixSlabLevel(self, 4, self.quadgram_count, self.quadgram_mem_bytes)
            self._close_memmap(quadgram_records)

    def _open_ngram_records(self, path, expected_n):
        with open(path, 'rb') as f:
            num_entries, = struct.unpack('<Q', f.read(8))
            tw, = struct.unpack('<I', f.read(4))
            assert tw == self.token_width
            ngram_n, = struct.unpack('<I', f.read(4))
            if expected_n == 2:
                assert ngram_n in (0, 2)
            else:
                assert ngram_n == expected_n
            data_offset = f.tell()
        return np.memmap(
            path,
            dtype=self.RECORD_DTYPE,
            mode='r',
            offset=data_offset,
            shape=(num_entries,),
        )

    def _close_memmap(self, mm):
        mmap_obj = getattr(mm, '_mmap', None)
        if mmap_obj is not None:
            mmap_obj.close()

    def _token_sort_array(self, tokens):
        if self.token_width == 1:
            return np.ascontiguousarray(tokens, dtype=self.token_dtype)
        return np.ascontiguousarray(tokens.byteswap(), dtype=self.token_dtype)

    def _token_sort_key(self, token_id):
        if self.token_width == 1:
            return int(token_id)
        return int.from_bytes(int(token_id).to_bytes(self.token_width, 'little'), 'big')

    def _group_starts_and_lens(self, prefix_keys):
        if len(prefix_keys) == 0:
            empty = np.zeros(0, dtype=self.offset_dtype)
            return empty, empty
        change = np.flatnonzero(prefix_keys[1:] != prefix_keys[:-1]) + 1
        starts = np.empty(len(change) + 1, dtype=self.offset_dtype)
        starts[0] = 0
        if len(change):
            starts[1:] = change.astype(self.offset_dtype, copy=False)
        ends = np.empty(len(starts), dtype=np.int64)
        if len(starts) > 1:
            ends[:-1] = starts[1:]
        ends[-1] = len(prefix_keys)
        lens = (ends - starts.astype(np.int64)).astype(self.offset_dtype, copy=False)
        return starts, lens

    def _build_bigram_level(self, records):
        keys = records['key']
        t1 = (keys & self.token_mask).astype(self.token_dtype, copy=False)
        t2 = ((keys >> self.token_bits) & self.token_mask).astype(self.token_dtype, copy=False)
        t2_sorted = self._token_sort_array(t2)
        lo = np.ascontiguousarray(records['lo'], dtype=self.range_dtype)
        hi = np.ascontiguousarray(records['hi'], dtype=self.range_dtype)

        starts, lens = self._group_starts_and_lens(t1.astype(np.uint64, copy=False))
        if len(starts):
            root_tokens = t1[starts].astype(np.intp, copy=False)
            self.root_start[root_tokens] = starts
            self.root_len[root_tokens] = lens

        ord1 = self._token_sort_array(t1).astype(np.uint64, copy=False)
        prefix_keys = np.ascontiguousarray((ord1 << self.token_bits) | t2_sorted.astype(np.uint64, copy=False))
        return t2_sorted, lo, hi, prefix_keys

    def _build_trigram_level(self, records, bigram_prefix_keys):
        keys = records['key']
        t1 = (keys & self.token_mask).astype(self.token_dtype, copy=False)
        t2 = ((keys >> self.token_bits) & self.token_mask).astype(self.token_dtype, copy=False)
        t3 = ((keys >> (2 * self.token_bits)) & self.token_mask).astype(self.token_dtype, copy=False)

        ord1 = self._token_sort_array(t1).astype(np.uint64, copy=False)
        ord2 = self._token_sort_array(t2).astype(np.uint64, copy=False)
        ord3 = self._token_sort_array(t3)
        prefix2 = np.ascontiguousarray((ord1 << self.token_bits) | ord2)

        group_starts, group_lens = self._group_starts_and_lens(prefix2)
        group_keys = prefix2[group_starts].astype(np.uint64, copy=False)
        parent_idx = np.searchsorted(bigram_prefix_keys, group_keys, side='left')
        if np.any(parent_idx >= len(bigram_prefix_keys)) or np.any(bigram_prefix_keys[parent_idx] != group_keys):
            raise ValueError("Trigram slab parent mapping did not align with bigram slab")

        child_start = np.zeros(self.bigram_count, dtype=self.offset_dtype)
        child_len = np.zeros(self.bigram_count, dtype=self.offset_dtype)
        child_start[parent_idx] = group_starts
        child_len[parent_idx] = group_lens

        trigram_tokens = np.ascontiguousarray(ord3, dtype=self.token_dtype)
        trigram_lo = np.ascontiguousarray(records['lo'], dtype=self.range_dtype)
        trigram_hi = np.ascontiguousarray(records['hi'], dtype=self.range_dtype)
        trigram_prefix_keys = np.ascontiguousarray((prefix2 << self.token_bits) | ord3.astype(np.uint64, copy=False))
        return trigram_tokens, trigram_lo, trigram_hi, child_start, child_len, trigram_prefix_keys

    def _build_quadgram_level(self, records, trigram_prefix_keys):
        keys = records['key']
        t1 = (keys & self.token_mask).astype(self.token_dtype, copy=False)
        t2 = ((keys >> self.token_bits) & self.token_mask).astype(self.token_dtype, copy=False)
        t3 = ((keys >> (2 * self.token_bits)) & self.token_mask).astype(self.token_dtype, copy=False)
        t4 = ((keys >> (3 * self.token_bits)) & self.token_mask).astype(self.token_dtype, copy=False)

        ord1 = self._token_sort_array(t1).astype(np.uint64, copy=False)
        ord2 = self._token_sort_array(t2).astype(np.uint64, copy=False)
        ord3 = self._token_sort_array(t3).astype(np.uint64, copy=False)
        ord4 = self._token_sort_array(t4)
        prefix3 = np.ascontiguousarray(
            (ord1 << (2 * self.token_bits))
            | (ord2 << self.token_bits)
            | ord3
        )

        group_starts, group_lens = self._group_starts_and_lens(prefix3)
        group_keys = prefix3[group_starts].astype(np.uint64, copy=False)
        parent_idx = np.searchsorted(trigram_prefix_keys, group_keys, side='left')
        if np.any(parent_idx >= len(trigram_prefix_keys)) or np.any(trigram_prefix_keys[parent_idx] != group_keys):
            raise ValueError("Quadgram slab parent mapping did not align with trigram slab")

        child_start = np.zeros(self.trigram_count, dtype=self.offset_dtype)
        child_len = np.zeros(self.trigram_count, dtype=self.offset_dtype)
        child_start[parent_idx] = group_starts
        child_len[parent_idx] = group_lens

        quad_tokens = np.ascontiguousarray(ord4, dtype=self.token_dtype)
        quad_lo = np.ascontiguousarray(records['lo'], dtype=self.range_dtype)
        quad_hi = np.ascontiguousarray(records['hi'], dtype=self.range_dtype)
        return quad_tokens, quad_lo, quad_hi, child_start, child_len

    def build_unigram_cache(self):
        cache = {}
        present = np.flatnonzero(self.root_len)
        for t1 in present:
            start = int(self.root_start[t1])
            count = int(self.root_len[t1])
            cache[int(t1)] = (int(self.bigram_lo[start]), int(self.bigram_hi[start + count - 1]))
        return cache

    def _lookup_child(self, child_keys, child_lo, child_hi, child_start, child_len, parent_idx, token_id):
        start = int(child_start[parent_idx])
        count = int(child_len[parent_idx])
        if count == 0:
            return None
        end = start + count
        sort_key = self._token_sort_key(token_id)
        idx = bisect.bisect_left(child_keys, sort_key, start, end)
        if idx >= end or int(child_keys[idx]) != sort_key:
            return None
        return int(child_lo[idx]), int(child_hi[idx]), idx

    def lookup_bigram_tokens(self, token_ids):
        a, b = int(token_ids[0]), int(token_ids[1])
        count = int(self.root_len[a])
        if count == 0:
            return None
        start = int(self.root_start[a])
        end = start + count
        sort_key = self._token_sort_key(b)
        idx = bisect.bisect_left(self.bigram_tokens, sort_key, start, end)
        if idx >= end or int(self.bigram_tokens[idx]) != sort_key:
            return None
        return int(self.bigram_lo[idx]), int(self.bigram_hi[idx]), idx

    def lookup_trigram_tokens(self, token_ids):
        if self.trigram_tokens is None:
            return None
        parent = self.lookup_bigram_tokens(token_ids[:2])
        if parent is None:
            return None
        return self._lookup_child(
            self.trigram_tokens,
            self.trigram_lo,
            self.trigram_hi,
            self.bigram_trigram_start,
            self.bigram_trigram_len,
            parent[2],
            token_ids[2],
        )

    def lookup_quadgram_tokens(self, token_ids):
        if self.quadgram_tokens is None:
            return None
        parent = self.lookup_trigram_tokens(token_ids[:3])
        if parent is None:
            return None
        return self._lookup_child(
            self.quadgram_tokens,
            self.quadgram_lo,
            self.quadgram_hi,
            self.trigram_quad_start,
            self.trigram_quad_len,
            parent[2],
            token_ids[3],
        )
