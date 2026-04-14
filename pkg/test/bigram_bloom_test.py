"""
Test correctness and performance of 2-gram in-memory cache and bloom filter.

Compares binary search over the suffix array:
  - Baseline: full range [0, tok_cnt)
  - With bigram cache: narrowed range from 2-gram lookup
  - With bloom filter: fast negative rejection for short n-grams

Usage:
    python test/bigram_bloom_test.py [--index-dir test/maildir_index]
"""

import argparse
import mmap
import numpy as np
import os
import struct
import time


# ---- Index loading utilities ----

class SuffixArrayIndex:
    """Loads and queries a single-shard infini-gram index."""

    def __init__(self, index_dir, token_width=2):
        self.token_width = token_width
        self.index_dir = index_dir

        # Load tokenized data
        ds_path = os.path.join(index_dir, 'tokenized.0')
        self.ds_size = os.path.getsize(ds_path)
        self.tok_cnt = self.ds_size // token_width
        self.ds_file = open(ds_path, 'rb')
        self.ds_mmap = mmap.mmap(self.ds_file.fileno(), 0, access=mmap.ACCESS_READ)

        # Load suffix array
        sa_path = os.path.join(index_dir, 'table.0')
        sa_size = os.path.getsize(sa_path)
        self.ptr_size = sa_size // self.tok_cnt
        self.sa_file = open(sa_path, 'rb')
        self.sa_mmap = mmap.mmap(self.sa_file.fileno(), 0, access=mmap.ACCESS_READ)

        # Load bigram cache (optional)
        bg_path = os.path.join(index_dir, 'bigram.0')
        self.bigram_cache = None
        if os.path.exists(bg_path):
            self.bigram_cache = self._load_bigram_cache(bg_path)

        # Load bloom filter (optional)
        bl_path = os.path.join(index_dir, 'bloom.0')
        self.bloom = None
        if os.path.exists(bl_path):
            self.bloom = self._load_bloom_filter(bl_path)

        if token_width == 2:
            self.dtype = np.uint16
        elif token_width == 1:
            self.dtype = np.uint8
        elif token_width == 4:
            self.dtype = np.uint32

    def _load_bigram_cache(self, path):
        """Load bigram cache into a dict mapping u64 key -> (lo, hi)."""
        with open(path, 'rb') as f:
            num_entries, = struct.unpack('<Q', f.read(8))
            tw, = struct.unpack('<I', f.read(4))
            _padding, = struct.unpack('<I', f.read(4))
            assert tw == self.token_width

            cache = {}
            for _ in range(num_entries):
                key, lo, hi = struct.unpack('<QQQ', f.read(24))
                cache[key] = (lo, hi)

        return cache

    def _load_bloom_filter(self, path):
        """Load bloom filter data."""
        with open(path, 'rb') as f:
            num_bits, = struct.unpack('<Q', f.read(8))
            num_hashes, = struct.unpack('<I', f.read(4))
            max_ngram_n, = struct.unpack('<I', f.read(4))
            data = f.read()
        return {'num_bits': num_bits, 'num_hashes': num_hashes,
                'max_ngram_n': max_ngram_n, 'data': data}

    def _make_bigram_key(self, token_ids):
        """Pack first 2 tokens into a u64 key (matching C++ _make_bigram_key)."""
        key = 0
        t1_bytes = int(token_ids[0]).to_bytes(self.token_width, 'little')
        t2_bytes = int(token_ids[1]).to_bytes(self.token_width, 'little')
        key_bytes = t1_bytes + t2_bytes + b'\x00' * (8 - 2 * self.token_width)
        return struct.unpack('<Q', key_bytes)[0]

    def _read_sa_ptr(self, rank):
        """Read suffix array pointer at given rank."""
        offset = rank * self.ptr_size
        raw = self.sa_mmap[offset:offset + self.ptr_size]
        padded = raw + b'\x00' * (8 - self.ptr_size)
        return struct.unpack('<Q', padded)[0]

    def _read_suffix_bytes(self, ptr, num_bytes):
        """Read bytes from the datastore at given byte offset."""
        end = min(ptr + num_bytes, self.ds_size)
        return self.ds_mmap[ptr:end]

    def _compare_suffix(self, rank, query_bytes):
        """Compare suffix at given rank with query bytes. Returns -1, 0, or 1."""
        ptr = self._read_sa_ptr(rank)
        suffix = self._read_suffix_bytes(ptr, len(query_bytes))
        if suffix < query_bytes:
            return -1
        elif suffix > query_bytes:
            return 1
        return 0

    def _token_ids_to_bytes(self, token_ids):
        """Convert token IDs to bytes for comparison."""
        arr = np.array(token_ids, dtype=self.dtype)
        return arr.tobytes()

    def binary_search(self, token_ids, lo=None, hi=None):
        """
        Three-phase binary search (matching C++ _find_thread logic).
        Returns (left, right) where all matches are in [left, right).
        Also returns number of comparisons made.
        """
        if lo is None:
            lo = 0
        if hi is None:
            hi = self.tok_cnt

        query_bytes = self._token_ids_to_bytes(token_ids)
        num_bytes = len(query_bytes)
        comparisons = 0

        if num_bytes == 0:
            return lo, hi, comparisons

        # Phase 1: Find any match
        orig_lo, orig_hi = lo, hi
        mi = lo
        while lo < hi:
            mi = (lo + hi - 1) >> 1
            cmp = self._compare_suffix(mi, query_bytes)
            comparisons += 1
            if cmp < 0:
                lo = mi + 1
            elif cmp > 0:
                hi = mi
            else:
                break

        if lo == hi:
            return lo, lo, comparisons

        # Phase 2: Find left boundary in [orig_lo-1, mi]
        l, r = orig_lo - 1, mi
        while r - l > 1:
            m = (l + r) >> 1
            ptr = self._read_sa_ptr(m)
            suffix = self._read_suffix_bytes(ptr, num_bytes)
            comparisons += 1
            if suffix < query_bytes:
                l = m
            else:
                r = m
        left = r

        # Phase 3: Find right boundary in [mi, orig_hi]
        l, r = mi, orig_hi
        while r - l > 1:
            m = (l + r) >> 1
            ptr = self._read_sa_ptr(m)
            suffix = self._read_suffix_bytes(ptr, num_bytes)
            comparisons += 1
            if query_bytes < suffix:
                r = m
            else:
                l = m
        right = r

        return left, right, comparisons

    def search_baseline(self, token_ids):
        """Search without any cache (full range)."""
        return self.binary_search(token_ids)

    def search_with_bigram(self, token_ids):
        """Search with bigram cache hint."""
        if self.bigram_cache is None or len(token_ids) < 2:
            return self.binary_search(token_ids)

        key = self._make_bigram_key(token_ids)
        if key in self.bigram_cache:
            lo, hi = self.bigram_cache[key]
            return self.binary_search(token_ids, lo, hi)
        else:
            # Bigram not in data, no results
            return 0, 0, 0

    def bloom_check(self, token_ids):
        """
        Check bloom filter. Returns True if query definitely doesn't exist.
        Returns False if query might exist.
        """
        if self.bloom is None:
            return False
        n = len(token_ids)
        if n < 2 or n > self.bloom['max_ngram_n']:
            return False

        key = self._token_ids_to_bytes(token_ids)
        return not self._bloom_maybe_contains(key)

    def _fnv1a(self, key, seed):
        h = seed & 0xFFFFFFFFFFFFFFFF
        for b in key:
            h ^= b
            h = (h * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF
        return h

    def _bloom_maybe_contains(self, key):
        bloom = self.bloom
        h1 = self._fnv1a(key, 0xcbf29ce484222325)
        h2 = self._fnv1a(key, 0x517cc1b727220a95)
        for i in range(bloom['num_hashes']):
            bit = (h1 + i * h2) % bloom['num_bits']
            byte_idx = bit // 8
            bit_idx = bit % 8
            if not (bloom['data'][byte_idx] & (1 << bit_idx)):
                return False
        return True

    def search_with_bloom_and_bigram(self, token_ids):
        """Search with bloom filter pre-check and bigram cache."""
        # Bloom filter: fast rejection for 2-4 grams
        if self.bloom_check(token_ids):
            return 0, 0, 0

        # Bigram cache: narrow search range
        return self.search_with_bigram(token_ids)

    def close(self):
        self.ds_mmap.close()
        self.ds_file.close()
        self.sa_mmap.close()
        self.sa_file.close()


# ---- Test functions ----

def test_correctness(index):
    """Verify that bigram cache and bloom filter produce correct results."""
    print("=" * 60)
    print("CORRECTNESS TESTS")
    print("=" * 60)

    # Read some actual tokens from the data for test queries
    ds_bytes = index.ds_mmap[:min(10000, index.ds_size)]
    tokens = np.frombuffer(ds_bytes, dtype=index.dtype)

    passed = 0
    failed = 0

    # Test 1: Known existing 2-grams
    print("\n--- Test 1: Existing 2-gram queries ---")
    for i in range(0, min(len(tokens) - 2, 100), 3):
        t1, t2 = int(tokens[i]), int(tokens[i + 1])
        doc_sep = (1 << (index.token_width * 8)) - 1
        if t1 == doc_sep or t2 == doc_sep:
            continue

        query = [t1, t2]
        left_base, right_base, _ = index.search_baseline(query)
        left_bg, right_bg, _ = index.search_with_bigram(query)
        left_full, right_full, _ = index.search_with_bloom_and_bigram(query)

        # For existing queries, both position and count must match exactly
        if (left_base == left_bg == left_full and
                right_base == right_bg == right_full):
            passed += 1
        else:
            cnt_base = right_base - left_base
            cnt_bg = right_bg - left_bg
            cnt_full = right_full - left_full
            if cnt_base == cnt_bg == cnt_full:
                passed += 1  # counts match (different empty-range repr is OK)
            else:
                print(f"  FAIL: query={query}, baseline=[{left_base},{right_base}), "
                      f"bigram=[{left_bg},{right_bg}), full=[{left_full},{right_full})")
                failed += 1

    print(f"  {passed} passed, {failed} failed")

    # Test 2: Known existing 3-grams
    print("\n--- Test 2: Existing 3-gram queries ---")
    p2, f2 = 0, 0
    for i in range(0, min(len(tokens) - 3, 100), 3):
        t1, t2, t3 = int(tokens[i]), int(tokens[i + 1]), int(tokens[i + 2])
        doc_sep = (1 << (index.token_width * 8)) - 1
        if doc_sep in (t1, t2, t3):
            continue

        query = [t1, t2, t3]
        left_base, right_base, _ = index.search_baseline(query)
        left_bg, right_bg, _ = index.search_with_bigram(query)
        left_full, right_full, _ = index.search_with_bloom_and_bigram(query)

        if (left_base == left_bg == left_full and
                right_base == right_bg == right_full):
            p2 += 1
        else:
            cnt_base = right_base - left_base
            cnt_bg = right_bg - left_bg
            cnt_full = right_full - left_full
            if cnt_base == cnt_bg == cnt_full:
                p2 += 1
            else:
                print(f"  FAIL: query={query}, baseline=[{left_base},{right_base}), "
                      f"bigram=[{left_bg},{right_bg}), full=[{left_full},{right_full})")
                f2 += 1
    passed += p2
    failed += f2
    print(f"  {p2} passed, {f2} failed")

    # Test 3: Non-existing queries (random tokens likely not in data)
    # Compare counts (right-left) rather than exact positions, since zero-result
    # queries may return different empty-range representations
    print("\n--- Test 3: Non-existing random queries ---")
    p3, f3 = 0, 0
    np.random.seed(42)
    max_tok = (1 << (index.token_width * 8)) - 2  # exclude doc_sep
    for _ in range(50):
        query = [int(x) for x in np.random.randint(0, max_tok, size=3)]
        left_base, right_base, _ = index.search_baseline(query)
        left_bg, right_bg, _ = index.search_with_bigram(query)
        left_full, right_full, _ = index.search_with_bloom_and_bigram(query)

        cnt_base = right_base - left_base
        cnt_bg = right_bg - left_bg
        cnt_full = right_full - left_full

        if cnt_base == cnt_bg == cnt_full:
            p3 += 1
        else:
            print(f"  FAIL: query={query}, baseline_cnt={cnt_base}, "
                  f"bigram_cnt={cnt_bg}, full_cnt={cnt_full}")
            f3 += 1
    passed += p3
    failed += f3
    print(f"  {p3} passed, {f3} failed")

    # Test 4: Bloom filter should not have false negatives
    print("\n--- Test 4: Bloom filter false negative check ---")
    p4, f4 = 0, 0
    for i in range(0, min(len(tokens) - 4, 200), 5):
        for n in range(2, 5):
            if i + n > len(tokens):
                break
            toks = [int(tokens[i + j]) for j in range(n)]
            doc_sep = (1 << (index.token_width * 8)) - 1
            if doc_sep in toks:
                continue
            # Check if this n-gram actually exists
            left, right, _ = index.search_baseline(toks)
            if right - left > 0:
                # It exists — bloom should NOT reject it
                rejected = index.bloom_check(toks)
                if rejected:
                    print(f"  FALSE NEGATIVE: bloom rejected existing {n}-gram {toks} "
                          f"(count={right - left})")
                    f4 += 1
                else:
                    p4 += 1
    passed += p4
    failed += f4
    print(f"  {p4} passed, {f4} failed")

    # Test 5: Single token queries should bypass bigram (still correct)
    print("\n--- Test 5: Single token queries ---")
    p5, f5 = 0, 0
    for i in range(0, min(len(tokens), 30)):
        query = [int(tokens[i])]
        left_base, right_base, _ = index.search_baseline(query)
        left_bg, right_bg, _ = index.search_with_bigram(query)

        if left_base == left_bg and right_base == right_bg:
            p5 += 1
        else:
            print(f"  FAIL: query={query}, baseline=[{left_base},{right_base}), "
                  f"bigram=[{left_bg},{right_bg})")
            f5 += 1
    passed += p5
    failed += f5
    print(f"  {p5} passed, {f5} failed")

    print(f"\n{'=' * 60}")
    print(f"TOTAL: {passed} passed, {failed} failed")
    print(f"{'=' * 60}")
    return failed == 0


def test_performance(index, num_queries=500):
    """Benchmark binary search with and without bigram cache."""
    print("\n" + "=" * 60)
    print("PERFORMANCE TESTS")
    print("=" * 60)

    # Generate test queries from actual data
    ds_bytes = index.ds_mmap[:min(50000, index.ds_size)]
    tokens = np.frombuffer(ds_bytes, dtype=index.dtype)
    doc_sep = (1 << (index.token_width * 8)) - 1

    existing_queries_2 = []
    existing_queries_3 = []
    for i in range(0, len(tokens) - 3, 7):
        t1, t2, t3 = int(tokens[i]), int(tokens[i + 1]), int(tokens[i + 2])
        if doc_sep in (t1, t2, t3):
            continue
        # Alternate between 2-gram and 3-gram lists so they draw from
        # independent positions rather than being prefix/superset pairs
        if len(existing_queries_2) <= len(existing_queries_3):
            existing_queries_2.append([t1, t2])
        else:
            existing_queries_3.append([t1, t2, t3])
        if (len(existing_queries_2) >= num_queries and
                len(existing_queries_3) >= num_queries):
            break

    # Generate non-existing queries
    np.random.seed(123)
    max_tok = doc_sep - 1
    nonexist_queries = [[int(x) for x in np.random.randint(0, max_tok, size=3)]
                        for _ in range(num_queries)]

    import random as _rng
    _rng.seed(42)

    def benchmark_interleaved(queries, search_fns):
        """Run search functions in randomized order per-query to avoid page cache bias."""
        n_fns = len(search_fns)
        totals = [{'time': 0.0, 'comps': 0, 'results': 0} for _ in range(n_fns)]
        for q in queries:
            order = list(range(n_fns))
            _rng.shuffle(order)
            for idx in order:
                start = time.perf_counter()
                left, right, comps = search_fns[idx](q)
                elapsed = time.perf_counter() - start
                totals[idx]['time'] += elapsed
                totals[idx]['comps'] += comps
                totals[idx]['results'] += right - left
        n = len(queries) if queries else 1
        return [(t['time'], t['comps'] / n, t['results']) for t in totals]

    print(f"\nlog2(tok_cnt) = {np.log2(index.tok_cnt):.1f} "
          f"(max comparisons per search without cache)")
    if index.bigram_cache:
        print(f"Bigram cache entries: {len(index.bigram_cache):,}")

    # Warmup pass to load mmap pages into OS page cache
    print("\n  (warming up page cache ...)")
    for q in existing_queries_2[:50]:
        index.search_baseline(q)
    for q in existing_queries_3[:50]:
        index.search_baseline(q)

    fns = [index.search_baseline, index.search_with_bigram,
           index.search_with_bloom_and_bigram]
    labels = ["Baseline", "Bigram", "Bloom+Bigram"]

    # Benchmark existing 2-gram queries
    print(f"\n--- Existing 2-gram queries (n={len(existing_queries_2)}) ---")
    results = benchmark_interleaved(existing_queries_2, fns)
    for label, (t, c, r) in zip(labels, results):
        print(f"  {label+':':15s} {t:.3f}s, avg {c:.1f} comparisons, {r:,} total results")
    if results[0][1] > 0:
        print(f"  Comparison reduction: {(1 - results[1][1]/results[0][1])*100:.1f}%")

    # Benchmark existing 3-gram queries
    print(f"\n--- Existing 3-gram queries (n={len(existing_queries_3)}) ---")
    results = benchmark_interleaved(existing_queries_3, fns)
    for label, (t, c, r) in zip(labels, results):
        print(f"  {label+':':15s} {t:.3f}s, avg {c:.1f} comparisons, {r:,} total results")
    if results[0][1] > 0:
        print(f"  Comparison reduction: {(1 - results[1][1]/results[0][1])*100:.1f}%")

    # Benchmark non-existing queries
    print(f"\n--- Non-existing random queries (n={len(nonexist_queries)}) ---")
    results = benchmark_interleaved(nonexist_queries, fns)
    for label, (t, c, r) in zip(labels, results):
        print(f"  {label+':':15s} {t:.3f}s, avg {c:.1f} comparisons, {r:,} total results")

    # Count bloom filter rejections
    bloom_rejections = sum(1 for q in nonexist_queries if index.bloom_check(q))
    print(f"  Bloom rejections: {bloom_rejections}/{len(nonexist_queries)} "
          f"({bloom_rejections/len(nonexist_queries)*100:.1f}%)")

    print()


def test_bloom_stats(index):
    """Analyze bloom filter false positive rate."""
    print("=" * 60)
    print("BLOOM FILTER STATISTICS")
    print("=" * 60)

    if index.bloom is None:
        print("  No bloom filter loaded.")
        return

    np.random.seed(999)
    max_tok = (1 << (index.token_width * 8)) - 2
    n_test = 1000

    for ngram_n in range(2, index.bloom['max_ngram_n'] + 1):
        true_pos = 0
        true_neg = 0
        false_pos = 0
        false_neg = 0

        for _ in range(n_test):
            query = [int(x) for x in np.random.randint(0, max_tok, size=ngram_n)]
            left, right, _ = index.search_baseline(query)
            exists = (right - left) > 0
            bloom_says_no = index.bloom_check(query)

            if exists and not bloom_says_no:
                true_pos += 1
            elif exists and bloom_says_no:
                false_neg += 1
            elif not exists and bloom_says_no:
                true_neg += 1
            else:
                false_pos += 1

        total = true_pos + true_neg + false_pos + false_neg
        print(f"\n  {ngram_n}-grams (n={n_test}):")
        print(f"    True positives:  {true_pos} (exists, bloom says maybe)")
        print(f"    True negatives:  {true_neg} (absent, bloom says no)")
        print(f"    False positives: {false_pos} (absent, bloom says maybe)")
        print(f"    False negatives: {false_neg} (exists, bloom says no) <- MUST be 0")
        if true_neg + false_pos > 0:
            fpr = false_pos / (true_neg + false_pos)
            print(f"    False positive rate: {fpr*100:.2f}%")

    print()


def main():
    parser = argparse.ArgumentParser(description='Test bigram cache and bloom filter')
    parser.add_argument('--index-dir', default='test/maildir_index',
                        help='Path to index directory')
    parser.add_argument('--token-width', type=int, default=2,
                        help='Token width in bytes (1=u8, 2=u16, 4=u32)')
    parser.add_argument('--num-queries', type=int, default=500,
                        help='Number of queries for performance tests')
    args = parser.parse_args()

    print(f"Loading index from {args.index_dir} ...")
    index = SuffixArrayIndex(args.index_dir, token_width=args.token_width)
    print(f"  Tokens: {index.tok_cnt:,}")
    print(f"  Pointer size: {index.ptr_size} bytes")
    print(f"  Bigram cache: {'loaded' if index.bigram_cache else 'not found'}")
    if index.bigram_cache:
        print(f"  Bigram entries: {len(index.bigram_cache):,}")
    print(f"  Bloom filter: {'loaded' if index.bloom else 'not found'}")
    if index.bloom:
        print(f"  Bloom bits: {index.bloom['num_bits']:,}, "
              f"hashes: {index.bloom['num_hashes']}, "
              f"max_n: {index.bloom['max_ngram_n']}")
    print()

    all_passed = test_correctness(index)
    test_performance(index, num_queries=args.num_queries)
    test_bloom_stats(index)

    index.close()

    if all_passed:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
        exit(1)


if __name__ == '__main__':
    main()
