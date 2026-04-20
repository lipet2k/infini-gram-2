"""
Benchmark exact and approximate suffix-array query methods with packed n-gram hints.

Usage:
    python test/ngram_query_benchmark.py [--index-dir test/maildir_index]
"""

import argparse
import os
import random as _random
import sys
import time

import numpy as np

from suffix_array_query_index import SuffixArrayIndex


def test_correctness(index):
    """Verify that the cache-backed query paths produce correct results."""
    print("=" * 60)
    print("CORRECTNESS TESTS")
    print("=" * 60)

    ds_bytes = index.ds_mmap[:min(10000, index.ds_size)]
    tokens = np.frombuffer(ds_bytes, dtype=index.dtype)

    def exact_range(method, query):
        return method(query)[:2]

    passed = 0
    failed = 0

    print("\n--- Test 1: Existing 2-gram queries ---")
    for i in range(0, min(len(tokens) - 2, 100), 3):
        t1, t2 = int(tokens[i]), int(tokens[i + 1])
        doc_sep = (1 << (index.token_width * 8)) - 1
        if t1 == doc_sep or t2 == doc_sep:
            continue

        query = [t1, t2]
        cnt_base = index.search_baseline(query)[1] - index.search_baseline(query)[0]
        cnts = {
            'bigram': index.search_with_bigram(query),
            'trigram': index.search_with_trigram(query),
            'quadgram': index.search_with_quadgram(query),
            'adaptive': index.search_with_adaptive(query),
            '1approx2': index.search_with_1approx2(query),
        }
        if index.lisa is not None:
            cnts['lisa(bin)'] = index.search_with_lisa_binary(query)
            cnts['lisa(rmi)'] = index.search_with_lisa_rmi(query)
        all_match = all((res[1] - res[0]) == cnt_base for res in cnts.values())
        if all_match:
            passed += 1
        else:
            diffs = {name: res[1] - res[0] for name, res in cnts.items() if (res[1] - res[0]) != cnt_base}
            print(f"  FAIL: query={query}, baseline={cnt_base}, {diffs}")
            failed += 1
    print(f"  {passed} passed, {failed} failed")

    print("\n--- Test 2: Existing 3-gram queries ---")
    p2, f2 = 0, 0
    for i in range(0, min(len(tokens) - 3, 100), 3):
        t1, t2, t3 = int(tokens[i]), int(tokens[i + 1]), int(tokens[i + 2])
        doc_sep = (1 << (index.token_width * 8)) - 1
        if doc_sep in (t1, t2, t3):
            continue

        query = [t1, t2, t3]
        cnt_base = index.search_baseline(query)[1] - index.search_baseline(query)[0]
        cnts = {
            'bigram': index.search_with_bigram(query),
            'trigram': index.search_with_trigram(query),
            'quadgram': index.search_with_quadgram(query),
            'adaptive': index.search_with_adaptive(query),
            '2approx3': index.search_with_2approx3(query),
        }
        if index.lisa is not None:
            cnts['lisa(bin)'] = index.search_with_lisa_binary(query)
            cnts['lisa(rmi)'] = index.search_with_lisa_rmi(query)
        all_match = all((res[1] - res[0]) == cnt_base for res in cnts.values())
        if all_match:
            p2 += 1
        else:
            diffs = {name: res[1] - res[0] for name, res in cnts.items() if (res[1] - res[0]) != cnt_base}
            print(f"  FAIL: query={query}, baseline={cnt_base}, {diffs}")
            f2 += 1
    passed += p2
    failed += f2
    print(f"  {p2} passed, {f2} failed")

    print("\n--- Test 2b: Existing 4-gram queries ---")
    p2b, f2b = 0, 0
    for i in range(0, min(len(tokens) - 4, 100), 3):
        query = [int(tokens[i + j]) for j in range(4)]
        doc_sep = (1 << (index.token_width * 8)) - 1
        if doc_sep in query:
            continue

        cnt_base = index.search_baseline(query)[1] - index.search_baseline(query)[0]
        cnts = {
            'bigram': index.search_with_bigram(query),
            'trigram': index.search_with_trigram(query),
            'quadgram': index.search_with_quadgram(query),
            'adaptive': index.search_with_adaptive(query),
        }
        if index.lisa is not None:
            cnts['lisa(bin)'] = index.search_with_lisa_binary(query)
            cnts['lisa(rmi)'] = index.search_with_lisa_rmi(query)
        all_match = all((res[1] - res[0]) == cnt_base for res in cnts.values())
        if all_match:
            p2b += 1
        else:
            diffs = {name: res[1] - res[0] for name, res in cnts.items() if (res[1] - res[0]) != cnt_base}
            print(f"  FAIL: query={query}, baseline={cnt_base}, {diffs}")
            f2b += 1
    passed += p2b
    failed += f2b
    print(f"  {p2b} passed, {f2b} failed")

    print("\n--- Test 3: Non-existing random queries ---")
    p3, f3 = 0, 0
    np.random.seed(42)
    max_tok = (1 << (index.token_width * 8)) - 2
    for _ in range(50):
        query = [int(value) for value in np.random.randint(0, max_tok, size=3)]
        cnt_base = index.search_baseline(query)[1] - index.search_baseline(query)[0]
        cnts = {
            'bigram': index.search_with_bigram(query),
            'trigram': index.search_with_trigram(query),
            'quadgram': index.search_with_quadgram(query),
            'adaptive': index.search_with_adaptive(query),
            '2approx3': index.search_with_2approx3(query),
        }
        if index.lisa is not None:
            cnts['lisa(bin)'] = index.search_with_lisa_binary(query)
            cnts['lisa(rmi)'] = index.search_with_lisa_rmi(query)
        q2 = query[:2]
        cnt_base_2 = index.search_baseline(q2)[1] - index.search_baseline(q2)[0]
        r_1a2 = index.search_with_1approx2(q2)
        all_match = (
            all((res[1] - res[0]) == cnt_base for res in cnts.values())
            and (r_1a2[1] - r_1a2[0]) == cnt_base_2
        )
        if all_match:
            p3 += 1
        else:
            diffs = {name: res[1] - res[0] for name, res in cnts.items() if (res[1] - res[0]) != cnt_base}
            if (r_1a2[1] - r_1a2[0]) != cnt_base_2:
                diffs['1approx2'] = (r_1a2[1] - r_1a2[0], f'base2={cnt_base_2}')
            print(f"  FAIL: query={query}, baseline={cnt_base}, {diffs}")
            f3 += 1
    passed += p3
    failed += f3
    print(f"  {p3} passed, {f3} failed")

    print("\n--- Test 4: Single token queries ---")
    p4, f4 = 0, 0
    for i in range(0, min(len(tokens), 30)):
        query = [int(tokens[i])]
        left_base, right_base, _ = index.search_baseline(query)
        left_bg, right_bg, _ = index.search_with_bigram(query)
        if index.lisa is not None:
            left_lisa, right_lisa, _ = index.search_with_lisa_rmi(query)
            lisa_ok = (left_base == left_lisa and right_base == right_lisa)
        else:
            lisa_ok = True

        if left_base == left_bg and right_base == right_bg and lisa_ok:
            p4 += 1
        else:
            print(f"  FAIL: query={query}, baseline=[{left_base},{right_base}), bigram=[{left_bg},{right_bg})")
            if index.lisa is not None and not lisa_ok:
                print(f"        lisa=[{left_lisa},{right_lisa})")
            f4 += 1
    passed += p4
    failed += f4
    print(f"  {p4} passed, {f4} failed")

    print("\n--- Test 5: Approx-search exact-range check ---")
    p5, f5 = 0, 0
    for i in range(0, min(len(tokens) - 3, 80), 3):
        t1, t2, t3 = int(tokens[i]), int(tokens[i + 1]), int(tokens[i + 2])
        doc_sep = (1 << (index.token_width * 8)) - 1
        if doc_sep in (t1, t2, t3):
            continue

        q2 = [t1, t2]
        lb, rb, _ = index.search_baseline(q2)
        la, ra, _ = index.search_with_1approx2(q2)
        if (lb, rb) == (la, ra):
            p5 += 1
        else:
            print(f"  FAIL 1approx2: query={q2}, baseline=[{lb},{rb}), got=[{la},{ra})")
            f5 += 1

        q3 = [t1, t2, t3]
        lb, rb, _ = index.search_baseline(q3)
        la, ra, _ = index.search_with_2approx3(q3)
        if (lb, rb) == (la, ra):
            p5 += 1
        else:
            print(f"  FAIL 2approx3: query={q3}, baseline=[{lb},{rb}), got=[{la},{ra})")
            f5 += 1
    passed += p5
    failed += f5
    print(f"  {p5} passed, {f5} failed")

    print("\n--- Test 6: LISA exact-range check ---")
    p6, f6 = 0, 0
    if index.lisa is not None:
        query_lengths = set(range(1, 7))
        k_tokens = index.lisa.k_tokens
        for base in (k_tokens, 2 * k_tokens, 3 * k_tokens):
            for delta in (-1, 0, 1):
                ngram_n = base + delta
                if ngram_n > 0:
                    query_lengths.add(ngram_n)

        max_test_n = min(max(query_lengths), len(tokens) - 1)
        query_lengths = sorted(ngram_n for ngram_n in query_lengths if ngram_n <= max_test_n)

        for ngram_n in query_lengths:
            seen = 0
            starts = list(range(0, min(len(tokens) - ngram_n, 300), 7))
            tail_limit = min(10, max(0, len(tokens) - ngram_n))
            for delta in range(tail_limit):
                starts.append(max(0, len(tokens) - ngram_n - 1 - delta))

            seen_starts = set()
            for start in starts:
                if start in seen_starts or start + ngram_n > len(tokens):
                    continue
                seen_starts.add(start)
                query = [int(tokens[start + j]) for j in range(ngram_n)]
                doc_sep = (1 << (index.token_width * 8)) - 1
                if doc_sep in query:
                    continue
                baseline = exact_range(index.search_baseline, query)
                lisa_bin = exact_range(index.search_with_lisa_binary, query)
                lisa_rmi = exact_range(index.search_with_lisa_rmi, query)
                if baseline == lisa_bin == lisa_rmi:
                    p6 += 1
                else:
                    print(f"  FAIL existing {ngram_n}-gram: query={query}, baseline={baseline}, lisa(bin)={lisa_bin}, lisa(rmi)={lisa_rmi}")
                    f6 += 1
                seen += 1
                if seen >= 24:
                    break

        np.random.seed(314159)
        for ngram_n in query_lengths:
            for _ in range(20):
                query = [int(value) for value in np.random.randint(0, max_tok, size=ngram_n)]
                baseline = exact_range(index.search_baseline, query)
                lisa_bin = exact_range(index.search_with_lisa_binary, query)
                lisa_rmi = exact_range(index.search_with_lisa_rmi, query)
                if baseline == lisa_bin == lisa_rmi:
                    p6 += 1
                else:
                    print(f"  FAIL random {ngram_n}-gram: query={query}, baseline={baseline}, lisa(bin)={lisa_bin}, lisa(rmi)={lisa_rmi}")
                    f6 += 1
    passed += p6
    failed += f6
    print(f"  {p6} passed, {f6} failed")

    print(f"\n{'=' * 60}")
    print(f"TOTAL: {passed} passed, {failed} failed")
    print(f"{'=' * 60}")
    return failed == 0


def test_performance(index, num_queries=500):
    """Benchmark binary search with and without n-gram cache hints."""
    print("\n" + "=" * 60)
    print("PERFORMANCE TESTS")
    print("=" * 60)

    query_stride = 11
    min_read_size = max(100000, num_queries * index.token_width)
    read_size = min(min_read_size, index.ds_size)
    doc_sep = (1 << (index.token_width * 8)) - 1

    while True:
        ds_bytes = index.ds_mmap[:read_size]
        tokens = np.frombuffer(ds_bytes, dtype=index.dtype)

        existing_queries_2 = []
        existing_queries_3 = []
        existing_queries_4 = []
        query_lists = [existing_queries_2, existing_queries_3, existing_queries_4]
        q_idx = 0
        for i in range(0, len(tokens) - 4, query_stride):
            toks = [int(tokens[i + j]) for j in range(4)]
            if doc_sep in toks:
                continue
            ngram_n = q_idx % 3
            query_lists[ngram_n].append(toks[:ngram_n + 2])
            q_idx += 1
            if all(len(query_list) >= num_queries for query_list in query_lists):
                break

        if all(len(query_list) >= num_queries for query_list in query_lists) or read_size == index.ds_size:
            break

        next_read_size = max(
            read_size * 2,
            read_size + (num_queries * 3 * query_stride * index.token_width),
        )
        read_size = min(next_read_size, index.ds_size)

    if not all(len(query_list) >= num_queries for query_list in query_lists):
        print(
            f"  Note: only generated {len(existing_queries_2):,}/{len(existing_queries_3):,}/{len(existing_queries_4):,} "
            "existing 2/3/4-gram queries from the sampled data"
        )

    np.random.seed(123)
    max_tok = doc_sep - 1
    nonexist_queries = [
        [int(value) for value in np.random.randint(0, max_tok, size=4)]
        for _ in range(num_queries)
    ]

    _random.seed(42)

    def benchmark_interleaved(queries, search_fns):
        totals = [{'time': 0.0, 'lookup': 0.0, 'refine': 0.0, 'comps': 0, 'results': 0}
                  for _ in search_fns]
        for query in queries:
            order = list(range(len(search_fns)))
            _random.shuffle(order)
            for idx in order:
                index._perf_begin_call()
                start = time.perf_counter()
                left, right, comps = search_fns[idx](query)
                elapsed = time.perf_counter() - start
                refine = index._perf_end_call()
                lookup = max(0.0, elapsed - refine)
                totals[idx]['time'] += elapsed
                totals[idx]['lookup'] += lookup
                totals[idx]['refine'] += refine
                totals[idx]['comps'] += comps
                totals[idx]['results'] += right - left
        n_queries = len(queries) if queries else 1
        return [
            (total['time'], total['lookup'], total['refine'], total['comps'] / n_queries, total['results'])
            for total in totals
        ]

    print(f"\nlog2(tok_cnt) = {np.log2(index.tok_cnt):.1f} (max comparisons per search without cache)")

    def fmt_size(nbytes):
        if nbytes >= 1 << 30:
            return f"{nbytes / (1 << 30):.2f} GB"
        if nbytes >= 1 << 20:
            return f"{nbytes / (1 << 20):.1f} MB"
        if nbytes >= 1 << 10:
            return f"{nbytes / (1 << 10):.1f} KB"
        return f"{nbytes} B"

    def dict_mem_size(mapping):
        if mapping is None:
            return 0
        return sys.getsizeof(mapping) + len(mapping) * (
            sys.getsizeof(0)
            + sys.getsizeof((0, 0))
            + 2 * sys.getsizeof(0)
        )

    def cache_mem_size(cache):
        if cache is None:
            return 0
        if hasattr(cache, 'mem_bytes_estimate'):
            return cache.mem_bytes_estimate
        return dict_mem_size(cache)

    print(f"\n{'Data structure':<20s} {'Entries':>12s} {'On-disk':>12s} {'In-memory':>12s}")
    print(f"{'-'*20} {'-'*12} {'-'*12} {'-'*12}")

    sa_disk = index.tok_cnt * index.ptr_size
    print(f"{'Suffix array':<20s} {index.tok_cnt:>12,} {fmt_size(sa_disk):>12s} {fmt_size(sa_disk):>12s}")

    ds_disk = index.ds_size
    print(f"{'Tokenized data':<20s} {index.tok_cnt:>12,} {fmt_size(ds_disk):>12s} {fmt_size(ds_disk):>12s}")

    if index.unigram_cache:
        ug_mem = dict_mem_size(index.unigram_cache)
        print(f"{'Unigram cache':<20s} {len(index.unigram_cache):>12,} {'(derived)':>12s} {fmt_size(ug_mem):>12s}")

    if index.bigram_cache:
        bg_disk_path = os.path.join(index.index_dir, 'bigram.0')
        bg_disk = os.path.getsize(bg_disk_path) if os.path.exists(bg_disk_path) else 0
        bg_mem = cache_mem_size(index.bigram_cache)
        print(f"{'Bigram cache':<20s} {len(index.bigram_cache):>12,} {fmt_size(bg_disk):>12s} {fmt_size(bg_mem):>12s}")

    if index.trigram_cache:
        tg_disk_path = os.path.join(index.index_dir, 'trigram.0')
        tg_disk = os.path.getsize(tg_disk_path) if os.path.exists(tg_disk_path) else 0
        tg_mem = cache_mem_size(index.trigram_cache)
        print(f"{'Trigram cache':<20s} {len(index.trigram_cache):>12,} {fmt_size(tg_disk):>12s} {fmt_size(tg_mem):>12s}")

    if index.quadgram_cache:
        qg_disk_path = os.path.join(index.index_dir, 'quadgram.0')
        qg_disk = os.path.getsize(qg_disk_path) if os.path.exists(qg_disk_path) else 0
        qg_mem = cache_mem_size(index.quadgram_cache)
        print(f"{'Quadgram cache':<20s} {len(index.quadgram_cache):>12,} {fmt_size(qg_disk):>12s} {fmt_size(qg_mem):>12s}")

    if index.adaptive_cache:
        ad_disk_path = os.path.join(index.index_dir, 'adaptive_ngram.0')
        ad_disk = os.path.getsize(ad_disk_path) if os.path.exists(ad_disk_path) else 0
        print(f"{'Adaptive cache':<20s} {index.adaptive_cache.total_entries:>12,} {fmt_size(ad_disk):>12s} {fmt_size(index.adaptive_cache.mem_bytes_estimate):>12s}")
        for ngram_n, cache in index.adaptive_cache.level_items():
            print(f"  {'level '+str(ngram_n):<18s} {len(cache):>12,}")

    if index.lisa is not None:
        print(f"{'LISA IP-SA':<20s} {len(index.lisa.keys):>12,} {'(derived)':>12s} {fmt_size(index.lisa.mem_bytes_estimate):>12s}")
        print(f"  {'LISA leaves':<18s} {len(index.lisa.leaves):>12,}")
        print(f"  {'LISA avg err':<18s} {index.lisa.avg_leaf_error:>12.2f}")

    print("\n  (warming up page cache ...)")
    for query in existing_queries_2[:50]:
        index.search_baseline(query)
    for query in existing_queries_3[:50]:
        index.search_baseline(query)
    for query in existing_queries_4[:50]:
        index.search_baseline(query)

    fns = [index.search_baseline, index.search_with_unigram,
           index.search_with_bigram, index.search_with_trigram,
           index.search_with_quadgram, index.search_with_adaptive]
    labels = ["Baseline", "Unigram", "Bigram", "Trigram", "Quadgram", "Adaptive"]

    if index.lisa is not None:
        fns += [index.search_with_lisa_binary, index.search_with_lisa_rmi]
        labels += ["LISA(bin)", "LISA(RMI)"]

    fns_2 = fns + [index.search_with_1approx2]
    labels_2 = labels + ["1approx2"]
    fns_3 = fns + [index.search_with_2approx3]
    labels_3 = labels + ["2approx3"]

    def print_reductions(results, lbls):
        baseline_by_name = {}
        if "Bigram" in lbls:
            baseline_by_name["Adaptive"] = lbls.index("Bigram")

        for i, name in enumerate(lbls[1:], 1):
            if i >= len(results):
                continue
            base_idx = baseline_by_name.get(name, 0)
            base_comps = results[base_idx][3]
            if base_comps <= 0:
                continue
            base_name = lbls[base_idx]
            reduction = (1 - results[i][3] / base_comps) * 100
            if base_name == "Baseline":
                print(f"  Comparison reduction ({name:16s}): {reduction:.1f}%")
            else:
                print(f"  Comparison reduction ({name:16s} vs {base_name}): {reduction:.1f}%")

    print(f"\n--- Existing 2-gram queries (n={len(existing_queries_2)}) ---")
    results = benchmark_interleaved(existing_queries_2, fns_2)
    for label, (total, lookup_t, refine_t, comps, results_total) in zip(labels_2, results):
        print(f"  {label+':':19s} total {total:.3f}s, lookup {lookup_t:.3f}s, SA {refine_t:.3f}s, avg {comps:.1f} comparisons, {results_total:,} total results")
    print_reductions(results, labels_2)

    print(f"\n--- Existing 3-gram queries (n={len(existing_queries_3)}) ---")
    results = benchmark_interleaved(existing_queries_3, fns_3)
    for label, (total, lookup_t, refine_t, comps, results_total) in zip(labels_3, results):
        print(f"  {label+':':19s} total {total:.3f}s, lookup {lookup_t:.3f}s, SA {refine_t:.3f}s, avg {comps:.1f} comparisons, {results_total:,} total results")
    print_reductions(results, labels_3)

    print(f"\n--- Existing 4-gram queries (n={len(existing_queries_4)}) ---")
    results = benchmark_interleaved(existing_queries_4, fns)
    for label, (total, lookup_t, refine_t, comps, results_total) in zip(labels, results):
        print(f"  {label+':':19s} total {total:.3f}s, lookup {lookup_t:.3f}s, SA {refine_t:.3f}s, avg {comps:.1f} comparisons, {results_total:,} total results")
    print_reductions(results, labels)

    print(f"\n--- Non-existing random queries (n={len(nonexist_queries)}) ---")
    results = benchmark_interleaved(nonexist_queries, fns)
    for label, (total, lookup_t, refine_t, comps, results_total) in zip(labels, results):
        print(f"  {label+':':19s} total {total:.3f}s, lookup {lookup_t:.3f}s, SA {refine_t:.3f}s, avg {comps:.1f} comparisons, {results_total:,} total results")

    nonexist_2 = [query[:2] for query in nonexist_queries]
    nonexist_3 = [query[:3] for query in nonexist_queries]
    r_1a2 = benchmark_interleaved(nonexist_2, [index.search_baseline, index.search_with_1approx2])
    r_2a3 = benchmark_interleaved(nonexist_3, [index.search_baseline, index.search_with_2approx3])
    for label, (total, lookup_t, refine_t, comps, results_total) in zip(["Baseline(2g)", "1approx2"], r_1a2):
        print(f"  {label+':':19s} total {total:.3f}s, lookup {lookup_t:.3f}s, SA {refine_t:.3f}s, avg {comps:.1f} comparisons, {results_total:,} total results")
    for label, (total, lookup_t, refine_t, comps, results_total) in zip(["Baseline(3g)", "2approx3"], r_2a3):
        print(f"  {label+':':19s} total {total:.3f}s, lookup {lookup_t:.3f}s, SA {refine_t:.3f}s, avg {comps:.1f} comparisons, {results_total:,} total results")

    print()


def main():
    parser = argparse.ArgumentParser(description='Test packed n-gram cache hints')
    parser.add_argument('--index-dir', default='test/maildir_index', help='Path to index directory')
    parser.add_argument('--token-width', type=int, default=2, help='Token width in bytes (1=u8, 2=u16, 4=u32)')
    parser.add_argument('--num-queries', type=int, default=500, help='Number of queries for performance tests')
    parser.add_argument('--build-lisa', default=False, action='store_true', help='Build a LISA-style IP-SA + RMI index in memory')
    parser.add_argument('--lisa-k', type=int, default=2, help='Chunk size in tokens for the LISA-style index')
    parser.add_argument('--lisa-leaf-alpha', type=float, default=16.0, help='Average absolute error target for LISA leaf models')
    args = parser.parse_args()

    print(f"Loading index from {args.index_dir} ...")
    index = SuffixArrayIndex(
        args.index_dir,
        token_width=args.token_width,
        build_lisa=args.build_lisa,
        lisa_k=args.lisa_k,
        lisa_leaf_alpha=args.lisa_leaf_alpha,
    )
    print(f"  Tokens: {index.tok_cnt:,}")
    print(f"  Pointer size: {index.ptr_size} bytes")
    print(f"  Bigram cache: {'loaded' if index.bigram_cache else 'not found'}")
    if index.bigram_cache:
        print(f"  Bigram entries: {len(index.bigram_cache):,}")
    print(f"  LISA: {'loaded' if index.lisa else 'not built'}")
    if index.lisa is not None:
        print(f"  LISA chunk size: {index.lisa.k_tokens} tokens")
        print(f"  LISA leaves: {len(index.lisa.leaves):,}, avg leaf error: {index.lisa.avg_leaf_error:.2f}")
    print()

    all_passed = test_correctness(index)
    test_performance(index, num_queries=args.num_queries)
    index.close()

    if all_passed:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
        raise SystemExit(1)


__all__ = ["SuffixArrayIndex", "test_correctness", "test_performance", "main"]


if __name__ == '__main__':
    main()
