"""
WARNING: EXPERIMENTAL CODE. This code is used to verify the performance of the LISA index. This index is not used in the final report.
This implementation is entirely experimental see https://khoury.northeastern.edu/home/pandey/courses/cs7800/spring26/papers/lisa.pdf for 
more details.

To run:

python test/lisa_benchmark.py
python test/lisa_benchmark.py --slice-tokens 250000 --num-queries 1000

or ask Peter to run it for you.
"""

import argparse
import json
import math
import os
import shutil
import subprocess
import sys

from ngram_query_benchmark import SuffixArrayIndex, test_correctness, test_performance


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_DIR = os.path.dirname(THIS_DIR)
RUST_INDEXING = os.path.join(PKG_DIR, 'target', 'release', 'rust_indexing')


def _run(cmd):
    print(f"  $ {' '.join(cmd)}")
    subprocess.run(cmd, cwd=PKG_DIR, check=True)


def _copy_prefix(src_path, dst_path, num_bytes):
    with open(src_path, 'rb') as src, open(dst_path, 'wb') as dst:
        remaining = num_bytes
        while remaining > 0:
            chunk = src.read(min(8 * 1024 * 1024, remaining))
            if not chunk:
                break
            dst.write(chunk)
            remaining -= len(chunk)
    actual = os.path.getsize(dst_path)
    if actual != num_bytes:
        raise RuntimeError(f"Expected {num_bytes} bytes in {dst_path}, got {actual}")


def ensure_slice_index(args):
    if not os.path.exists(RUST_INDEXING):
        raise RuntimeError(f"Missing Rust indexing binary at {RUST_INDEXING}")

    os.makedirs(args.slice_index_dir, exist_ok=True)
    tokenized_path = os.path.join(args.slice_index_dir, 'tokenized.0')
    table_path = os.path.join(args.slice_index_dir, 'table.0')
    bigram_path = os.path.join(args.slice_index_dir, 'bigram.0')
    trigram_path = os.path.join(args.slice_index_dir, 'trigram.0')
    quadgram_path = os.path.join(args.slice_index_dir, 'quadgram.0')
    adaptive_path = os.path.join(args.slice_index_dir, 'adaptive_ngram.0')
    manifest_path = os.path.join(args.slice_index_dir, 'slice_manifest.json')

    slice_bytes = args.slice_tokens * args.token_width
    ratio = max(1, math.ceil(math.log2(max(2, slice_bytes)) / 8))
    source_tokenized = os.path.join(args.source_index_dir, 'tokenized.0')
    if not os.path.exists(source_tokenized):
        raise RuntimeError(f"Missing source tokenized shard: {source_tokenized}")

    source_size = os.path.getsize(source_tokenized)
    source_mtime_ns = os.stat(source_tokenized).st_mtime_ns
    if slice_bytes > source_size:
        raise RuntimeError(
            f"Requested slice ({slice_bytes} bytes) exceeds source shard size ({source_size} bytes)"
        )

    manifest = {
        'source_index_dir': os.path.abspath(args.source_index_dir),
        'source_tokenized_size': source_size,
        'source_tokenized_mtime_ns': source_mtime_ns,
        'slice_tokens': args.slice_tokens,
        'token_width': args.token_width,
        'ratio': ratio,
        'build_adaptive': bool(args.build_adaptive),
        'adaptive_budget': args.adaptive_budget,
        'adaptive_max_n': args.adaptive_max_n,
        'adaptive_min_range': args.adaptive_min_range,
    }

    if os.path.exists(manifest_path):
        with open(manifest_path, 'r', encoding='utf-8') as f:
            existing = json.load(f)
        comparable_existing = {k: existing.get(k) for k in manifest}
        if comparable_existing != manifest:
            raise RuntimeError(
                f"Existing slice manifest at {manifest_path} does not match requested build.\n"
                f"Requested: {manifest}\nExisting: {existing}"
            )

    if not os.path.exists(tokenized_path):
        print(f"Creating token slice: {tokenized_path} ({args.slice_tokens:,} tokens)")
        _copy_prefix(source_tokenized, tokenized_path, slice_bytes)

    if not os.path.exists(table_path):
        print("Building suffix array for slice ...")
        parts_dir = os.path.join(args.slice_index_dir, '_parts')
        os.makedirs(parts_dir, exist_ok=True)
        _run([
            RUST_INDEXING, 'make-part',
            '--data-file', tokenized_path,
            '--parts-dir', parts_dir,
            '--start-byte', '0',
            '--end-byte', str(slice_bytes),
            '--ratio', str(ratio),
            '--token-width', str(args.token_width),
        ])
        part_path = os.path.join(parts_dir, f'0-{slice_bytes}')
        if not os.path.exists(part_path):
            raise RuntimeError(f"Expected part file not found: {part_path}")
        shutil.copyfile(part_path, table_path)

    if not os.path.exists(bigram_path):
        print("Building bigram cache ...")
        _run([
            RUST_INDEXING, 'build-bigram',
            '--data-file', tokenized_path,
            '--table-file', table_path,
            '--bigram-file', bigram_path,
            '--token-width', str(args.token_width),
            '--ratio', str(ratio),
        ])

    if not os.path.exists(trigram_path):
        print("Building trigram cache ...")
        _run([
            RUST_INDEXING, 'build-trigram',
            '--data-file', tokenized_path,
            '--table-file', table_path,
            '--trigram-file', trigram_path,
            '--token-width', str(args.token_width),
            '--ratio', str(ratio),
        ])

    if not os.path.exists(quadgram_path):
        print("Building quadgram cache ...")
        _run([
            RUST_INDEXING, 'build-quadgram',
            '--data-file', tokenized_path,
            '--table-file', table_path,
            '--quadgram-file', quadgram_path,
            '--token-width', str(args.token_width),
            '--ratio', str(ratio),
        ])

    if args.build_adaptive and not os.path.exists(adaptive_path):
        print("Building adaptive n-gram cache ...")
        _run([
            RUST_INDEXING, 'build-adaptive-ngram',
            '--data-file', tokenized_path,
            '--table-file', table_path,
            '--output-file', adaptive_path,
            '--token-width', str(args.token_width),
            '--ratio', str(ratio),
            '--budget', str(args.adaptive_budget),
            '--max-n', str(args.adaptive_max_n),
            '--min-range', str(args.adaptive_min_range),
        ])

    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, sort_keys=True)


def main():
    default_source = os.path.join(THIS_DIR, 'maildir_index')
    default_slice = os.path.join(THIS_DIR, 'maildir_lisa_slice_200k')

    parser = argparse.ArgumentParser(
        description='Build a small slice index and benchmark a LISA-style query method'
    )
    parser.add_argument('--source-index-dir', default=default_source,
                        help='Existing shard directory to slice from')
    parser.add_argument('--slice-index-dir', default=default_slice,
                        help='Output directory for the slice index')
    parser.add_argument('--slice-tokens', type=int, default=200_000,
                        help='Number of tokens to copy into the slice index')
    parser.add_argument('--token-width', type=int, default=2,
                        help='Token width in bytes (1=u8, 2=u16, 4=u32)')
    parser.add_argument('--num-queries', type=int, default=500,
                        help='Number of benchmark queries per length')
    parser.add_argument('--lisa-k', type=int, default=2,
                        help='Chunk size in tokens for the LISA-style index')
    parser.add_argument('--lisa-leaf-alpha', type=float, default=16.0,
                        help='Average absolute error target for LISA leaf models')
    parser.add_argument('--build-adaptive', default=True, action=argparse.BooleanOptionalAction,
                        help='Build the adaptive n-gram cache for the slice index')
    parser.add_argument('--adaptive-budget', type=int, default=260_000,
                        help='Adaptive n-gram total entry budget')
    parser.add_argument('--adaptive-max-n', type=int, default=4,
                        help='Adaptive n-gram maximum order')
    parser.add_argument('--adaptive-min-range', type=int, default=16,
                        help='Adaptive n-gram expansion threshold')
    args = parser.parse_args()

    print(f"Source index: {args.source_index_dir}")
    print(f"Slice index:  {args.slice_index_dir}")
    ensure_slice_index(args)

    index = SuffixArrayIndex(
        args.slice_index_dir,
        token_width=args.token_width,
        build_lisa=True,
        lisa_k=args.lisa_k,
        lisa_leaf_alpha=args.lisa_leaf_alpha,
    )

    print("=" * 60)
    print(f"Loaded slice index with {index.tok_cnt:,} tokens and ptr_size={index.ptr_size}")
    print(f"LISA chunk size: {index.lisa.k_tokens}, leaves: {len(index.lisa.leaves):,}, "
          f"avg leaf error: {index.lisa.avg_leaf_error:.2f}")
    print("=" * 60)

    all_passed = test_correctness(index)
    test_performance(index, num_queries=args.num_queries)
    index.close()

    if not all_passed:
        sys.exit(1)


if __name__ == '__main__':
    main()
