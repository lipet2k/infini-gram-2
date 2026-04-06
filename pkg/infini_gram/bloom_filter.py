import struct
import os


class BloomFilter:
    """Bloom filter for fast negative n-gram query rejection.

    Loaded from a file produced by the Rust `build-bloom` command.
    File format: [8 bytes num_bits][4 bytes num_hashes][4 bytes max_ngram_n][bit array]
    """

    def __init__(self, path: str):
        with open(path, 'rb') as f:
            header = f.read(16)
            self.num_bits, self.num_hashes, self.max_ngram_n = struct.unpack('<QII', header)
            self.data = f.read()

    @staticmethod
    def _fnv1a(key: bytes, seed: int) -> int:
        h = seed & 0xFFFFFFFFFFFFFFFF
        for b in key:
            h ^= b
            h = (h * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF
        return h

    def maybe_contains(self, key: bytes) -> bool:
        h1 = self._fnv1a(key, 0xcbf29ce484222325)
        h2 = self._fnv1a(key, 0x517cc1b727220a95)
        for i in range(self.num_hashes):
            bit = (h1 + i * h2) % self.num_bits
            if not (self.data[bit >> 3] & (1 << (bit & 7))):
                return False
        return True


def load_bloom_filters(index_dirs, token_width):
    """Load bloom filters from index directories, returning a flat list (one per shard)."""
    bloom_filters = []
    for index_dir in index_dirs:
        if not os.path.isdir(index_dir):
            continue
        bf_paths = sorted(
            p for p in (os.path.join(index_dir, f) for f in os.listdir(index_dir))
            if 'bloom' in os.path.basename(p)
        )
        # Count how many shards this dir has (by counting tokenized.* files)
        ds_count = sum(1 for f in os.listdir(index_dir) if 'tokenized' in f)
        for s in range(ds_count):
            if s < len(bf_paths):
                bloom_filters.append(BloomFilter(bf_paths[s]))
            else:
                bloom_filters.append(None)
    return bloom_filters


def query_ids_to_bytes(query_ids, token_width):
    """Convert a list of token IDs to little-endian bytes."""
    if token_width == 1:
        fmt = f'<{len(query_ids)}B'
    elif token_width == 2:
        fmt = f'<{len(query_ids)}H'
    elif token_width == 4:
        fmt = f'<{len(query_ids)}I'
    else:
        raise ValueError(f'Unsupported token_width: {token_width}')
    return struct.pack(fmt, *query_ids)
