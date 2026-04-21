import struct

from packed_ngram_table import PackedRangeTable


class AdaptiveNgramIndex:
    """Packed adaptive n-gram index with shorter-prefix fallback."""

    def __init__(self, path, token_width):
        self.path = path
        self.token_width = int(token_width)

        with open(path, 'rb') as f:
            magic = f.read(4)
            assert magic == b'ANGR', f"Bad magic: {magic}"
            version, = struct.unpack('<I', f.read(4))
            assert version == 1
            tw, = struct.unpack('<I', f.read(4))
            assert tw == self.token_width
            self.max_n, = struct.unpack('<I', f.read(4))
            num_levels, = struct.unpack('<I', f.read(4))
            _reserved, = struct.unpack('<I', f.read(4))
            self.total_entries, = struct.unpack('<Q', f.read(8))

            level_info = []
            for _ in range(num_levels):
                ngram_n, = struct.unpack('<I', f.read(4))
                _padding, = struct.unpack('<I', f.read(4))
                num_entries, = struct.unpack('<Q', f.read(8))
                data_offset, = struct.unpack('<Q', f.read(8))
                level_info.append((ngram_n, num_entries, data_offset))

        self.caches = {}
        for ngram_n, num_entries, data_offset in level_info:
            self.caches[ngram_n] = PackedRangeTable(
                path,
                num_entries,
                data_offset,
                ngram_n=ngram_n,
                materialize_search_keys=True,
            )

        self.mem_bytes_estimate = sum(cache.mem_bytes_estimate for cache in self.caches.values())

    def lookup(self, token_ids, make_ngram_key):
        for ngram_n in range(min(len(token_ids), self.max_n), 1, -1):
            cache = self.caches.get(ngram_n)
            if cache is None:
                continue
            key = make_ngram_key(token_ids, ngram_n)
            entry = cache.lookup(key)
            if entry is not None:
                return entry
            if ngram_n == 2:
                return None
        return None

    def level_items(self):
        return sorted(self.caches.items())

    def close(self):
        for cache in self.caches.values():
            cache.close()
