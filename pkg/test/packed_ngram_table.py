# WARNING: EXPERIMENTAL CODE. This code is used to verify the ability of the adaptive n-gram to improve suffix-array query performance.
# This implementation is entirely experimental and demonstrates the improvement in query performance, after reducing comparisons in
# binary search. See https://www.cs.sfu.ca/~ashriram/Courses/CS295/assets/books/CSAPP_2016.pdf Chapter 2 for more details.

import bisect

import numpy as np

# Packed, searchable n-gram -> (lo, hi) table backed by a binary file.
class PackedRangeTable:

    RECORD_DTYPE = np.dtype([("key", "<u8"), ("lo", "<u8"), ("hi", "<u8")])
    RAW_KEY_DTYPE = np.dtype("V8")

    def __init__(
        self,
        path,
        num_entries,
        data_offset,
        ngram_n=None,
        sort_order="numeric",
        materialize_search_keys=False,
    ):
        self.path = path
        self.num_entries = int(num_entries)
        self.data_offset = int(data_offset)
        self.ngram_n = ngram_n
        self.sort_order = sort_order
        self.materialize_search_keys = materialize_search_keys
        self.records = np.memmap(
            path,
            dtype=self.RECORD_DTYPE,
            mode="r",
            offset=self.data_offset,
            shape=(self.num_entries,),
        )
        self.keys = self.records["key"]
        if self.sort_order == "numeric":
            if self.materialize_search_keys:
                self.search_keys = np.array(self.keys, copy=True)
            else:
                self.search_keys = self.keys
        else:
            self.search_keys = np.ndarray(
                shape=(self.num_entries,),
                dtype=self.RAW_KEY_DTYPE,
                buffer=self.records,
                offset=0,
                strides=(self.records.strides[0],),
            )
        self.mem_bytes_estimate = self.records.nbytes
        if self.materialize_search_keys:
            self.mem_bytes_estimate += self.search_keys.nbytes

    def __len__(self):
        return self.num_entries

    def lookup(self, key):
        if self.sort_order == "numeric":
            search_key = int(key)
            idx = bisect.bisect_left(self.search_keys, search_key)
        else:
            search_key = np.array(
                int(key).to_bytes(8, "little"), dtype=self.RAW_KEY_DTYPE
            )
            idx = int(np.searchsorted(self.search_keys, search_key, side="left"))
        if idx >= self.num_entries:
            return None
        if self.sort_order == "numeric":
            if int(self.search_keys[idx]) != search_key:
                return None
        else:
            if self.search_keys[idx].tobytes() != search_key.tobytes():
                return None
        rec = self.records[idx]
        return int(rec["lo"]), int(rec["hi"])

    def iter_ranges(self):
        for rec in self.records:
            yield int(rec["key"]), int(rec["lo"]), int(rec["hi"])

    def close(self):
        mm = getattr(self.records, "_mmap", None)
        if mm is not None:
            mm.close()
