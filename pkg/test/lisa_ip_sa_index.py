import time

import numpy as np


class LinearModel:
    """Simple linear regression model y ~= slope * (x - x0) + intercept."""

    __slots__ = ('x0', 'slope', 'intercept')

    def __init__(self, x0, slope, intercept):
        self.x0 = float(x0)
        self.slope = float(slope)
        self.intercept = float(intercept)

    def predict(self, x):
        return self.slope * (float(x) - self.x0) + self.intercept


class LisaLeaf:
    """Leaf model over a contiguous key range."""

    __slots__ = ('start', 'end', 'min_key', 'max_key', 'model', 'avg_error', 'max_abs_error')

    def __init__(self, start, end, min_key, max_key, model, avg_error, max_abs_error):
        self.start = start
        self.end = end
        self.min_key = int(min_key)
        self.max_key = int(max_key)
        self.model = model
        self.avg_error = avg_error
        self.max_abs_error = max_abs_error


class LisaIndex:
    """Suffix-array analogue of LISA's IP-BWT + RMI exact search."""

    def __init__(self, index, k_tokens=2, leaf_alpha=16.0):
        self.index = index
        self.k_tokens = int(k_tokens)
        if self.k_tokens <= 0:
            raise ValueError(f"lisa_k must be positive, got {self.k_tokens}")
        self.token_width = index.token_width
        self.chunk_bytes = self.k_tokens * self.token_width
        self.byte_base = 257
        self.leaf_alpha = float(leaf_alpha)

        self.tok_cnt = index.tok_cnt
        self.ds_size = index.ds_size
        self.aug_n = self.tok_cnt + 1
        self.rank_domain = self.aug_n + 1
        self.max_pair_key = ((self.byte_base ** self.chunk_bytes) - 1) * self.rank_domain + (self.rank_domain - 1)
        if self.max_pair_key > np.iinfo(np.uint64).max:
            raise ValueError(
                f"IP-SA key space does not fit in uint64 for token_width={self.token_width}, "
                f"lisa_k={self.k_tokens}. Reduce lisa_k for this benchmark implementation."
            )

        self.keys = self._build_ip_sa_keys()
        assert np.all(self.keys[:-1] <= self.keys[1:]), "IP-SA keys must be globally sorted"

        self._keys_f64 = self.keys.astype(np.float64)
        self._positions_f64 = np.arange(self.aug_n, dtype=np.float64)
        self.leaves = []
        self._build_leaf_models(0, len(self.keys))
        self.leaf_max_keys = np.fromiter(
            (leaf.max_key for leaf in self.leaves),
            dtype=np.uint64,
            count=len(self.leaves),
        )
        self.root_model = self._build_root_model()
        self.avg_leaf_error = (
            sum(leaf.avg_error for leaf in self.leaves) / len(self.leaves)
            if self.leaves else 0.0
        )
        del self._keys_f64
        del self._positions_f64

        self.mem_bytes_estimate = (
            self.keys.nbytes
            + self.leaf_max_keys.nbytes
            + sum(leaf.model.__sizeof__() for leaf in self.leaves)
            + sum(leaf.__sizeof__() for leaf in self.leaves)
            + self.root_model.__sizeof__()
        )

    def _build_ip_sa_keys(self):
        isa = np.empty(self.tok_cnt + 1, dtype=np.int64)
        isa[self.tok_cnt] = 0
        for orig_rank in range(self.tok_cnt):
            ptr = self.index._read_sa_ptr(orig_rank)
            isa[ptr // self.token_width] = orig_rank + 1

        keys = np.empty(self.aug_n, dtype=np.uint64)
        keys[0] = 0

        for aug_rank in range(1, self.aug_n):
            ptr = self.index._read_sa_ptr(aug_rank - 1)
            next_tok = min(self.tok_cnt, (ptr // self.token_width) + self.k_tokens)
            next_rank = int(isa[next_tok])
            head_code = self._encode_head_bytes(self.index.ds_mmap[ptr:ptr + self.chunk_bytes])
            keys[aug_rank] = head_code * self.rank_domain + next_rank

        return keys

    def _encode_head_bytes(self, raw_bytes):
        code = 0
        raw_len = len(raw_bytes)
        for i in range(self.chunk_bytes):
            if i < raw_len:
                digit = raw_bytes[i] + 1
            else:
                digit = 0
            code = code * self.byte_base + digit
        return code

    def _encode_full_chunk(self, chunk_bytes):
        return self._encode_head_bytes(chunk_bytes)

    def _encode_partial_chunk_low(self, chunk_bytes):
        code = 0
        for value in chunk_bytes:
            code = code * self.byte_base + (value + 1)
        for _ in range(self.chunk_bytes - len(chunk_bytes)):
            code = code * self.byte_base + 0
        return code

    def _encode_partial_chunk_high(self, chunk_bytes):
        code = 0
        for value in chunk_bytes:
            code = code * self.byte_base + (value + 1)
        for _ in range(self.chunk_bytes - len(chunk_bytes)):
            code = code * self.byte_base + 256
        return code

    def _fit_linear_model(self, start, end):
        n = end - start
        if n <= 1:
            return LinearModel(self.keys[start], 0.0, float(start)), 0.0, 0

        raw_xs = self._keys_f64[start:end]
        ys = self._positions_f64[start:end]
        x0 = float(raw_xs[0])
        xs = raw_xs - x0

        x_mean = xs.mean()
        y_mean = ys.mean()
        var = np.mean((xs - x_mean) ** 2)
        if var == 0.0:
            slope = 0.0
        else:
            cov = np.mean((xs - x_mean) * (ys - y_mean))
            slope = cov / var
        intercept = y_mean - slope * x_mean
        preds = slope * xs + intercept
        abs_err = np.abs(preds - ys)
        avg_error = float(abs_err.mean())
        max_abs_error = int(np.ceil(abs_err.max()))
        return LinearModel(x0, slope, intercept), avg_error, max_abs_error

    def _build_leaf_models(self, start, end):
        model, avg_error, max_abs_error = self._fit_linear_model(start, end)
        if (end - start) <= 32 or avg_error <= self.leaf_alpha:
            self.leaves.append(LisaLeaf(
                start, end, self.keys[start], self.keys[end - 1], model, avg_error, max_abs_error
            ))
            return

        mid = (start + end) >> 1
        if mid == start or mid == end:
            self.leaves.append(LisaLeaf(
                start, end, self.keys[start], self.keys[end - 1], model, avg_error, max_abs_error
            ))
            return
        self._build_leaf_models(start, mid)
        self._build_leaf_models(mid, end)

    def _build_root_model(self):
        if len(self.leaves) <= 1:
            return LinearModel(0, 0.0, 0.0)

        boundary_keys = self.leaf_max_keys.astype(np.float64)
        x0 = float(boundary_keys[0])
        xs = boundary_keys - x0
        ys = np.arange(len(boundary_keys), dtype=np.float64)
        x_mean = xs.mean()
        y_mean = ys.mean()
        var = np.mean((xs - x_mean) ** 2)
        if var == 0.0:
            slope = 0.0
        else:
            cov = np.mean((xs - x_mean) * (ys - y_mean))
            slope = cov / var
        intercept = y_mean - slope * x_mean
        return LinearModel(x0, slope, intercept)

    def _exp_search_boundary_arr(self, arr, key, est, lo_bnd, hi_bnd):
        if lo_bnd >= hi_bnd:
            return lo_bnd, 0

        comparisons = [0]

        def is_left(pos):
            comparisons[0] += 1
            return arr[pos] < key

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

    def _binary_lower_bound_arr(self, arr, key, lo_bnd, hi_bnd):
        comparisons = 0
        lo, hi = lo_bnd, hi_bnd
        while lo < hi:
            mid = (lo + hi) >> 1
            comparisons += 1
            if arr[mid] < key:
                lo = mid + 1
            else:
                hi = mid
        return lo, comparisons

    def lower_bound_binary(self, pair_key):
        return self._binary_lower_bound_arr(self.keys, pair_key, 0, len(self.keys))

    def lower_bound_rmi(self, pair_key):
        if not self.leaves:
            return 0, 0

        pred_leaf = int(round(self.root_model.predict(pair_key)))
        pred_leaf = max(0, min(pred_leaf, len(self.leaves) - 1))
        leaf_idx, c0 = self._exp_search_boundary_arr(
            self.leaf_max_keys, pair_key, pred_leaf, 0, len(self.leaves)
        )
        if leaf_idx >= len(self.leaves):
            return len(self.keys), c0

        leaf = self.leaves[leaf_idx]
        pred_pos = int(round(leaf.model.predict(pair_key)))
        pred_pos = max(leaf.start, min(pred_pos, leaf.end - 1))
        pos, c1 = self._exp_search_boundary_arr(
            self.keys, pair_key, pred_pos, leaf.start, leaf.end
        )
        return pos, c0 + c1

    def search(self, token_ids, use_rmi):
        start_t = time.perf_counter() if self.index._perf_active else None
        try:
            query_bytes = self.index._token_ids_to_bytes(token_ids)
            if len(query_bytes) == 0:
                return 0, self.index.tok_cnt, 0

            chunks = [
                query_bytes[i:i + self.chunk_bytes]
                for i in range(0, len(query_bytes), self.chunk_bytes)
            ]

            low = 0
            high = self.aug_n
            comparisons = 0
            lower_bound = self.lower_bound_rmi if use_rmi else self.lower_bound_binary

            for chunk in reversed(chunks):
                if len(chunk) < self.chunk_bytes:
                    low_head = self._encode_partial_chunk_low(chunk)
                    high_head = self._encode_partial_chunk_high(chunk)
                    low, c1 = lower_bound(low_head * self.rank_domain + low)
                    high, c2 = lower_bound(high_head * self.rank_domain + high)
                else:
                    head = self._encode_full_chunk(chunk)
                    low, c1 = lower_bound(head * self.rank_domain + low)
                    high, c2 = lower_bound(head * self.rank_domain + high)
                comparisons += c1 + c2

            left = max(0, min(self.index.tok_cnt, low - 1))
            right = max(0, min(self.index.tok_cnt, high - 1))
            if right < left:
                right = left
            return left, right, comparisons
        finally:
            if start_t is not None:
                self.index._perf_add_sa_time(time.perf_counter() - start_t)
