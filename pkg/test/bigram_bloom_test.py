"""Compatibility wrapper for the legacy benchmark entrypoint.

Use `ngram_query_benchmark.py` for the current packed-cache benchmark code.
"""

from ngram_query_benchmark import SuffixArrayIndex, main, test_correctness, test_performance

__all__ = ["SuffixArrayIndex", "test_correctness", "test_performance", "main"]


if __name__ == '__main__':
    main()
