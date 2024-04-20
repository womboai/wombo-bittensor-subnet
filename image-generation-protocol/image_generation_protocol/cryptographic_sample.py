#  The MIT License (MIT)
#  Copyright © 2023 Yuma Rao
#  Copyright © 2024 WOMBO
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
#  documentation files (the “Software”), to deal in the Software without restriction, including without limitation
#  the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
#  and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all copies or substantial portions of
#  the Software.
#
#  THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
#  THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
#  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
#  OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
#  DEALINGS IN THE SOFTWARE.

import os
from bisect import bisect
from functools import reduce
from itertools import accumulate
from math import ceil, log
from typing import TypeVar, Sequence, Generic

T = TypeVar("T")


def __get_rand_bits(k: int) -> int:
    # k is the bits, os.urandom takes in a byte count
    mask = reduce(lambda aggregate, bit: aggregate | (1 << bit), range(k), 0)
    return int.from_bytes(os.urandom(ceil(k / 8)), "little") & mask


def __rand_below(n: int):
    if not n:
        return 0
    k = n.bit_length()  # don't use (n-1) here because n can be 1
    r = __get_rand_bits(k)  # 0 <= r < 2**k
    while r >= n:
        r = __get_rand_bits(k)
    return r


def cryptographic_sample(population: Sequence[T], k: int = 1):
    """
    Version of `random.sample` that uses os.urandom for random number generation
    """

    n = len(population)

    if not 0 <= k <= n:
        raise ValueError("Sample larger than population or is negative")

    result = [None] * k
    set_size = 21  # size of a small set minus size of an empty list

    if k > 5:
        set_size += 4 ** ceil(log(k * 3, 4))  # table size for big sets

    if n <= set_size:
        # An n-length list is smaller than a k-length set.
        # Invariant:  non-selected at pool[0 : n-i]
        pool = list(population)

        for i in range(k):
            j = __rand_below(n - i)
            result[i] = pool[j]
            pool[j] = pool[n - i - 1]  # move non-selected item into vacancy
    else:
        selected = set()
        selected_add = selected.add
        for i in range(k):
            j = __rand_below(n)
            while j in selected:
                j = __rand_below(n)
            selected_add(j)
            result[i] = population[j]

    return result


class WeightedList(Generic[T], Sequence[T]):
    _population: list[T]
    _cumulative_weights: list[float]

    def __init__(self, weighted_items: Sequence[tuple[float, T]]):
        self._population = list([item for _, item in weighted_items])
        self._cumulative_weights = list(accumulate([weight for weight, _ in weighted_items]))

    def __len__(self):
        return int(self._cumulative_weights[-1])

    def __getitem__(self, index: int):
        if not 0 <= index < len(self):
            raise IndexError(index)

        return self._population[bisect(self._cumulative_weights, index, 0, len(self._population) - 1)]
