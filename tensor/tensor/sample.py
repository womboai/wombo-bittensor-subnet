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
#
#
from bisect import bisect
from itertools import accumulate
from random import random
from typing import TypeVar, Sequence

T = TypeVar("T")


def weighted_sample(weighted_items: Sequence[tuple[float, T]], k=1):
    k = min(k, len(weighted_items))

    enumerated_population: list[tuple[int, T]] = list([(index, item) for index, (_, item) in enumerate(weighted_items)])
    cumulative_weights: list[float] = list(accumulate([weight for weight, _ in weighted_items]))
    population_size = len(enumerated_population)
    total = cumulative_weights[-1]

    result: list[T] = []

    while len(result) < k:
        index, item = enumerated_population[bisect(
            cumulative_weights,
            random() * total,
            0,
            population_size - 1,
        )]

        if item in result:
            continue

        result.append(item)

    return result
