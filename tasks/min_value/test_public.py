import dataclasses
from typing import Callable

import pytest
import numpy as np
import scipy

from .min_value import find_minimum


@dataclasses.dataclass
class FindMinimumTestCase:
    f: Callable[[float], float]
    x_min: float
    y_min: float


TEST_CASES = [
    FindMinimumTestCase(lambda x: (x - 2) ** 2 + 1, 2., 1.),
    FindMinimumTestCase(lambda x: -scipy.stats.norm.pdf(x), 0., -0.3989422804014327)
]


@pytest.mark.parametrize('t', TEST_CASES, ids=str)
def test_find_minimum(t: FindMinimumTestCase) -> None:
    y_min, x_min = find_minimum(t.f)
    assert np.isclose(y_min, t.y_min), f"Minimal value is {y_min} but expected {t.y_min}"
    assert np.isclose(t.f(x_min), t.y_min), f"Minimal value preimage {x_min} isn't mapped to minimal value {t.y_min}"
