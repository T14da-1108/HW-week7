import dataclasses
from typing import Callable

import pytest
import math
from scipy.integrate import IntegrationWarning
from scipy.stats import norm

from .integrate import integrate_1


@dataclasses.dataclass
class IntegrateTestCase:
    f: Callable[[float], float]
    string_rep: str
    y: float


TEST_CASES = [
    IntegrateTestCase(lambda x: x ** 2, "x ** 2", 0.3333333333333333),
    IntegrateTestCase(lambda x: math.cos(x * math.pi), "cos(x * pi)", 0.0),
    IntegrateTestCase(lambda x: norm.pdf(x, loc=0.5, scale=0.1),
                      "PDF of normal distribution centered at 0.5 with variance 0.01", 1.0)
]


@pytest.mark.parametrize('t', TEST_CASES, ids=str)
def test_integrate(t: IntegrateTestCase) -> None:
    assert integrate_1(t.f) == pytest.approx(t.y), \
        f"Function '{t.string_rep}' integral over [0, 1] expected {t.y}, got {integrate_1(t.f)}"


def test_no_convergence() -> None:
    with pytest.warns(IntegrationWarning):
        integrate_1(lambda x: 1 / x)
