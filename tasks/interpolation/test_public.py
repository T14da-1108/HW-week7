import dataclasses

import math
import pytest
import numpy as np
import numpy.typing as npt
from numpy.testing import assert_allclose

from .interpolation import interpolate_polynomial


@dataclasses.dataclass
class InterpolationTestCase:
    x_values: npt.NDArray[np.float64]
    y_values: npt.NDArray[np.float64]
    inter_poly: np.poly1d


TEST_CASES = [
    InterpolationTestCase(np.array([0., 1., 2.]), np.array([1., 2., 1.]), np.poly1d([-1., 2., 1.])),
    InterpolationTestCase(np.array([0., 1.]), np.array([1., 2.]), np.poly1d([1., 1.])),
    InterpolationTestCase(np.array([0., 1.]), np.array([1., 1.]), np.poly1d([1.])),
    InterpolationTestCase(np.arange(5.), np.cos(np.arange(5.)),
                          np.poly1d(np.array([-0.01465683,  0.23450012, -0.8492783,  0.16973731,  1.])))
]


@pytest.mark.parametrize('t', TEST_CASES, ids=str)
def test_interpolation(t: InterpolationTestCase) -> None:
    inter_poly = interpolate_polynomial(t.x_values, t.y_values)
    assert_allclose(inter_poly.coeffs, t.inter_poly.coeffs, atol=1e-6)
