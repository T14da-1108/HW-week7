import dataclasses
from typing import Any

import pytest
import numpy as np
import numpy.typing as npt
from numpy.testing import assert_allclose

from .linear_system import solve_linear_system


@dataclasses.dataclass
class LinearSystemTestCase:
    a: npt.NDArray[Any]
    b: npt.NDArray[np.float64]
    inverse_exists: bool


TEST_CASES = [
    LinearSystemTestCase(np.array([[2., 3.], [4., 1.]]), np.array([5., 6.]), True),
    LinearSystemTestCase(
        np.array([[0.9607209, 0.06291704, 0.43130119, 0.53618775],
                  [0.05736444, 0.65549538, 0.18983549, 0.48275221],
                  [0.45369277, 0.91423868, 0.06960813, 0.19425341],
                  [0.69222474, 0.73894997, 0.24191081, 0.5104917]]),
        np.array([0.94618744, 0.47779297, 0.46774308, 0.12920489]),
        True
    ),
    LinearSystemTestCase(np.array([[1., 2.], [-36., -72.]]), np.array([0, 0]), False)
]


@pytest.mark.parametrize('t', TEST_CASES, ids=str)
def test_linear_system(t: LinearSystemTestCase) -> None:
    x, inverse_exists = solve_linear_system(t.a, t.b)
    assert t.inverse_exists == inverse_exists, \
        f"Inverse is expected to {"" if t.inverse_exists else "not"} exist"
    if inverse_exists:
        assert_allclose(t.a @ x, t.b, err_msg="Solution is not correct")
    else:
        assert x is None
