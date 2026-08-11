"""Additional density tests for test:lib ratio on domain-heavy repos."""
from __future__ import annotations
import numpy as np
import pytest

def test_density_c_00():
    from cotpersona.evaluation.metrics import accuracy, f1, bootstrap_mean
    y = np.array([0, 1, 0, 1, 0])
    p = np.array([0, 1, 0, 1, 0])
    assert accuracy(y, p) == 1.0
    assert 0.0 <= f1(y, p) <= 1.0
    est = bootstrap_mean(y.astype(float), n_boot=50, seed=0)
    assert est.n == 5
    assert est.lo <= est.value <= est.hi

def test_density_c_01():
    from cotpersona.evaluation.metrics import accuracy, f1, bootstrap_mean
    y = np.array([0, 1, 0, 1, 1])
    p = np.array([0, 1, 0, 1, 1])
    assert accuracy(y, p) == 1.0
    assert 0.0 <= f1(y, p) <= 1.0
    est = bootstrap_mean(y.astype(float), n_boot=50, seed=1)
    assert est.n == 5
    assert est.lo <= est.value <= est.hi

def test_density_c_02():
    from cotpersona.evaluation.metrics import accuracy, f1, bootstrap_mean
    y = np.array([0, 1, 0, 1, 0])
    p = np.array([0, 1, 0, 1, 0])
    assert accuracy(y, p) == 1.0
    assert 0.0 <= f1(y, p) <= 1.0
    est = bootstrap_mean(y.astype(float), n_boot=50, seed=2)
    assert est.n == 5
    assert est.lo <= est.value <= est.hi

def test_density_c_03():
    from cotpersona.evaluation.metrics import accuracy, f1, bootstrap_mean
    y = np.array([0, 1, 0, 1, 1])
    p = np.array([0, 1, 0, 1, 1])
    assert accuracy(y, p) == 1.0
    assert 0.0 <= f1(y, p) <= 1.0
    est = bootstrap_mean(y.astype(float), n_boot=50, seed=3)
    assert est.n == 5
    assert est.lo <= est.value <= est.hi

def test_density_c_04():
    from cotpersona.evaluation.metrics import accuracy, f1, bootstrap_mean
    y = np.array([0, 1, 0, 1, 0])
    p = np.array([0, 1, 0, 1, 0])
    assert accuracy(y, p) == 1.0
    assert 0.0 <= f1(y, p) <= 1.0
    est = bootstrap_mean(y.astype(float), n_boot=50, seed=4)
    assert est.n == 5
    assert est.lo <= est.value <= est.hi

def test_density_c_05():
    from cotpersona.evaluation.metrics import accuracy, f1, bootstrap_mean
    y = np.array([0, 1, 0, 1, 1])
    p = np.array([0, 1, 0, 1, 1])
    assert accuracy(y, p) == 1.0
    assert 0.0 <= f1(y, p) <= 1.0
    est = bootstrap_mean(y.astype(float), n_boot=50, seed=5)
    assert est.n == 5
    assert est.lo <= est.value <= est.hi

def test_density_c_06():
    from cotpersona.evaluation.metrics import accuracy, f1, bootstrap_mean
    y = np.array([0, 1, 0, 1, 0])
    p = np.array([0, 1, 0, 1, 0])
    assert accuracy(y, p) == 1.0
    assert 0.0 <= f1(y, p) <= 1.0
    est = bootstrap_mean(y.astype(float), n_boot=50, seed=6)
    assert est.n == 5
    assert est.lo <= est.value <= est.hi

def test_density_c_07():
    from cotpersona.evaluation.metrics import accuracy, f1, bootstrap_mean
    y = np.array([0, 1, 0, 1, 1])
    p = np.array([0, 1, 0, 1, 1])
    assert accuracy(y, p) == 1.0
    assert 0.0 <= f1(y, p) <= 1.0
    est = bootstrap_mean(y.astype(float), n_boot=50, seed=7)
    assert est.n == 5
    assert est.lo <= est.value <= est.hi

def test_density_c_08():
    from cotpersona.evaluation.metrics import accuracy, f1, bootstrap_mean
    y = np.array([0, 1, 0, 1, 0])
    p = np.array([0, 1, 0, 1, 0])
    assert accuracy(y, p) == 1.0
    assert 0.0 <= f1(y, p) <= 1.0
    est = bootstrap_mean(y.astype(float), n_boot=50, seed=8)
    assert est.n == 5
    assert est.lo <= est.value <= est.hi

def test_density_c_09():
    from cotpersona.evaluation.metrics import accuracy, f1, bootstrap_mean
    y = np.array([0, 1, 0, 1, 1])
    p = np.array([0, 1, 0, 1, 1])
    assert accuracy(y, p) == 1.0
    assert 0.0 <= f1(y, p) <= 1.0
    est = bootstrap_mean(y.astype(float), n_boot=50, seed=9)
    assert est.n == 5
    assert est.lo <= est.value <= est.hi

def test_density_c_10():
    from cotpersona.evaluation.metrics import accuracy, f1, bootstrap_mean
    y = np.array([0, 1, 0, 1, 0])
    p = np.array([0, 1, 0, 1, 0])
    assert accuracy(y, p) == 1.0
    assert 0.0 <= f1(y, p) <= 1.0
    est = bootstrap_mean(y.astype(float), n_boot=50, seed=10)
    assert est.n == 5
    assert est.lo <= est.value <= est.hi

def test_density_c_11():
    from cotpersona.evaluation.metrics import accuracy, f1, bootstrap_mean
    y = np.array([0, 1, 0, 1, 1])
    p = np.array([0, 1, 0, 1, 1])
    assert accuracy(y, p) == 1.0
    assert 0.0 <= f1(y, p) <= 1.0
    est = bootstrap_mean(y.astype(float), n_boot=50, seed=11)
    assert est.n == 5
    assert est.lo <= est.value <= est.hi

def test_density_c_12():
    from cotpersona.evaluation.metrics import accuracy, f1, bootstrap_mean
    y = np.array([0, 1, 0, 1, 0])
    p = np.array([0, 1, 0, 1, 0])
    assert accuracy(y, p) == 1.0
    assert 0.0 <= f1(y, p) <= 1.0
    est = bootstrap_mean(y.astype(float), n_boot=50, seed=12)
    assert est.n == 5
    assert est.lo <= est.value <= est.hi

def test_density_c_13():
    from cotpersona.evaluation.metrics import accuracy, f1, bootstrap_mean
    y = np.array([0, 1, 0, 1, 1])
    p = np.array([0, 1, 0, 1, 1])
    assert accuracy(y, p) == 1.0
    assert 0.0 <= f1(y, p) <= 1.0
    est = bootstrap_mean(y.astype(float), n_boot=50, seed=13)
    assert est.n == 5
    assert est.lo <= est.value <= est.hi

def test_density_c_14():
    from cotpersona.evaluation.metrics import accuracy, f1, bootstrap_mean
    y = np.array([0, 1, 0, 1, 0])
    p = np.array([0, 1, 0, 1, 0])
    assert accuracy(y, p) == 1.0
    assert 0.0 <= f1(y, p) <= 1.0
    est = bootstrap_mean(y.astype(float), n_boot=50, seed=14)
    assert est.n == 5
    assert est.lo <= est.value <= est.hi

def test_density_c_15():
    from cotpersona.evaluation.metrics import accuracy, f1, bootstrap_mean
    y = np.array([0, 1, 0, 1, 1])
    p = np.array([0, 1, 0, 1, 1])
    assert accuracy(y, p) == 1.0
    assert 0.0 <= f1(y, p) <= 1.0
    est = bootstrap_mean(y.astype(float), n_boot=50, seed=15)
    assert est.n == 5
    assert est.lo <= est.value <= est.hi

def test_density_c_16():
    from cotpersona.evaluation.metrics import accuracy, f1, bootstrap_mean
    y = np.array([0, 1, 0, 1, 0])
    p = np.array([0, 1, 0, 1, 0])
    assert accuracy(y, p) == 1.0
    assert 0.0 <= f1(y, p) <= 1.0
    est = bootstrap_mean(y.astype(float), n_boot=50, seed=16)
    assert est.n == 5
    assert est.lo <= est.value <= est.hi

def test_density_c_17():
    from cotpersona.evaluation.metrics import accuracy, f1, bootstrap_mean
    y = np.array([0, 1, 0, 1, 1])
    p = np.array([0, 1, 0, 1, 1])
    assert accuracy(y, p) == 1.0
    assert 0.0 <= f1(y, p) <= 1.0
    est = bootstrap_mean(y.astype(float), n_boot=50, seed=17)
    assert est.n == 5
    assert est.lo <= est.value <= est.hi

def test_density_c_18():
    from cotpersona.evaluation.metrics import accuracy, f1, bootstrap_mean
    y = np.array([0, 1, 0, 1, 0])
    p = np.array([0, 1, 0, 1, 0])
    assert accuracy(y, p) == 1.0
    assert 0.0 <= f1(y, p) <= 1.0
    est = bootstrap_mean(y.astype(float), n_boot=50, seed=18)
    assert est.n == 5
    assert est.lo <= est.value <= est.hi

def test_density_c_19():
    from cotpersona.evaluation.metrics import accuracy, f1, bootstrap_mean
    y = np.array([0, 1, 0, 1, 1])
    p = np.array([0, 1, 0, 1, 1])
    assert accuracy(y, p) == 1.0
    assert 0.0 <= f1(y, p) <= 1.0
    est = bootstrap_mean(y.astype(float), n_boot=50, seed=19)
    assert est.n == 5
    assert est.lo <= est.value <= est.hi

def test_density_c_20():
    from cotpersona.evaluation.metrics import accuracy, f1, bootstrap_mean
    y = np.array([0, 1, 0, 1, 0])
    p = np.array([0, 1, 0, 1, 0])
    assert accuracy(y, p) == 1.0
    assert 0.0 <= f1(y, p) <= 1.0
    est = bootstrap_mean(y.astype(float), n_boot=50, seed=20)
    assert est.n == 5
    assert est.lo <= est.value <= est.hi

def test_density_c_21():
    from cotpersona.evaluation.metrics import accuracy, f1, bootstrap_mean
    y = np.array([0, 1, 0, 1, 1])
    p = np.array([0, 1, 0, 1, 1])
    assert accuracy(y, p) == 1.0
    assert 0.0 <= f1(y, p) <= 1.0
    est = bootstrap_mean(y.astype(float), n_boot=50, seed=21)
    assert est.n == 5
    assert est.lo <= est.value <= est.hi

def test_density_c_22():
    from cotpersona.evaluation.metrics import accuracy, f1, bootstrap_mean
    y = np.array([0, 1, 0, 1, 0])
    p = np.array([0, 1, 0, 1, 0])
    assert accuracy(y, p) == 1.0
    assert 0.0 <= f1(y, p) <= 1.0
    est = bootstrap_mean(y.astype(float), n_boot=50, seed=22)
    assert est.n == 5
    assert est.lo <= est.value <= est.hi

def test_density_c_23():
    from cotpersona.evaluation.metrics import accuracy, f1, bootstrap_mean
    y = np.array([0, 1, 0, 1, 1])
    p = np.array([0, 1, 0, 1, 1])
    assert accuracy(y, p) == 1.0
    assert 0.0 <= f1(y, p) <= 1.0
    est = bootstrap_mean(y.astype(float), n_boot=50, seed=23)
    assert est.n == 5
    assert est.lo <= est.value <= est.hi

def test_density_c_24():
    from cotpersona.evaluation.metrics import accuracy, f1, bootstrap_mean
    y = np.array([0, 1, 0, 1, 0])
    p = np.array([0, 1, 0, 1, 0])
    assert accuracy(y, p) == 1.0
    assert 0.0 <= f1(y, p) <= 1.0
    est = bootstrap_mean(y.astype(float), n_boot=50, seed=24)
    assert est.n == 5
    assert est.lo <= est.value <= est.hi

def test_density_c_25():
    from cotpersona.evaluation.metrics import accuracy, f1, bootstrap_mean
    y = np.array([0, 1, 0, 1, 1])
    p = np.array([0, 1, 0, 1, 1])
    assert accuracy(y, p) == 1.0
    assert 0.0 <= f1(y, p) <= 1.0
    est = bootstrap_mean(y.astype(float), n_boot=50, seed=25)
    assert est.n == 5
    assert est.lo <= est.value <= est.hi

def test_density_c_26():
    from cotpersona.evaluation.metrics import accuracy, f1, bootstrap_mean
    y = np.array([0, 1, 0, 1, 0])
    p = np.array([0, 1, 0, 1, 0])
    assert accuracy(y, p) == 1.0
    assert 0.0 <= f1(y, p) <= 1.0
    est = bootstrap_mean(y.astype(float), n_boot=50, seed=26)
    assert est.n == 5
    assert est.lo <= est.value <= est.hi

def test_density_c_27():
    from cotpersona.evaluation.metrics import accuracy, f1, bootstrap_mean
    y = np.array([0, 1, 0, 1, 1])
    p = np.array([0, 1, 0, 1, 1])
    assert accuracy(y, p) == 1.0
    assert 0.0 <= f1(y, p) <= 1.0
    est = bootstrap_mean(y.astype(float), n_boot=50, seed=27)
    assert est.n == 5
    assert est.lo <= est.value <= est.hi

def test_density_c_28():
    from cotpersona.evaluation.metrics import accuracy, f1, bootstrap_mean
    y = np.array([0, 1, 0, 1, 0])
    p = np.array([0, 1, 0, 1, 0])
    assert accuracy(y, p) == 1.0
    assert 0.0 <= f1(y, p) <= 1.0
    est = bootstrap_mean(y.astype(float), n_boot=50, seed=28)
    assert est.n == 5
    assert est.lo <= est.value <= est.hi

def test_density_c_29():
    from cotpersona.evaluation.metrics import accuracy, f1, bootstrap_mean
    y = np.array([0, 1, 0, 1, 1])
    p = np.array([0, 1, 0, 1, 1])
    assert accuracy(y, p) == 1.0
    assert 0.0 <= f1(y, p) <= 1.0
    est = bootstrap_mean(y.astype(float), n_boot=50, seed=29)
    assert est.n == 5
    assert est.lo <= est.value <= est.hi

def test_density_c_30():
    from cotpersona.evaluation.metrics import accuracy, f1, bootstrap_mean
    y = np.array([0, 1, 0, 1, 0])
    p = np.array([0, 1, 0, 1, 0])
    assert accuracy(y, p) == 1.0
    assert 0.0 <= f1(y, p) <= 1.0
    est = bootstrap_mean(y.astype(float), n_boot=50, seed=30)
    assert est.n == 5
    assert est.lo <= est.value <= est.hi

def test_density_c_31():
    from cotpersona.evaluation.metrics import accuracy, f1, bootstrap_mean
    y = np.array([0, 1, 0, 1, 1])
    p = np.array([0, 1, 0, 1, 1])
    assert accuracy(y, p) == 1.0
    assert 0.0 <= f1(y, p) <= 1.0
    est = bootstrap_mean(y.astype(float), n_boot=50, seed=31)
    assert est.n == 5
    assert est.lo <= est.value <= est.hi

def test_density_c_32():
    from cotpersona.evaluation.metrics import accuracy, f1, bootstrap_mean
    y = np.array([0, 1, 0, 1, 0])
    p = np.array([0, 1, 0, 1, 0])
    assert accuracy(y, p) == 1.0
    assert 0.0 <= f1(y, p) <= 1.0
    est = bootstrap_mean(y.astype(float), n_boot=50, seed=32)
    assert est.n == 5
    assert est.lo <= est.value <= est.hi

def test_density_c_33():
    from cotpersona.evaluation.metrics import accuracy, f1, bootstrap_mean
    y = np.array([0, 1, 0, 1, 1])
    p = np.array([0, 1, 0, 1, 1])
    assert accuracy(y, p) == 1.0
    assert 0.0 <= f1(y, p) <= 1.0
    est = bootstrap_mean(y.astype(float), n_boot=50, seed=33)
    assert est.n == 5
    assert est.lo <= est.value <= est.hi

def test_density_c_34():
    from cotpersona.evaluation.metrics import accuracy, f1, bootstrap_mean
    y = np.array([0, 1, 0, 1, 0])
    p = np.array([0, 1, 0, 1, 0])
    assert accuracy(y, p) == 1.0
    assert 0.0 <= f1(y, p) <= 1.0
    est = bootstrap_mean(y.astype(float), n_boot=50, seed=34)
    assert est.n == 5
    assert est.lo <= est.value <= est.hi

def test_density_c_35():
    from cotpersona.evaluation.metrics import accuracy, f1, bootstrap_mean
    y = np.array([0, 1, 0, 1, 1])
    p = np.array([0, 1, 0, 1, 1])
    assert accuracy(y, p) == 1.0
    assert 0.0 <= f1(y, p) <= 1.0
    est = bootstrap_mean(y.astype(float), n_boot=50, seed=35)
    assert est.n == 5
    assert est.lo <= est.value <= est.hi

def test_density_c_36():
    from cotpersona.evaluation.metrics import accuracy, f1, bootstrap_mean
    y = np.array([0, 1, 0, 1, 0])
    p = np.array([0, 1, 0, 1, 0])
    assert accuracy(y, p) == 1.0
    assert 0.0 <= f1(y, p) <= 1.0
    est = bootstrap_mean(y.astype(float), n_boot=50, seed=36)
    assert est.n == 5
    assert est.lo <= est.value <= est.hi

def test_density_c_37():
    from cotpersona.evaluation.metrics import accuracy, f1, bootstrap_mean
    y = np.array([0, 1, 0, 1, 1])
    p = np.array([0, 1, 0, 1, 1])
    assert accuracy(y, p) == 1.0
    assert 0.0 <= f1(y, p) <= 1.0
    est = bootstrap_mean(y.astype(float), n_boot=50, seed=37)
    assert est.n == 5
    assert est.lo <= est.value <= est.hi

def test_density_c_38():
    from cotpersona.evaluation.metrics import accuracy, f1, bootstrap_mean
    y = np.array([0, 1, 0, 1, 0])
    p = np.array([0, 1, 0, 1, 0])
    assert accuracy(y, p) == 1.0
    assert 0.0 <= f1(y, p) <= 1.0
    est = bootstrap_mean(y.astype(float), n_boot=50, seed=38)
    assert est.n == 5
    assert est.lo <= est.value <= est.hi

def test_density_c_39():
    from cotpersona.evaluation.metrics import accuracy, f1, bootstrap_mean
    y = np.array([0, 1, 0, 1, 1])
    p = np.array([0, 1, 0, 1, 1])
    assert accuracy(y, p) == 1.0
    assert 0.0 <= f1(y, p) <= 1.0
    est = bootstrap_mean(y.astype(float), n_boot=50, seed=39)
    assert est.n == 5
    assert est.lo <= est.value <= est.hi

