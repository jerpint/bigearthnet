import pytest

from bigearthnet.utils.hp_utils import check_hp


def test_check_hp__all_params_are_there():
    names = ['a', 'b']
    hps = {'a': 0, 'b': 1}
    check_hp(names, hps)


def test_check_hp__param_is_missing():
    names = ['a', 'b']
    hps = {'a': 0}
    with pytest.raises(ValueError):
        check_hp(names, hps)


def test_check_hp__extra_param_allowed():
    names = ['a']
    hps = {'a': 0, 'b': 1}
    check_hp(names, hps)


def test_check_hp__extra_param_not_allowed():
    names = ['a']
    hps = {'a': 0, 'b': 1}
    with pytest.raises(ValueError):
        check_hp(names, hps, allow_extra=False)
