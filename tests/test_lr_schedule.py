import math

from tinylm.training.lr_schedule import get_lr


def test_warmup_increases_linearly() -> None:
    lrs = [get_lr(step, warmup_steps=10, max_steps=100, max_lr=1.0) for step in range(10)]
    assert lrs == sorted(lrs)
    assert lrs[0] == 1.0 * 1 / 10
    assert math.isclose(lrs[-1], 1.0 * 10 / 10)


def test_decays_toward_min_lr_after_warmup() -> None:
    max_lr = 1.0
    min_lr_ratio = 0.1
    kwargs = dict(warmup_steps=10, max_steps=100, max_lr=max_lr, min_lr_ratio=min_lr_ratio)
    lr_at_warmup_end = get_lr(10, **kwargs)
    lr_mid = get_lr(55, **kwargs)
    lr_at_end = get_lr(100, **kwargs)

    assert lr_at_warmup_end > lr_mid > lr_at_end
    assert math.isclose(lr_at_end, max_lr * min_lr_ratio, abs_tol=1e-6)


def test_lr_never_exceeds_max_lr() -> None:
    for step in range(0, 200, 5):
        lr = get_lr(step, warmup_steps=20, max_steps=100, max_lr=2.0, min_lr_ratio=0.1)
        assert lr <= 2.0 + 1e-9


def test_lr_floors_at_min_lr_beyond_max_steps() -> None:
    lr = get_lr(500, warmup_steps=10, max_steps=100, max_lr=1.0, min_lr_ratio=0.2)
    assert math.isclose(lr, 0.2, abs_tol=1e-9)
