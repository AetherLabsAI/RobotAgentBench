from collections import Counter
import numpy as np
import pytest
from real2sim.production.task8_initial_sampling import sample_assets, sample_centres

CLOTH = np.array([[0., 0., .452], [.342, .51, .4525]])


def test_maximum_two_per_identity_and_seed_reproducibility():
    for seed in range(100):
        for count in (1, 2, 3):
            values = sample_assets(['can', 'book', 'can'], count, np.random.default_rng(seed))
            assert len(values) == count
            assert max(Counter(values).values()) <= 2
            assert values == sample_assets(['can', 'book'], count, np.random.default_rng(seed))
    with pytest.raises(ValueError):
        sample_assets(['can'], 3, np.random.default_rng(0))


def test_three_native_books_fit_without_overlap_and_vary_in_both_axes():
    sizes = np.array([[.18, .25, .018], [.15, .22, .011], [.15, .22, .011]])
    samples = []
    for seed in range(100):
        centres = sample_centres(sizes, CLOTH, np.random.default_rng(seed))
        samples.append(centres)
        assert np.all(centres[:, :2] - sizes[:, :2]/2 >= CLOTH[0, :2])
        assert np.all(centres[:, :2] + sizes[:, :2]/2 <= CLOTH[1, :2])
        for i in range(3):
            for j in range(i):
                assert np.any(abs(centres[i, :2] - centres[j, :2]) + 1e-9 >= (sizes[i, :2] + sizes[j, :2])/2 + .015)
    assert np.all(np.ptp(np.array(samples)[:, :, :2], axis=0) > .003)
    assert np.array_equal(samples[0], sample_centres(sizes, CLOTH, np.random.default_rng(0)))


def test_impossible_compositions_rejected_instead_of_overlapping_or_shrinking():
    with pytest.raises(ValueError):
        sample_centres([[.18, .25, .018]] * 3, CLOTH, np.random.default_rng(0))
    with pytest.raises(ValueError):
        sample_centres([[.5, .6, .1]], CLOTH, np.random.default_rng(0))
