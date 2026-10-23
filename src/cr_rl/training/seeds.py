"""The three disjoint seed ranges: Training episodes, Selection and Report."""

from __future__ import annotations

# Training episode seeds (including admission-check seeds) are drawn below this limit, so they
# never land in the Selection or Report seeds.
TRAINING_SEED_LIMIT = 1_000_000
# Selection seeds choose a run's checkpoint; Report seeds produce every reported number.
SELECTION_SEEDS = range(TRAINING_SEED_LIMIT, TRAINING_SEED_LIMIT + 100)
REPORT_SEEDS = range(2_000_000, 2_000_200)
