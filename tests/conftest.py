import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cric_sim.features import add_features  # noqa: E402
from cric_sim.loader import load_matches  # noqa: E402
from cric_sim.model import RateModel  # noqa: E402
from cric_sim.players import PACE, SPIN  # noqa: E402
from tests.synth import make_dataset  # noqa: E402

TYPES = {f"T{t}_P{k}": (SPIN if k % 2 == 0 else PACE) for t in range(8) for k in range(14)}


@pytest.fixture(scope="session")
def world(tmp_path_factory):
    folder = tmp_path_factory.mktemp("league")
    make_dataset(folder, n_matches=240)
    balls, matches = load_matches(folder)
    bf = add_features(balls)
    return {"balls": balls, "matches": matches, "bf": bf, "model": RateModel().fit(bf, TYPES)}
