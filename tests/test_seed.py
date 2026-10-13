"""Auto seed = hash of the experiment identity in the resolved config."""

import hydra
import pytest

from spe.experiment_setup import resolve_seed

BASE = ["model=qwen3_14b", "perturbation=localization", "perturbation.dropout_rate=0.4"]


@pytest.fixture(scope="module")
def compose():
    with hydra.initialize_config_module(config_module="spe.conf", version_base=None):
        yield lambda *extra: hydra.compose("localization", [*BASE, *extra])


def test_seed_is_stable_and_32bit(compose) -> None:
    a, b = resolve_seed(compose()), resolve_seed(compose())
    assert a == b and 0 <= a < 2**32


@pytest.mark.parametrize(
    "override",
    [
        "perturbation.dropout_rate=0.42",
        "model=qwen3_32b",
        "prompts/turns=localization/control_topic_animals_cities",
        "perturbation.same_sentence=false",
        "experiment.stage=fine",
        "experiment.replicate=1",
        "experiment.num_samples=999",
    ],
)
def test_seed_changes_with_experiment_identity(compose, override: str) -> None:
    assert resolve_seed(compose(override)) != resolve_seed(compose())


@pytest.mark.parametrize(
    "override",
    [
        "model.device_map=cpu",
        "wandb.enabled=false",
        "wandb.project=other",
        "output.dpi=150",
    ],
)
def test_seed_ignores_run_bookkeeping(compose, override: str) -> None:
    assert resolve_seed(compose(override)) == resolve_seed(compose())


def test_explicit_int_seed_passes_through(compose) -> None:
    assert resolve_seed(compose("experiment.seed=7")) == 7
