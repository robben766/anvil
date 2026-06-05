import anvil_obs.exporter as exp_mod
import pytest


@pytest.fixture(autouse=True)
def reset_exporter_singleton():
    yield
    exp_mod._active = None
