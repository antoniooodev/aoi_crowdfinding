import numpy as np
import pytest
from src.config import PhysicalParams, GameParams, SimulationParams, SimConfig


def test_physical_rho():
    p = PhysicalParams(L=100.0, R=10.0)
    assert np.isclose(p.rho, np.pi * 10.0**2 / 100.0**2)


def test_physical_validation():
    with pytest.raises(AssertionError):
        PhysicalParams(L=0.0, R=1.0)
    with pytest.raises(AssertionError):
        PhysicalParams(L=10.0, R=10.0)


def test_game_validation():
    with pytest.raises(AssertionError):
        GameParams(N=0)
    with pytest.raises(AssertionError):
        GameParams(B=0.0)
    with pytest.raises(AssertionError):
        GameParams(c=0.0)


def test_simulation_validation():
    with pytest.raises(AssertionError):
        SimulationParams(T=0)
    with pytest.raises(AssertionError):
        SimulationParams(n_runs=0)


def test_simconfig_properties():
    cfg = SimConfig()
    assert cfg.L == cfg.physical.L
    assert cfg.R == cfg.physical.R
    assert cfg.rho == cfg.physical.rho
    assert cfg.N == cfg.game.N
    assert cfg.B == cfg.game.B
    assert cfg.c == cfg.game.c
