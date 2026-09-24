"""Instrument table (sidebar "Instrument" section) and its helpers."""
import pytest

from config.constants import (INSTRUMENTS, DEFAULT_INSTRUMENT, MC_DEFAULT_BALANCE,
                              MC_DEFAULT_TRADES_PER_SIM, instrument_point_value,
                              instrument_margin)


@pytest.mark.parametrize("symbol", list(INSTRUMENTS))
def test_every_entry_positive(symbol):
    min_tick, tick_value, margin = INSTRUMENTS[symbol]
    assert min_tick > 0
    assert tick_value > 0
    assert margin > 0


@pytest.mark.parametrize("symbol,expected", [
    ("ES", 50.0), ("GC", 100.0), ("ZN", 1000.0), ("MBT", 0.1), ("6J", 12_500_000.0),
])
def test_point_value(symbol, expected):
    assert instrument_point_value(symbol) == pytest.approx(expected)


def test_margin():
    assert instrument_margin("ES") == 28000
    assert instrument_margin("GC") == 22000


def test_btc_replaced_by_mbt():
    assert "BTC" not in INSTRUMENTS
    assert "MBT" in INSTRUMENTS


def test_defaults():
    assert DEFAULT_INSTRUMENT == "ES"
    assert MC_DEFAULT_BALANCE == 100_000.0
    assert MC_DEFAULT_TRADES_PER_SIM == 100
