"""
Tests for the tzap optimisation path. Skipped unless the optimize extra is installed.
"""

import random

import pytest

pytest.importorskip("tzap")
pytest.importorskip("qiskit_aer")

from basis_simulator import evaluate, pack_state, unpack_state
from classical_simon import from_hex
from params import simon_params
from quantum_simon import build_simon_encrypt, build_simon_decrypt
from tzap_optimize import (
    lower_to_clifford_t, optimize_circuit, clifford_t_counts,
    simulate_basis, verify_by_simulation, verify_zx,
)


@pytest.mark.parametrize("key", [0x1918111009080100, None], ids=["fixed", "quantum"])
def test_tzap_saves_two_t_gates_per_toffoli(key):
    params = simon_params(32, 64, rounds=8)
    toffolis = params.rounds * params.word_size
    original = build_simon_encrypt(32, 64, key=key, rounds=8)
    assert clifford_t_counts(lower_to_clifford_t(original))["t"] == 7 * toffolis
    assert clifford_t_counts(optimize_circuit(original))["t"] == 5 * toffolis


@pytest.mark.parametrize("key", [0x1918111009080100, None], ids=["fixed", "quantum"])
def test_optimised_circuit_matches_paper_vector(key):
    params = simon_params(32, 64)
    optimized = optimize_circuit(build_simon_encrypt(32, 64, key=key))
    quantum_key = None if key is not None else 0x1918111009080100
    out = simulate_basis(optimized, pack_state(params, 0x65656877, quantum_key))
    block = unpack_state(params, out, with_key=quantum_key is not None)
    if quantum_key is not None:
        block = block[0]
    assert block == from_hex("c69b e9bb")


def test_optimised_circuit_agrees_with_original_on_random_inputs():
    original = build_simon_encrypt(48, 72, key=None, rounds=13)
    assert verify_by_simulation(original, optimize_circuit(original), 4, random.Random(7)) == 4


KEY = 0x1918111009080100
PLAINTEXT = 0x65656877
CIPHERTEXT = 0xc69be9bb


@pytest.mark.parametrize("quantum_key", [False, True], ids=["fixed", "quantum"])
def test_builder_optimize_option_encrypts(quantum_key):
    params = simon_params(32, 64)
    qc = build_simon_encrypt(32, 64, key=None if quantum_key else KEY, optimize="O3")
    assert qc.name == "simon32/64"
    assert "ccx" not in qc.count_ops()
    assert clifford_t_counts(qc)["t"] == 5 * params.rounds * params.word_size
    out = simulate_basis(qc, pack_state(params, PLAINTEXT, KEY if quantum_key else None))
    got = unpack_state(params, out, with_key=quantum_key)
    assert (got[0] if quantum_key else got) == CIPHERTEXT


@pytest.mark.parametrize("quantum_key", [False, True], ids=["fixed", "quantum"])
def test_builder_optimize_option_decrypts_and_restores_key(quantum_key):
    params = simon_params(32, 64)
    qc = build_simon_decrypt(32, 64, key=None if quantum_key else KEY, optimize="O3")
    assert qc.name == "simon32/64-inv"
    out = simulate_basis(qc, pack_state(params, CIPHERTEXT, KEY if quantum_key else None))
    if quantum_key:
        assert unpack_state(params, out, with_key=True) == (PLAINTEXT, KEY)
    else:
        assert unpack_state(params, out) == PLAINTEXT


def test_builder_optimize_handles_odd_round_swaps():
    params = simon_params(48, 72, rounds=5)
    rng = random.Random(11)
    key, block = rng.getrandbits(72), rng.getrandbits(48)
    reference = build_simon_encrypt(48, 72, key=key, rounds=5)
    optimized = build_simon_encrypt(48, 72, key=key, rounds=5, optimize="O1")
    assert verify_by_simulation(reference, optimized, 3, rng) == 3
    assert unpack_state(params, simulate_basis(optimized, pack_state(params, block))) == \
        unpack_state(params, evaluate(reference, pack_state(params, block)))


def test_zx_proves_equivalence():
    pytest.importorskip("mqt.qcec")
    original = build_simon_encrypt(32, 64, key=None)
    assert verify_zx(original, optimize_circuit(original), timeout=300) == "equivalent"


def test_zx_rejects_a_tampered_circuit():
    pytest.importorskip("mqt.qcec")
    original = build_simon_encrypt(32, 64, key=None, rounds=8)
    optimized = optimize_circuit(original)
    tampered = optimized.copy_empty_like()
    flipped = False
    for instruction in optimized.data:
        if not flipped and instruction.operation.name == "t":
            tampered.tdg(instruction.qubits[0])
            flipped = True
        else:
            tampered.append(instruction)
    assert verify_zx(original, tampered, timeout=300) != "equivalent"
