"""
Tests for the Qiskit SIMON circuits.

Every variant is checked against the paper's Appendix B vectors in both key
modes, then against the classical reference on round-reduced instances.

Runs under pytest, or directly with: python test_quantum_simon.py
"""

import random

from params import (
    simon_params, SIMON_PARAMS,
    words_to_block, key_to_words, words_to_key,
)
from classical_simon import (
    simon_encrypt, simon_decrypt, words_from_hex, from_hex, key_expand,
)
from quantum_simon import (
    Word, build_simon_encrypt, build_simon_decrypt,
    round_function_gate, resource_counts,
)
from basis_simulator import (
    evaluate, pack_state, run_block, run_words, SUPPORTED_GATES,
)
from test_classical_simon import TEST_VECTORS, _split_block


def test_word_rotation_is_a_view():
    from qiskit import QuantumRegister
    reg = QuantumRegister(16, "w")
    word = Word(reg)
    # Qiskit builds a fresh Qubit object per index lookup, so compare by value
    for shift in (1, 2, 8, -1, -3, -4):
        rotated = word.rot(shift)
        for i in range(16):
            assert rotated[i] == reg[(i - shift) % 16]
    # rotating by the full width is the identity
    assert list(word.rot(16)) == list(word)


def test_circuit_uses_only_reversible_classical_gates():
    for (block_size, key_size) in SIMON_PARAMS:
        for key in (None, 0):
            qc = build_simon_encrypt(block_size, key_size, key=key)
            for name in qc.count_ops():
                assert name in SUPPORTED_GATES, f"unexpected gate {name} in Simon{block_size}/{key_size}"


def test_encryption_with_fixed_key_matches_paper_vectors():
    for block_size, key_size, key_text, pt_text, ct_text in TEST_VECTORS:
        params = simon_params(block_size, key_size)
        qc = build_simon_encrypt(block_size, key_size, key=from_hex(key_text))
        got = run_block(qc, params, from_hex(pt_text))
        assert got == from_hex(ct_text), (
            f"Simon{block_size}/{key_size} fixed key: "
            f"expected {from_hex(ct_text):#x}, got {got:#x}"
        )


def test_encryption_with_quantum_key_matches_paper_vectors():
    for block_size, key_size, key_text, pt_text, ct_text in TEST_VECTORS:
        params = simon_params(block_size, key_size)
        qc = build_simon_encrypt(block_size, key_size, key=None)
        got, _ = run_block(qc, params, from_hex(pt_text), key=from_hex(key_text))
        assert got == from_hex(ct_text), (
            f"Simon{block_size}/{key_size} quantum key: "
            f"expected {from_hex(ct_text):#x}, got {got:#x}"
        )


def test_quantum_key_register_holds_the_last_round_keys():
    """The key register is left advanced, not restored. Confirm it is exactly
    the last m round keys, which makes the leftover state a known bijection."""
    for block_size, key_size, key_text, pt_text, _ in TEST_VECTORS:
        params = simon_params(block_size, key_size)
        qc = build_simon_encrypt(block_size, key_size, key=None)
        _, key_out = run_block(qc, params, from_hex(pt_text), key=from_hex(key_text))
        slots = key_to_words(params, key_out)
        expanded = key_expand(params, words_from_hex(key_text))
        m, T = params.key_words, params.rounds
        for i in range(T - m, T):
            assert slots[i % m] == expanded[i], (
                f"Simon{block_size}/{key_size} key slot {i % m} should hold k[{i}]"
            )


def test_decryption_recovers_plaintext_with_fixed_key():
    for block_size, key_size, key_text, pt_text, ct_text in TEST_VECTORS:
        params = simon_params(block_size, key_size)
        qc = build_simon_decrypt(block_size, key_size, key=from_hex(key_text))
        got = run_block(qc, params, from_hex(ct_text))
        assert got == from_hex(pt_text), (
            f"Simon{block_size}/{key_size} decrypt: "
            f"expected {from_hex(pt_text):#x}, got {got:#x}"
        )


def test_decryption_restores_the_quantum_key_register():
    for block_size, key_size, key_text, pt_text, ct_text in TEST_VECTORS:
        params = simon_params(block_size, key_size)
        qc = build_simon_decrypt(block_size, key_size, key=None)
        got, key_out = run_block(qc, params, from_hex(ct_text), key=from_hex(key_text))
        assert got == from_hex(pt_text)
        assert key_out == from_hex(key_text), (
            f"Simon{block_size}/{key_size} decrypt should restore the key register"
        )


def test_word_layer_agrees_with_block_layer():
    """run_words is the specification-shaped layer and must stay in step."""
    for block_size, key_size, key_text, pt_text, ct_text in TEST_VECTORS:
        params = simon_params(block_size, key_size)
        pt, ct = _split_block(pt_text), _split_block(ct_text)

        fixed = build_simon_encrypt(block_size, key_size, key=from_hex(key_text))
        assert run_words(fixed, params, pt[0], pt[1]) == tuple(ct)

        quantum = build_simon_encrypt(block_size, key_size, key=None)
        L, R, slots = run_words(quantum, params, pt[0], pt[1],
                                key_words=words_from_hex(key_text))
        assert (L, R) == tuple(ct)
        assert words_to_key(params, slots) == run_block(
            quantum, params, from_hex(pt_text), key=from_hex(key_text))[1]


def test_round_reduced_matches_classical_reference():
    rng = random.Random(20250920)
    for (block_size, key_size), spec in SIMON_PARAMS.items():
        m = spec.key_words
        # include an odd round count, which exercises the output swap
        for rounds in (m, m + 1, 5, spec.rounds // 2):
            if not m <= rounds <= spec.rounds:
                continue
            params = simon_params(block_size, key_size, rounds)
            key = rng.getrandbits(key_size)
            block = rng.getrandbits(block_size)

            expected = simon_encrypt(block_size, key_size, block, key, rounds=rounds)

            fixed = build_simon_encrypt(block_size, key_size, key=key, rounds=rounds)
            assert run_block(fixed, params, block) == expected, (
                f"Simon{block_size}/{key_size} r={rounds} fixed key mismatch"
            )

            quantum = build_simon_encrypt(block_size, key_size, key=None, rounds=rounds)
            assert run_block(quantum, params, block, key=key)[0] == expected, (
                f"Simon{block_size}/{key_size} r={rounds} quantum key mismatch"
            )

            back = build_simon_decrypt(block_size, key_size, key=key, rounds=rounds)
            assert run_block(back, params, expected) == block
            assert simon_decrypt(block_size, key_size, expected, key, rounds=rounds) == block


def test_odd_round_count_swaps_words_without_normalisation():
    """With swap_output off, an odd round count leaves the words exchanged.
    Simon128/192 is the only variant whose spec round count is odd."""
    params = simon_params(128, 192)
    assert params.rounds % 2 == 1

    key = from_hex("1716151413121110 0f0e0d0c0b0a0908 0706050403020100")
    pt = from_hex("206572656874206e 6568772065626972")
    ct_words = _split_block("c4ac61effcdc0d4f 6c9c8d6e2597b85b")

    normalised = build_simon_encrypt(128, 192, key=key, swap_output=True)
    assert run_block(normalised, params, pt) == words_to_block(params, *ct_words)

    raw = build_simon_encrypt(128, 192, key=key, swap_output=False)
    assert run_block(raw, params, pt) == words_to_block(params, ct_words[1], ct_words[0])
    assert raw.count_ops().get("swap", 0) == 0


def test_toffoli_and_qubit_counts_match_the_formulas():
    for (block_size, key_size), spec in SIMON_PARAMS.items():
        n, m, T = spec.word_size, spec.key_words, spec.rounds

        quantum = resource_counts(build_simon_encrypt(block_size, key_size, key=None))
        assert quantum["ccx"] == T * n, f"Simon{block_size}/{key_size} Toffoli count"
        assert quantum["qubits"] == 2 * n + m * n, f"Simon{block_size}/{key_size} qubit count"

        fixed = resource_counts(build_simon_encrypt(block_size, key_size, key=0))
        assert fixed["ccx"] == T * n
        assert fixed["qubits"] == 2 * n


def test_oversized_key_is_rejected_by_the_builder():
    try:
        build_simon_encrypt(32, 64, key=1 << 64)
    except ValueError:
        return
    raise AssertionError("expected ValueError for a key wider than the key size")


def test_round_function_gate_is_inspectable():
    gate = round_function_gate(16)
    assert gate.num_qubits == 32
    assert gate.name == "f_16"


def test_basis_simulator_rejects_non_classical_gates():
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(1)
    qc.h(0)
    try:
        evaluate(qc, [0])
    except ValueError:
        return
    raise AssertionError("expected ValueError for a Hadamard")


def test_matches_aer_on_a_small_instance():
    """Cross-check the bit-vector simulator against Aer. A basis input stays a
    product state through X, CX, and CCX, so a matrix product state simulation
    is exact and stays at bond dimension one."""
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator

    params = simon_params(32, 64, rounds=8)
    key = from_hex("1918 1110 0908 0100")
    block = from_hex("6565 6877")

    qc = build_simon_encrypt(32, 64, key=key, rounds=8)
    expected = run_block(qc, params, block)

    prepared = QuantumCircuit(*qc.qregs)
    for position, bit in enumerate(pack_state(params, block)):
        if bit:
            prepared.x(position)
    prepared.compose(qc, inplace=True)
    prepared.measure_all()

    backend = AerSimulator(method="matrix_product_state")
    counts = backend.run(prepared, shots=1).result().get_counts()
    assert len(counts) == 1, "a basis input must give a single deterministic outcome"

    bitstring = next(iter(counts))[::-1]          # Qiskit prints most significant first
    n = params.word_size
    measured = words_to_block(
        params,
        int(bitstring[:n][::-1], 2),
        int(bitstring[n:2 * n][::-1], 2),
    )
    assert measured == expected, f"Aer gave {measured:#x}, bit-vector gave {expected:#x}"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS  {name}")
    print("\nall tests passed")
