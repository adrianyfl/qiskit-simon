"""
T-count optimisation of the SIMON circuits with tzap (github.com/qqq-wisc/tzap).

tzap works on Clifford+T, so each CCX is first lowered to the standard 7-T
decomposition. That lowering is the baseline: it is what the circuit costs on a
fault-tolerant machine before optimisation. tzap then phase-folds across the
lowered circuit.

Correctness is checked two ways:
  - formally, by the ZX-calculus equivalence checker in MQT QCEC, against the
    original X/CX/CCX circuit;
  - by simulation, running the optimised circuit on random basis inputs with
    Aer's matrix product state method and comparing every output bit against
    the exact bit-vector evaluation of the original circuit.

Needs the optional extras: pip install -e ".[optimize]"
Run: python tzap_optimize.py
"""

import random
import time

from qiskit import QuantumCircuit, transpile

from basis_simulator import evaluate
from params import SIMON_PARAMS
from quantum_simon import build_simon_encrypt

CLIFFORD_T = ["x", "z", "h", "s", "sdg", "t", "tdg", "cx"]


def lower_to_clifford_t(circuit):
    """Replace every CCX and SWAP with Clifford+T, without optimising."""
    return transpile(circuit, basis_gates=CLIFFORD_T, optimization_level=0)


def optimize_circuit(circuit, level="O3"):
    """Lower to Clifford+T and run tzap. Equal to the input up to global phase."""
    from tzap.qiskit import optimize
    return optimize(lower_to_clifford_t(circuit), level=level)


def clifford_t_counts(circuit):
    ops = circuit.count_ops()
    return {
        "qubits": circuit.num_qubits,
        "t": ops.get("t", 0) + ops.get("tdg", 0),
        "t_depth": circuit.depth(lambda ins: ins.operation.name in ("t", "tdg")),
        "s": ops.get("s", 0) + ops.get("sdg", 0),
        "cx": ops.get("cx", 0),
        "h": ops.get("h", 0),
        "depth": circuit.depth(),
    }


def verify_zx(reference, optimized, timeout=600):
    """Formal equivalence check. Returns the QCEC criterion name.

    The ZX checker is sound but incomplete: "equivalent" is a proof, while
    "no_information" means it gave up, not that the circuits differ.
    """
    from mqt import qcec
    result = qcec.verify(
        reference, optimized,
        run_construction_checker=False, run_simulation_checker=False,
        run_alternating_checker=False, run_zx_checker=True, timeout=timeout,
    )
    return result.equivalence.name


def simulate_basis(circuit, bits, shots=16):
    """Run a basis state through a Clifford+T circuit with Aer MPS.

    A basis input to a permutation circuit must give one outcome with
    certainty, so more than one observed outcome is itself a failure.
    """
    from qiskit_aer import AerSimulator

    prepared = QuantumCircuit(circuit.num_qubits)
    for position, bit in enumerate(bits):
        if bit:
            prepared.x(position)
    prepared.compose(circuit, inplace=True)
    prepared.measure_all()

    counts = AerSimulator(method="matrix_product_state").run(prepared, shots=shots).result().get_counts()
    if len(counts) != 1:
        raise AssertionError(f"basis input gave {len(counts)} outcomes, expected 1")
    return [int(b) for b in next(iter(counts))[::-1]]


def verify_by_simulation(reference, optimized, samples, rng):
    """Compare every output bit on random basis inputs. Returns samples checked."""
    for _ in range(samples):
        bits = [rng.getrandbits(1) for _ in range(reference.num_qubits)]
        expected = evaluate(reference, bits)
        got = simulate_basis(optimized, bits)
        if got != expected:
            raise AssertionError("optimised circuit disagrees with the original on a basis input")
    return samples


def main(samples=3, zx_timeout=600):
    rng = random.Random(20260923)
    header = ("| Variant | Key | Qubits | T before | T after | T-depth before | T-depth after "
              "| CX | Depth before | Depth after | tzap s | Simulated | ZX proof |")
    print(header)
    print("|---" * 13 + "|")

    for (block_size, key_size) in SIMON_PARAMS:
        for mode, key in (("fixed", rng.getrandbits(key_size)), ("quantum", None)):
            original = build_simon_encrypt(block_size, key_size, key=key)
            lowered = lower_to_clifford_t(original)

            start = time.time()
            optimized = optimize_circuit(original)
            tzap_seconds = time.time() - start

            before, after = clifford_t_counts(lowered), clifford_t_counts(optimized)
            simulated = verify_by_simulation(original, optimized, samples, rng)
            proof = verify_zx(original, optimized, timeout=zx_timeout)

            cx = str(before["cx"]) if before["cx"] == after["cx"] else f"{before['cx']} → {after['cx']}"
            print(f"| Simon{block_size}/{key_size} | {mode} | {after['qubits']} "
                  f"| {before['t']} | {after['t']} ({100 * (after['t'] - before['t']) / before['t']:+.1f}%) "
                  f"| {before['t_depth']} | {after['t_depth']} | {cx} "
                  f"| {before['depth']} | {after['depth']} | {tzap_seconds:.2f} "
                  f"| {simulated}/{samples} | {proof} |", flush=True)


if __name__ == "__main__":
    main()
