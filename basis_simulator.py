"""
Exact simulation of a SIMON circuit on a computational basis state.

The circuit is built from X, CX, CCX, and SWAP only, so it is a permutation of
basis states. A basis input therefore never produces superposition, and the whole
circuit can be evaluated by walking circuit.data over a vector of bits. That is
exact and linear in the gate count, where a statevector simulator would need
2^(2n + mn) amplitudes.

Use this for correctness testing. Use Aer when the input is in superposition,
which is only feasible for small round-reduced instances.
"""

SUPPORTED_GATES = ("x", "cx", "ccx", "swap", "barrier", "id")


def evaluate(circuit, bits):
    """
    Run a basis state through a reversible circuit.

    INPUT
        circuit: QuantumCircuit of X, CX, CCX, SWAP only
        bits: list of 0/1, one per qubit, indexed by qubit position
    OUTPUT
        new list of bits
    """
    bits = list(bits)
    if len(bits) != circuit.num_qubits:
        raise ValueError(f"expected {circuit.num_qubits} bits, got {len(bits)}")
    index = {qubit: i for i, qubit in enumerate(circuit.qubits)}

    for instruction in circuit.data:
        name = instruction.operation.name
        q = [index[qubit] for qubit in instruction.qubits]

        if name == "x":
            bits[q[0]] ^= 1
        elif name == "cx":
            bits[q[1]] ^= bits[q[0]]
        elif name == "ccx":
            bits[q[2]] ^= bits[q[0]] & bits[q[1]]
        elif name == "swap":
            bits[q[0]], bits[q[1]] = bits[q[1]], bits[q[0]]
        elif name in ("barrier", "id"):
            continue
        else:
            raise ValueError(
                f"{name} is not a reversible classical gate, "
                f"expected one of {SUPPORTED_GATES}"
            )
    return bits


def _word_to_bits(value, width):
    return [(value >> j) & 1 for j in range(width)]


def _bits_to_word(bits):
    return sum(bit << j for j, bit in enumerate(bits))


def pack_state(params, L, R, key_words=None):
    """
    Lay out a plaintext, and optionally a key, as a bit vector.

    Qubit order follows the registers built by quantum_simon: x, then y, then k
    when the key is quantum. Within a word, bit j sits at offset j.

    INPUT
        params: SimonParams
        L, R: the two block words
        key_words: m key words ordered k[0] .. k[m-1], or None for a fixed key
    OUTPUT
        list of bits
    """
    n = params.word_size
    bits = _word_to_bits(L, n) + _word_to_bits(R, n)
    if key_words is not None:
        if len(key_words) != params.key_words:
            raise ValueError(f"expected {params.key_words} key words, got {len(key_words)}")
        for word in key_words:
            bits += _word_to_bits(word, n)
    return bits


def unpack_state(params, bits, with_key=False):
    """
    Read a bit vector back as block words, and the key register if present.

    INPUT
        params: SimonParams
        bits: list of bits as returned by evaluate
        with_key: also return the m words sitting in the key register
    OUTPUT
        (L, R) or (L, R, key_words)
    """
    n = params.word_size
    L = _bits_to_word(bits[0:n])
    R = _bits_to_word(bits[n:2 * n])
    if not with_key:
        return L, R
    slots = [
        _bits_to_word(bits[2 * n + j * n: 2 * n + (j + 1) * n])
        for j in range(params.key_words)
    ]
    return L, R, slots


def run_block(circuit, params, L, R, key_words=None):
    """
    Convenience wrapper: load a block, run the circuit, read the block back.

    INPUT
        circuit: a circuit from quantum_simon
        params: SimonParams matching that circuit
        L, R: the two input block words
        key_words: m key words when the circuit has a quantum key register
    OUTPUT
        (L, R) when the key is fixed at build time
        (L, R, key_words) when the key is quantum
    """
    out = evaluate(circuit, pack_state(params, L, R, key_words))
    return unpack_state(params, out, with_key=key_words is not None)
