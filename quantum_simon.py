"""
SIMON as a reversible Qiskit circuit.

The cipher uses only XOR, bitwise AND, and rotation, so the circuit needs only
X, CX, and CCX. Two structural facts keep it cheap:

  - Rotation costs nothing. Bit i of S^j x is bit (i-j) mod n of x, so a rotation
    is a reordered view of a qubit list, never a SWAP.
  - The Feistel exchange costs nothing. Swapping which register plays the role of
    the left word is a relabeling in the builder.

Bit convention: qubit j of a word register holds bit j of the word, least
significant first. This matches Qiskit's little endian ordering, so a measured
bitstring reversed is the word value.

Circuits are emitted flat, as primitive X, CX, and CCX, because downstream
tooling iterates circuit.data and expects primitives. The optional optimize
argument instead returns a tzap-optimised Clifford+T circuit, which
basis_simulator cannot run.
"""

from qiskit import QuantumCircuit, QuantumRegister

from params import Z_SEQUENCES, Z_PERIOD, simon_params, key_to_words
from classical_simon import key_expand


class Word:
    """
    A view over n qubits holding one n-bit word, least significant bit first.

    Rotations return a new view over the same qubits, so they emit no gates.
    """

    __slots__ = ("qubits", "width")

    def __init__(self, qubits):
        self.qubits = list(qubits)
        self.width = len(self.qubits)

    def rot(self, shift):
        """S^shift as a view. Bit i of the result is bit (i - shift) mod n."""
        n = self.width
        shift %= n
        if shift == 0:
            return self
        return Word(self.qubits[(i - shift) % n] for i in range(n))

    def __getitem__(self, i):
        return self.qubits[i]

    def __iter__(self):
        return iter(self.qubits)

    def __len__(self):
        return self.width


def _apply_round(qc, x, y, round_key=None, key_bits=None):
    """
    One Feistel round body: y ^= f(x) ^ k, where f(x) = (S1 x & S8 x) ^ S2 x.

    Reads x only, writes y only, so the gates all commute and the body is its own
    inverse. The caller performs the Feistel exchange by relabeling x and y.

    Exactly one of round_key (a Word on a key register) and key_bits (an integer,
    for a key fixed at build time) must be given.
    """
    n = len(x)
    x1, x8, x2 = x.rot(1), x.rot(8), x.rot(2)

    for i in range(n):
        qc.ccx(x1[i], x8[i], y[i])      # the only non-linear part of the cipher
    for i in range(n):
        qc.cx(x2[i], y[i])

    if round_key is not None:
        for i in range(n):
            qc.cx(round_key[i], y[i])
    else:
        for i in range(n):
            if (key_bits >> i) & 1:
                qc.x(y[i])


def _apply_key_step(qc, slots, i, params):
    """
    Produce round key k[i] in place, overwriting the slot that holds k[i-m].

    Uses the expanded linear operators, so no temporary and no ancilla are
    needed. Every term is read from a different slot than the target, which makes
    the step its own inverse: applying it twice restores k[i-m]. The backward key
    schedule needed for decryption relies on that.
    """
    n, m = params.word_size, params.key_words
    target = slots[i % m]
    a = slots[(i - 1) % m]
    a3, a4 = a.rot(-3), a.rot(-4)

    for j in range(n):
        qc.cx(a3[j], target[j])
        qc.cx(a4[j], target[j])

    if m == 4:
        b = slots[(i - 3) % m]
        b1 = b.rot(-1)
        for j in range(n):
            qc.cx(b[j], target[j])
            qc.cx(b1[j], target[j])

    z_bit = int(Z_SEQUENCES[params.z_index][(i - m) % Z_PERIOD])
    const = (((1 << n) - 1) ^ 3) ^ z_bit        # c = 2^n - 4, then the round constant
    for j in range(n):
        if (const >> j) & 1:
            qc.x(target[j])


def _setup(params, key, num_blocks):
    """
    Build the empty circuit, one (x, y) register pair per block, and the key.

    Registers are laid out x0, y0, x1, y1, ..., then k. A single block keeps
    the plain names x and y.
    """
    if not isinstance(num_blocks, int) or num_blocks < 1:
        raise ValueError(f"num_blocks must be a positive integer, got {num_blocks!r}")
    n, m = params.word_size, params.key_words
    suffixes = [""] if num_blocks == 1 else [str(b) for b in range(num_blocks)]
    pairs = [(QuantumRegister(n, f"x{s}"), QuantumRegister(n, f"y{s}")) for s in suffixes]
    registers = [reg for pair in pairs for reg in pair]

    if key is None:
        kr = QuantumRegister(m * n, "k")
        qc = QuantumCircuit(*registers, kr)
        slots = [Word(kr[j * n:(j + 1) * n]) for j in range(m)]
        round_keys = None
    else:
        qc = QuantumCircuit(*registers)
        slots = None
        round_keys = key_expand(params, key_to_words(params, key))

    return qc, pairs, slots, round_keys


def _normalise_output(qc, xr, yr, params, swap_output):
    """
    An odd round count leaves the two words exchanged, since each round relabels.

    Only Simon128/192 has an odd spec round count, at T=69, but any reduced round
    count can be odd too. Emitting n SWAPs restores the convention that the left
    word ends in register x.
    """
    if params.rounds % 2 == 0:
        return True
    if not swap_output:
        return False
    for i in range(params.word_size):
        qc.swap(xr[i], yr[i])
    return True


def _finish(qc, name, optimize):
    qc.name = name
    if optimize is None:
        return qc
    from tzap_optimize import optimize_circuit
    optimized = optimize_circuit(qc, level=optimize)
    optimized.name = name
    return optimized


def _name(block_size, key_size, rounds, T, num_blocks, suffix=""):
    return (f"simon{block_size}/{key_size}{suffix}"
            + ("" if rounds is None else f"-r{T}")
            + ("" if num_blocks == 1 else f"-x{num_blocks}"))


def build_simon_encrypt(block_size, key_size, key=None, rounds=None, swap_output=True,
                        optimize=None, num_blocks=1):
    """
    SIMON 2n/mn encryption circuit.

    INPUT
        block_size: size of the block, 2n
        key_size: size of the key, mn
        key: mn-bit integer, k[0] in the low position, to fix the key at build
             time, or None to put the key in its own quantum register
        rounds: optional reduced round count, defaults to the spec value T
        swap_output: emit SWAPs so the left word always ends in register x
        optimize: None for the X/CX/CCX circuit, or a tzap level ("O1", "O2",
                  "O3", "Osuper") to return an optimised Clifford+T circuit,
                  equal up to global phase. Needs the optimize extra.
        num_blocks: encrypt this many blocks under the one key. Each round key
                    is produced once and applied to every block, so the key
                    schedule is paid once per key rather than once per block.
    OUTPUT
        QuantumCircuit over registers x, y, and k when the key is quantum, or
        x0, y0, x1, y1, ..., k when num_blocks > 1

    With a quantum key the key register is left holding the last m round keys,
    not the original key. That is a bijection of the key, so it is reversible,
    but callers that need the key back must uncompute.
    """
    params = simon_params(block_size, key_size, rounds)
    m, T = params.key_words, params.rounds
    qc, pairs, slots, round_keys = _setup(params, key, num_blocks)
    words = [(Word(xr), Word(yr)) for xr, yr in pairs]

    for i in range(T):
        if key is None and i >= m:
            _apply_key_step(qc, slots, i, params)
        for b, (x, y) in enumerate(words):
            if key is None:
                _apply_round(qc, x, y, round_key=slots[i % m])
            else:
                _apply_round(qc, x, y, key_bits=round_keys[i])
            words[b] = (y, x)            # Feistel exchange, no gates

    for xr, yr in pairs:
        _normalise_output(qc, xr, yr, params, swap_output)
    return _finish(qc, _name(block_size, key_size, rounds, T, num_blocks), optimize)


def build_simon_decrypt(block_size, key_size, key=None, rounds=None, swap_output=True,
                        optimize=None, num_blocks=1):
    """
    SIMON 2n/mn decryption circuit, the inverse round applied T times.

        R_k inverse (x, y) = (y, x ^ f(y) ^ k)

    This is not the same as inverting the encryption circuit. It takes the
    ciphertext together with the original key, so with a quantum key it first
    runs the key schedule forward to reach k[T-1], then walks it back one step
    per round. The key register is restored to the original key at the end.
    With num_blocks > 1 both walks are shared by every block.

    INPUT and OUTPUT are as for build_simon_encrypt.
    """
    params = simon_params(block_size, key_size, rounds)
    m, T = params.key_words, params.rounds
    qc, pairs, slots, round_keys = _setup(params, key, num_blocks)
    words = [(Word(xr), Word(yr)) for xr, yr in pairs]

    if key is None:
        for i in range(m, T):            # advance the register to the last round keys
            _apply_key_step(qc, slots, i, params)

    for i in reversed(range(T)):
        for b, (x, y) in enumerate(words):
            if key is None:
                _apply_round(qc, y, x, round_key=slots[i % m])
            else:
                _apply_round(qc, y, x, key_bits=round_keys[i])
            words[b] = (y, x)
        if key is None and i >= m:
            _apply_key_step(qc, slots, i, params)   # self inverse, so this steps back

    for xr, yr in pairs:
        _normalise_output(qc, xr, yr, params, swap_output)
    return _finish(qc, _name(block_size, key_size, rounds, T, num_blocks, "-inv"), optimize)


def round_function_gate(word_size):
    """
    The round body as a single named gate, for drawing and inspection.

    The builders above emit flat primitives instead, because downstream tooling
    iterates circuit.data. This is a convenience, not what gets composed.
    """
    n = word_size
    xr = QuantumRegister(n, "x")
    yr = QuantumRegister(n, "y")
    qc = QuantumCircuit(xr, yr, name=f"f_{n}")
    _apply_round(qc, Word(xr), Word(yr), key_bits=0)
    return qc.to_gate()


def resource_counts(circuit):
    """
    Gate and qubit counts for a built circuit.

    Toffoli count is B*T*n and qubit count is 2nB + mn for B blocks with a
    quantum key, but counting the real circuit is the honest way to report it.
    """
    ops = circuit.count_ops()
    return {
        "qubits": circuit.num_qubits,
        "ccx": ops.get("ccx", 0),
        "cx": ops.get("cx", 0),
        "x": ops.get("x", 0),
        "swap": ops.get("swap", 0),
        "depth": circuit.depth(),
    }
