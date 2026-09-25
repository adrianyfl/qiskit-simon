# qiskit-simon

The SIMON family of lightweight block ciphers, implemented as reversible Qiskit
circuits and validated against the specification's test vectors.

> **This is not Simon's algorithm.** SIMON is a block cipher published by the NSA
> in *The SIMON and SPECK Families of Lightweight Block Ciphers* (Beaulieu,
> Shors, Smith, Treatman-Clark, Weeks, Wingers, 19 June 2013). The name collision
> with Simon's period-finding algorithm is unfortunate and unrelated.

All ten variants are supported, in both a fixed-key and a quantum-key mode, with
encryption, decryption, and arbitrary round reduction.

## Install

```bash
pip install -e ".[dev]"
```

Qiskit is the only runtime dependency. Aer is needed for the simulator
cross-check test, not for building circuits.

## Use

```python
from params import simon_params
from quantum_simon import build_simon_encrypt
from basis_simulator import run_block

params = simon_params(32, 64)

# key fixed at build time: 2n qubits, round keys become X gates
circuit = build_simon_encrypt(32, 64, key=0x1918111009080100)
print(hex(run_block(circuit, params, 0x65656877)))
# 0xc69be9bb

# key in its own register: 2n + mn qubits, schedule computed reversibly
circuit = build_simon_encrypt(32, 64, key=None)
block, key_out = run_block(circuit, params, 0x65656877, key=0x1918111009080100)

# round-reduced, for garbling or for simulation with the block in superposition
small = build_simon_encrypt(32, 64, key=0x1918111009080100, rounds=8)
```

A block is one `2n`-bit integer with the left Feistel word in the high half, and
a key is one `mn`-bit integer with `k[0]` in the low position. Both match the
paper's printed vectors directly, since the spaces it puts between words are
cosmetic. `Plaintext: 6565 6877` is the block `0x65656877`, and
`classical_simon.from_hex` will read either form for you.

`build_simon_decrypt` has the same signature. With a quantum key it restores the
key register to the original key, which is not the same as inverting the
encryption circuit.

If you would rather work in the specification's two-word language, every
operation has a word-level counterpart: `simon_encrypt_words`,
`simon_decrypt_words`, and `run_words`. The block-level functions are those plus
the encoding, and `params` exposes the conversions as `block_to_words`,
`words_to_block`, `key_to_words`, and `words_to_key`.

## How it maps to a circuit

SIMON uses only XOR, bitwise AND, and left rotation, so the circuit needs exactly
three gates: `X`, `CX`, `CCX`. Circuits are emitted flat as those primitives,
because downstream tooling iterates `circuit.data`.

Three things make it cheap:

- **Rotations are free.** Bit `i` of `S^j x` is bit `(i-j) mod n` of `x`, so a
  rotation is a reordered view of a qubit list. No SWAP gates.
- **The Feistel exchange is free.** Swapping which register plays the left word
  is a relabeling in the builder.
- **The round needs no ancilla.** `f(x)` is accumulated straight into the other
  word, so each output bit is one Toffoli plus one CNOT.

The key schedule needed rewriting before it could be made reversible. The
paper's pseudocode builds a temporary with `tmp ^= S⁻¹tmp`, but that map is two
to one, since its kernel contains both zero and the all-ones word. Expanding the
linear operators lets every term be XORed directly into the slot being
overwritten, which needs no temporary and no ancilla:

```
k[i] = k[i-m] ⊕ S⁻³k[i-1] ⊕ S⁻⁴k[i-1] ⊕ c ⊕ z_bit               (m = 2 or 3)
k[i] = k[i-m] ⊕ S⁻³k[i-1] ⊕ S⁻⁴k[i-1]
              ⊕ k[i-3] ⊕ S⁻¹k[i-3] ⊕ c ⊕ z_bit                   (m = 4)
```

Each step reads only other slots, so it is its own inverse. Decryption uses that
to walk the schedule backward.

Note that Simon128/192 has an odd round count of 69, so its words would end
exchanged. The builders emit `n` SWAPs to normalise. Pass `swap_output=False` to
skip them and take the words in the other order.

## Cost

Toffoli count is `T·n` and qubit count is `2n + mn`, both confirmed against the
built circuits.

| Variant | Qubits, quantum key | Qubits, fixed key | Toffoli | CNOT |
|---|---|---|---|---|
| Simon32/64 | 96 | 32 | 512 | 2816 |
| Simon48/72 | 120 | 48 | 864 | 3312 |
| Simon48/96 | 144 | 48 | 864 | 4800 |
| Simon64/96 | 160 | 64 | 1344 | 5184 |
| Simon64/128 | 192 | 64 | 1408 | 7936 |
| Simon96/96 | 192 | 96 | 2496 | 9792 |
| Simon96/144 | 240 | 96 | 2592 | 10080 |
| Simon128/128 | 256 | 128 | 4352 | 17152 |
| Simon128/192 | 320 | 128 | 4416 | 17280 |
| Simon128/256 | 384 | 128 | 4608 | 26624 |

### Several blocks under one key

With a quantum key, each encryption normally runs its own key schedule. Pass
`num_blocks` to encrypt several blocks in one circuit that shares a single key
register. Each round key is produced once and applied to every block, so the
schedule is paid once per key rather than once per block, and needs no extra
qubits.

```python
from basis_simulator import run_blocks

circuit = build_simon_encrypt(32, 64, key=None, num_blocks=8)
blocks, key_out = run_blocks(circuit, params, [0x65656877] * 8, key=0x1918111009080100)
```

Registers are laid out `x0, y0, x1, y1, …, k`. `build_simon_decrypt` takes the
same option and still restores the key register. The key schedule is linear, so
this saves CX and X gates, not Toffolis. Compared with running `B` separate
quantum-key encryptions:

| Variant | Blocks | CX, separate | CX, shared | X, separate | X, shared | Qubits, separate | Qubits, shared |
|---|---|---|---|---|---|---|---|
| Simon32/64 | 8 | 22528 | 9984 | 3248 | 406 | 768 | 320 |
| Simon32/64 | 64 | 180224 | 67328 | 25984 | 406 | 6144 | 2112 |
| Simon64/128 | 64 | 507904 | 185344 | 78016 | 1219 | 12288 | 4224 |
| Simon128/256 | 64 | 1703936 | 607232 | 272000 | 4250 | 24576 | 8448 |

The saving grows with the number of blocks: at 64 blocks, Simon128/256 needs
9488 CX per block instead of 26624. The qubit saving assumes the separate
encryptions each hold their own copy of the key. The option combines with
`optimize`.

### T-count optimisation with tzap

Pass `optimize` to either builder to get a Clifford+T circuit optimised by
[tzap](https://github.com/qqq-wisc/tzap):

```bash
pip install -e ".[optimize]"
```

```python
circuit = build_simon_encrypt(32, 64, key=None, optimize="O3")
```

Each CCX is lowered to the standard 7-T decomposition, then tzap phase-folds.
Within a round every `x` qubit controls two Toffolis, so two T phases on it merge
into one S. That brings every variant from 7 to 5 T gates per Toffoli, a 28.6%
cut, with CX, H, and qubit counts unchanged. `O3` and `Osuper` give the same
result.

| Variant | T, 7-T lowering | T, tzap |
|---|---|---|
| Simon32/64 | 3584 | 2560 |
| Simon64/128 | 9856 | 7040 |
| Simon96/144 | 18144 | 12960 |
| Simon128/256 | 32256 | 23040 |

The result is equal to the original up to global phase, but it is no longer
X/CX/CCX, so `basis_simulator` cannot run it. `tzap_optimize.simulate_basis`
runs it with Aer instead, and `python tzap_optimize.py` benchmarks every
variant. That script also proves each optimised circuit equivalent with the
MQT QCEC ZX checker and cross-checks random inputs bit for bit. The ZX proof
succeeds for every case except Simon128/256 with a quantum key, where it times
out without a verdict.

## Testing

```bash
pytest
```

Because the circuits are permutations of basis states, `basis_simulator`
evaluates them exactly by walking `circuit.data` over a vector of bits, linear in
the gate count. A statevector simulator would need `2^(2n+mn)` amplitudes. One
test cross-checks the smallest instance against Aer's matrix product state
simulator, which stays at bond dimension one on a basis input.

Every variant is checked against the Appendix B vectors in both key modes, plus
round-reduced agreement with the classical reference, decryption round trips, and
key register restoration.

## Layout

| File | Contents |
|---|---|
| `params.py` | Variant table, the five z sequences, round reduction |
| `classical_simon.py` | Plain Python reference, key expansion, encrypt, decrypt |
| `quantum_simon.py` | Word views, round function, key schedule, circuit builders |
| `basis_simulator.py` | Exact basis-state evaluation of a reversible circuit |
| `tzap_optimize.py` | tzap T-count optimisation, verification, and benchmark |

## Using this as a submodule

The modules are flat and top level, so `params` and `classical_simon` occupy
common names on `sys.path`. If you vendor this into a larger project, add it as a
path entry rather than merging it into an existing source root, or move the
modules under a package directory first.

## Acknowledgments

The cipher specification is from *The SIMON and SPECK Families of Lightweight
Block Ciphers* by Ray Beaulieu, Douglas Shors, Jason Smith, Stefan
Treatman-Clark, Bryan Weeks, and Louis Wingers, National Security Agency,
19 June 2013. The paper states that the algorithms are free from intellectual
property restrictions.

The reversible formulation is not from the paper. In particular the ancilla-free
key schedule, which replaces the specification's non-invertible temporary, was
derived for this implementation.

Parts of this repository were written with
[Claude Code](https://claude.com/claude-code).
