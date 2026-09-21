from typing import NamedTuple

class SimonParams(NamedTuple):
    block_size: int   # 2n
    key_size: int     # mn
    word_size: int    # n
    key_words: int    # m
    z_index: int      # which z sequence
    rounds: int       # T

# The five round-constant sequences from section 3.2 of the paper.
# z0 = u and z1 = v have period 31 (written out twice to reach 62 characters);
# z2, z3, z4 are u, v, w XORed with the period-2 sequence 0101... and have period 62.
# Indexed by SimonParams.z_index.
Z_SEQUENCES = (
    "11111010001001010110000111001101111101000100101011000011100110",
    "10001110111110010011000010110101000111011111001001100001011010",
    "10101111011100000011010010011000101000010001111110010110110011",
    "11011011101011000110010111100000010010001010011100110100001111",
    "11010001111001101011011000100000010111000011001010010011101111",
)

Z_PERIOD = 62

SIMON_PARAMS = {
    ( 32,  64): SimonParams( 32,  64, 16, 4, 0, 32),
    ( 48,  72): SimonParams( 48,  72, 24, 3, 0, 36),
    ( 48,  96): SimonParams( 48,  96, 24, 4, 1, 36),
    ( 64,  96): SimonParams( 64,  96, 32, 3, 2, 42),
    ( 64, 128): SimonParams( 64, 128, 32, 4, 3, 44),
    ( 96,  96): SimonParams( 96,  96, 48, 2, 2, 52),
    ( 96, 144): SimonParams( 96, 144, 48, 3, 3, 54),
    (128, 128): SimonParams(128, 128, 64, 2, 2, 68),
    (128, 192): SimonParams(128, 192, 64, 3, 3, 69),
    (128, 256): SimonParams(128, 256, 64, 4, 4, 72),
}


def simon_params(block_size, key_size, rounds=None):
    """Look up a variant, optionally reducing the round count.

    Round-reduced instances keep every other parameter at spec. They are the
    standard object of study in the cryptanalysis literature, and they are the
    only way the quantum circuits stay small enough to be garbled or simulated
    with the block held in superposition.

    INPUT
        block_size: size of the block, 2n
        key_size: size of the key, mn
        rounds: optional round count, defaults to the spec value T
    OUTPUT
        SimonParams
    """
    try:
        params = SIMON_PARAMS[(block_size, key_size)]
    except KeyError:
        available = ", ".join(f"{b}/{k}" for b, k in sorted(SIMON_PARAMS))
        raise KeyError(
            f"Simon{block_size}/{key_size} is not a defined variant, choose from: {available}"
        ) from None

    if rounds is None:
        return params
    if not params.key_words <= rounds <= params.rounds:
        raise ValueError(
            f"rounds for Simon{block_size}/{key_size} must be between "
            f"{params.key_words} and {params.rounds}, got {rounds}"
        )
    return params._replace(rounds=rounds)


# ----------------------------------------------------------------------------
# Block and key encoding.
#
# A block is one 2n-bit integer holding both Feistel words, the left word in the
# high half: block = (L << n) | R. A key is one mn-bit integer holding the m key
# words, k[0] in the low position. Both match how the paper prints its test
# vectors, where the spaces between words are cosmetic. "6565 6877" is the block
# 0x65656877, and "1918 1110 0908 0100" is the key 0x1918111009080100.
# ----------------------------------------------------------------------------


def _check_width(value, width, label):
    if value < 0:
        raise ValueError(f"{label} must not be negative, got {value}")
    if value >> width:
        raise ValueError(
            f"{label} does not fit in {width} bits, got {value:#x} "
            f"which needs {value.bit_length()}"
        )
    return value


def block_to_words(params, block):
    """
    Split a 2n-bit block into its two Feistel words.

    INPUT
        params: SimonParams
        block: 2n-bit integer, left word in the high half
    OUTPUT
        (L, R)
    """
    n = params.word_size
    _check_width(block, params.block_size, "block")
    return block >> n, block & ((1 << n) - 1)


def words_to_block(params, L, R):
    """
    Join two Feistel words into a 2n-bit block.

    INPUT
        params: SimonParams
        L: left word, the paper's x
        R: right word, the paper's y
    OUTPUT
        2n-bit integer
    """
    n = params.word_size
    _check_width(L, n, "left word")
    _check_width(R, n, "right word")
    return (L << n) | R


def key_to_words(params, key):
    """
    Split an mn-bit key into its m words.

    INPUT
        params: SimonParams
        key: mn-bit integer, k[0] in the low position
    OUTPUT
        list of m key words ordered k[0] .. k[m-1]
    """
    n = params.word_size
    _check_width(key, params.key_size, "key")
    mask = (1 << n) - 1
    return [(key >> (i * n)) & mask for i in range(params.key_words)]


def words_to_key(params, key_words):
    """
    Join m key words into a single mn-bit key.

    INPUT
        params: SimonParams
        key_words: list of m key words ordered k[0] .. k[m-1]
    OUTPUT
        mn-bit integer
    """
    if len(key_words) != params.key_words:
        raise ValueError(
            f"expected {params.key_words} key words for "
            f"Simon{params.block_size}/{params.key_size}, got {len(key_words)}"
        )
    n = params.word_size
    key = 0
    for i, word in enumerate(key_words):
        _check_width(word, n, f"key word {i}")
        key |= word << (i * n)
    return key
