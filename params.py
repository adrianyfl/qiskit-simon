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
