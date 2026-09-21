from params import *

"""
Bit-wise rotation, defined as S in the paper
INPUT
    value: value to be rotated
    shift: amount of rotation to do positive=left, negative=right
    width: width of the value (determines rotation cutoff)
OUTPUT
    rotated value
"""


def S(value, shift, width):
    mask = (1 << width) - 1
    value &= mask
    shift %= width          # negative shifts wrap, so S(v, -3, n) is a right rotation
    if shift == 0:
        return value
    return ((value << shift) | (value >> (width - shift))) & mask


"""
Round function f, defined in section 3.1 of the paper
    f(x) = (S1 x & S8 x) ^ S2 x
INPUT
    x: n-bit word
    width: word size n
OUTPUT
    f(x)
"""


def f(x, width):
    return (S(x, 1, width) & S(x, 8, width)) ^ S(x, 2, width)


"""
Split the paper's hex key notation into word values.
The paper prints keys most significant word first, as k[m-1] ... k[0],
so the text is reversed to give k[0] first.
INPUT
    text: whitespace separated hex words, e.g. "1918 1110 0908 0100"
OUTPUT
    list of key words ordered k[0] .. k[m-1]
"""


def words_from_hex(text):
    return [int(word, 16) for word in text.split()][::-1]


"""
Key expansion, defined in section 3.2 of the paper.

The paper's pseudocode builds a temporary with tmp ^= S(tmp, -1), but that map
is two to one, its kernel holds both 0 and the all ones word, so it can never be
done in place reversibly. Expanding the linear operators lets every term be
XORed straight into the k[i-m] slot instead, which is what the quantum version
needs and is equivalent here:

    (I ^ S-1) S-3 k[i-1]  ==  S-3 k[i-1] ^ S-4 k[i-1]
    (I ^ S-1) k[i-3]      ==  k[i-3] ^ S-1 k[i-3]

INPUT
    params: SimonParams for the variant
    key_words: list of m key words, ordered k[0] .. k[m-1]
OUTPUT
    list of T round keys, k[0] .. k[T-1]
"""


def key_expand(params, key_words):
    n = params.word_size
    m = params.key_words
    if len(key_words) != m:
        raise ValueError(f"expected {m} key words for Simon{params.block_size}/{params.key_size}, "
                         f"got {len(key_words)}")

    mask = (1 << n) - 1
    c = mask ^ 3                                # 2^n - 4
    z = Z_SEQUENCES[params.z_index]

    k = [word & mask for word in key_words]
    for i in range(m, params.rounds):
        tmp = S(k[i - 1], -3, n) ^ S(k[i - 1], -4, n)
        if m == 4:
            tmp ^= k[i - 3] ^ S(k[i - 3], -1, n)
        z_bit = int(z[(i - m) % Z_PERIOD])
        k.append((k[i - m] ^ tmp ^ c ^ z_bit) & mask)
    return k


"""
SIMON 2n/mn encryption
INPUT
    block_size: size of the block, 2n
    key_size: size of the key, mn
    L: left plaintext word, the paper's x
    R: right plaintext word, the paper's y
    key_words: list of m key words, ordered k[0] .. k[m-1]
OUTPUT
    L, R ciphertext words
"""


def simon_encrypt(block_size, key_size, L, R, key_words):
    params = SIMON_PARAMS[(block_size, key_size)]
    n = params.word_size
    mask = (1 << n) - 1
    L, R = L & mask, R & mask

    for k in key_expand(params, key_words):
        # Feistel step: R_k(x, y) = (y ^ f(x) ^ k, x)
        L, R = R ^ f(L, n) ^ k, L
    return L, R


"""
SIMON 2n/mn decryption, the round function run in reverse
    R_k inverse (x, y) = (y, x ^ f(y) ^ k)
INPUT
    block_size: size of the block, 2n
    key_size: size of the key, mn
    L: left ciphertext word
    R: right ciphertext word
    key_words: list of m key words, ordered k[0] .. k[m-1]
OUTPUT
    L, R plaintext words
"""


def simon_decrypt(block_size, key_size, L, R, key_words):
    params = SIMON_PARAMS[(block_size, key_size)]
    n = params.word_size
    mask = (1 << n) - 1
    L, R = L & mask, R & mask

    for k in reversed(key_expand(params, key_words)):
        L, R = R, L ^ f(R, n) ^ k
    return L, R
