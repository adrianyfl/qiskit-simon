"""
Test vectors for all ten SIMON variants, from appendix B of the paper.
Runs under pytest, or directly with: python test_classical_simon.py
"""

from params import SIMON_PARAMS
from classical_simon import S, words_from_hex, simon_encrypt, simon_decrypt

# (block_size, key_size, key, plaintext, ciphertext) exactly as printed in appendix B
TEST_VECTORS = [
    (32, 64,
     "1918 1110 0908 0100",
     "6565 6877",
     "c69b e9bb"),
    (48, 72,
     "121110 0a0908 020100",
     "612067 6e696c",
     "dae5ac 292cac"),
    (48, 96,
     "1a1918 121110 0a0908 020100",
     "726963 20646e",
     "6e06a5 acf156"),
    (64, 96,
     "13121110 0b0a0908 03020100",
     "6f722067 6e696c63",
     "5ca2e27f 111a8fc8"),
    (64, 128,
     "1b1a1918 13121110 0b0a0908 03020100",
     "656b696c 20646e75",
     "44c8fc20 b9dfa07a"),
    (96, 96,
     "0d0c0b0a0908 050403020100",
     "2072616c6c69 702065687420",
     "602807a462b4 69063d8ff082"),
    (96, 144,
     "151413121110 0d0c0b0a0908 050403020100",
     "746168742074 73756420666f",
     "ecad1c6c451e 3f59c5db1ae9"),
    (128, 128,
     "0f0e0d0c0b0a0908 0706050403020100",
     "6373656420737265 6c6c657661727420",
     "49681b1e1e54fe3f 65aa832af84e0bbc"),
    (128, 192,
     "1716151413121110 0f0e0d0c0b0a0908 0706050403020100",
     "206572656874206e 6568772065626972",
     "c4ac61effcdc0d4f 6c9c8d6e2597b85b"),
    (128, 256,
     "1f1e1d1c1b1a1918 1716151413121110 0f0e0d0c0b0a0908 0706050403020100",
     "74206e69206d6f6f 6d69732061207369",
     "8d2b5579afc8a3a0 3bf72a87efe7b868"),
]


def _split_block(text):
    """Plaintext and ciphertext are printed left word first, so no reversal."""
    return [int(word, 16) for word in text.split()]


def test_rotation_stays_inside_the_word():
    # left rotation must not leak bits above the word width
    assert S(0x8000, 1, 16) == 0x0001
    assert S(0xffff, 5, 16) == 0xffff
    # negative shift is a right rotation
    assert S(0x0001, -1, 16) == 0x8000
    for width in (16, 24, 32, 48, 64):
        assert S(1, 8, width) == S(1, -(width - 8), width)
        assert S((1 << width) - 1, 3, width) == (1 << width) - 1


def test_all_variants_present():
    assert len(SIMON_PARAMS) == 10
    for params in SIMON_PARAMS.values():
        assert params.word_size * 2 == params.block_size
        assert params.word_size * params.key_words == params.key_size


def test_encryption_matches_paper_vectors():
    for block_size, key_size, key_text, pt_text, ct_text in TEST_VECTORS:
        key = words_from_hex(key_text)
        pt = _split_block(pt_text)
        ct = _split_block(ct_text)
        got = simon_encrypt(block_size, key_size, pt[0], pt[1], key)
        assert got == tuple(ct), (
            f"Simon{block_size}/{key_size} encrypt: expected {ct}, got {list(got)}"
        )


def test_decryption_recovers_plaintext():
    for block_size, key_size, key_text, pt_text, ct_text in TEST_VECTORS:
        key = words_from_hex(key_text)
        pt = _split_block(pt_text)
        ct = _split_block(ct_text)
        got = simon_decrypt(block_size, key_size, ct[0], ct[1], key)
        assert got == tuple(pt), (
            f"Simon{block_size}/{key_size} decrypt: expected {pt}, got {list(got)}"
        )


def test_round_trip_on_random_blocks():
    import random
    rng = random.Random(0xc0ffee)
    for (block_size, key_size), params in SIMON_PARAMS.items():
        n = params.word_size
        for _ in range(20):
            key = [rng.getrandbits(n) for _ in range(params.key_words)]
            L, R = rng.getrandbits(n), rng.getrandbits(n)
            ct = simon_encrypt(block_size, key_size, L, R, key)
            assert simon_decrypt(block_size, key_size, ct[0], ct[1], key) == (L, R)


def test_wrong_key_word_count_is_rejected():
    for bad in ([], [0, 0, 0]):
        try:
            simon_encrypt(32, 64, 0, 0, bad)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for key words {bad}")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS  {name}")
    print("\nall tests passed")
