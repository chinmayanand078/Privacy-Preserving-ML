from typing import Sequence
import numpy as np
from phe import paillier


class PaillierContext:
    """
    Simple wrapper around python-paillier for encrypted linear models.
    """

    def __init__(self, key_length: int = 1024):
        self.public_key, self.private_key = paillier.generate_paillier_keypair(
            n_length=key_length
        )

    def encrypt_vector(self, x: Sequence[float]):
        return [self.public_key.encrypt(float(v)) for v in x]

    def decrypt(self, value):
        return self.private_key.decrypt(value)

    def encrypted_dot(self, enc_x, w: Sequence[float], bias: float = 0.0):
        """
        Secure linear combination:
            enc(score) = sum_i (w_i * enc_x_i) + bias
        using Paillier's homomorphic addition + scalar multiplication.
        """
        assert len(enc_x) == len(w)
        enc_score = self.public_key.encrypt(float(bias))
        for enc_feature, weight in zip(enc_x, w):
            enc_score += weight * enc_feature
        return enc_score

    def encrypted_sigmoid(self, enc_score, num_terms: int = 3):
        """
        Approximate sigmoid via low-degree polynomial in encrypted domain.
        For demo purposes only: use a simple linear approximation.

        NOTE: full non-linear sigmoid is expensive under PHE;
        here we just pass through the linear score and threshold after decryption.
        """
        return enc_score
