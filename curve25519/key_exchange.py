import time

from cryptography.hazmat.primitives.asymmetric import x25519
from cryptography.hazmat.primitives import serialization

# Generate Bob's private key
bob_private_key = x25519.X25519PrivateKey.generate()

# Derive Bob's public key
bob_public_key = bob_private_key.public_key()

start = time.time()
# Generate Alice's private key
alice_private_key = x25519.X25519PrivateKey.generate()

# Derive Alice's public key
alice_public_key = alice_private_key.public_key()

# Alice computes the shared secret
alice_shared_secret = alice_private_key.exchange(bob_public_key)
print("Time taken:", time.time() - start)

# Bob computes the shared secret
bob_shared_secret = bob_private_key.exchange(alice_public_key)

print("Alice's shared secret:", alice_shared_secret.hex())
print("Bob's shared secret:", bob_shared_secret.hex())