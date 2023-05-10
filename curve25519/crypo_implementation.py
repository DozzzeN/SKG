from cryptography.hazmat.primitives.asymmetric import x25519
from cryptography.hazmat.primitives import serialization

def generate_curve25519_private_key_from_integer(integer):
    # Convert the integer to bytes
    private_bytes = integer.to_bytes(32, 'little')

    # Generate the Curve25519 private key from bytes
    private_key = x25519.X25519PrivateKey.from_private_bytes(private_bytes)

    return private_key

# Alice's key generation
# alice_private_key = x25519.X25519PrivateKey.from_private_bytes(b'\x15\xcd[\x07\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
integer = 97
alice_private_key = generate_curve25519_private_key_from_integer(integer)
alice_private_key_bytes = alice_private_key.private_bytes(
    encoding=serialization.Encoding.Raw,
    format=serialization.PrivateFormat.Raw,
    encryption_algorithm=serialization.NoEncryption()
)
print(int.from_bytes(alice_private_key_bytes, 'little'))

alice_public_key = alice_private_key.public_key()
alice_public_key_bytes = alice_public_key.public_bytes(
    encoding=serialization.Encoding.Raw,
    format=serialization.PublicFormat.Raw
)
print(int.from_bytes(alice_public_key_bytes, 'little'))

# Bob's key generation
bob_private_key = x25519.X25519PrivateKey.generate()
bob_public_key = bob_private_key.public_key()

# Alice computes the shared secret
alice_shared_secret = alice_private_key.exchange(bob_public_key)

# Bob computes the shared secret
bob_shared_secret = bob_private_key.exchange(alice_public_key)

# Print the outcomes
print("Alice's private key:", alice_private_key.private_bytes(
    encoding=serialization.Encoding.Raw,
    format=serialization.PrivateFormat.Raw,
    encryption_algorithm=serialization.NoEncryption()
))
print("Alice's public key:", alice_public_key.public_bytes(
    encoding=serialization.Encoding.Raw,
    format=serialization.PublicFormat.Raw
))
print("Bob's private key:", bob_private_key.private_bytes(
    encoding=serialization.Encoding.Raw,
    format=serialization.PrivateFormat.Raw,
    encryption_algorithm=serialization.NoEncryption()
))
print("Bob's public key:", bob_public_key.public_bytes(
    encoding=serialization.Encoding.Raw,
    format=serialization.PublicFormat.Raw
))
print("Alice's shared secret:", alice_shared_secret.hex())
print("Bob's shared secret:", bob_shared_secret.hex())