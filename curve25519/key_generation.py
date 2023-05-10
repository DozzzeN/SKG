from cryptography.hazmat.primitives.asymmetric import x25519
from cryptography.hazmat.primitives import serialization

# Generate a private key
private_key = x25519.X25519PrivateKey.generate()

# Derive the corresponding public key
public_key = private_key.public_key()

# Serialize the private key to PEM format
private_key_pem = private_key.private_bytes(
    encoding=serialization.Encoding.PEM,
    format=serialization.PrivateFormat.PKCS8,
    encryption_algorithm=serialization.NoEncryption()
)

# Serialize the public key to PEM format
public_key_pem = public_key.public_bytes(
    encoding=serialization.Encoding.PEM,
    format=serialization.PublicFormat.SubjectPublicKeyInfo
)

# Print the private and public keys in PEM format
print("Private Key:")
print(private_key_pem.decode())
print("Public Key:")
print(public_key_pem.decode())
