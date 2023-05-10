import numpy as np
import math

# Curve25519 parameters
a = 486662
b = 1
p = 2**255 - 19
Gx = 9

def mod_inv(x, p):
    # Compute the modular inverse of x modulo p using the extended Euclidean algorithm
    return pow(x, p - 2, p)

def curve25519_private_key():
    # Generate a random private key
    return np.random.randint(2**25)

def curve25519_public_key(private_key):
    # Derive the corresponding public key from the private key
    X = pow(Gx, private_key, p)
    return X

def curve25519_shared_secret(private_key, public_key):
    # Compute the shared secret given a private key and the other party's public key
    X = pow(public_key, private_key, p)
    return X

# Alice's key generation and public key derivation
# alice_private_key = curve25519_private_key()
alice_private_key = 97
alice_public_key = curve25519_public_key(alice_private_key)

# Bob's key generation and public key derivation
bob_private_key = curve25519_private_key()
bob_public_key = curve25519_public_key(bob_private_key)

# Alice computes the shared secret
alice_shared_secret = curve25519_shared_secret(alice_private_key, bob_public_key)

# Bob computes the shared secret
bob_shared_secret = curve25519_shared_secret(bob_private_key, alice_public_key)

# Print the outcomes
print("Alice's private key:", alice_private_key)
print("Alice's public key:", alice_public_key)
print("Bob's private key:", bob_private_key)
print("Bob's public key:", bob_public_key)
print("Alice's shared secret:", alice_shared_secret)
print("Bob's shared secret:", bob_shared_secret)

# Serialize the keys
serialized_alice_private_key = alice_private_key.to_bytes(32, byteorder='little')
serialized_alice_public_key = alice_public_key.to_bytes(32, byteorder='little')
serialized_bob_private_key = bob_private_key.to_bytes(32, byteorder='little')
serialized_bob_public_key = bob_public_key.to_bytes(32, byteorder='little')

# Print the serialized keys
print("Serialized Alice's private key:", serialized_alice_private_key)
print("Serialized Alice's public key:", serialized_alice_public_key)
print("Serialized Bob's private key:", serialized_bob_private_key)
print("Serialized Bob's public key:", serialized_bob_public_key)

# Alice's private key: 123456789
# Alice's public key: 47859039200124609264377712422837634806364497573728604887328028806723990900839
# Bob's private key: 4412330
# Bob's public key: 38059712838497007757848350225578447817998915702829143092466257163477950495974
# Alice's shared secret: 32690576710806050764186938829578806397805906331979359494572025807044630982723
# Bob's shared secret: 32690576710806050764186938829578806397805906331979359494572025807044630982723
# Serialized Alice's private key: b'\x15\xcd[\x07\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
# Serialized Alice's public key: b'g\xd0\x1e\xc7\xabA\x1a\x0b\x95\xa0\xb3^}~G\n\x07\x91\x9f\xde\xcb\xb1k\xf5\xd8\xcf\xc1\xab\x99A\xcfi'
# Serialized Bob's private key: b'\xaaSC\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
# Serialized Bob's public key: b'\xe6|\xf9\x84\xe6\xb4\xb3c4\xf1\xf1\xadi\xc4~$kHm\xa6\xe5F\x92\x86\xe8\xbf\x1c\x8a\xb9\x08%T'
