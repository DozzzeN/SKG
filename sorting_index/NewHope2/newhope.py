import poly, params
import os, hashlib

# keygen() is a server-side function that generates the private key and returns
# a message in the form of a tuple. This message should be encoded using JSON or
# another portable format and transmitted (over an open channel) to the client.
# Returns a pair: (privateKey, publicMessage)
def keygen(a_coeffs):
    s_coeffs = get_noise()
    e_coeffs = get_noise()
    r_coeffs = poly.pointwise(s_coeffs, a_coeffs)
    p_coeffs = poly.add(e_coeffs, r_coeffs)
    return s_coeffs, p_coeffs

# get_noise() returns a random sampling from a normal distribution in the NTT domain.
def get_noise():
    noise = poly.get_noise()
    # coefficients = poly.poly_ntt(noise)
    # return coefficients
    return noise

# def get_noise_from_channel(s_coeffs):
#     noise = poly.get_noise_from_channel(s_coeffs)
#     coefficients = poly.poly_ntt(noise)
#     return coefficients

# sharedB() is a client-side function that takes the (decoded) message received from
# the server as an argument. It generates the shared key b_key and returns it, along
# with a message in the form of a tuple. This message should be encoded using JSON
# or another portable format and transmitted (over an open channel) to the server.
# Returns a pair: (sharedKey, publicMessage)
def sharedB(receivedMsg):
    (a_coeffs, pka) = receivedMsg
    s_coeffs = get_noise()
    e_coeffs = get_noise()
    b_coeffs = poly.pointwise(a_coeffs, s_coeffs)
    b_coeffs = poly.add(b_coeffs, e_coeffs)
    v_coeffs = poly.pointwise(pka, s_coeffs)
    # v_coeffs = poly.invntt(v_coeffs)
    e_prime = poly.get_noise()
    v_coeffs = poly.add(v_coeffs, e_prime)
    c_coeffs = poly.helprec(v_coeffs)
    b_key = poly.rec(v_coeffs, c_coeffs)
    return b_key, (b_coeffs, c_coeffs), v_coeffs, s_coeffs

# gen_a() returns a list of random coefficients.
def gen_a(seed):
    hashing_algorithm = hashlib.shake_128()
    hashing_algorithm.update(seed)
    # 3072 bytes from SHAKE-128 function should be enough data to get
    # N (1024) coefficients smaller than 5*Q (5*Q = 5 * 12289 = 61445):
    shake_output = hashing_algorithm.digest(3072)
    output = []
    j = 0
    for i in range(0, params.N):
        coefficient = 5 * params.Q
        # Reject coefficients that are greater than or equal to 5*Q:
        while coefficient >= 5 * params.Q:
            coefficient = int.from_bytes(
                shake_output[j * 2 : j * 2 + 2], byteorder = 'little')
            j += 1
            if j * 2 >= len(shake_output):
                print('Error: Not enough data from SHAKE-128')
                exit(1)
        output.append(coefficient)
    return output

# sharedA() is a server-side function that takes the (decoded) message received
# from the client as an argument. It generates and returns the shared key a_key.
# Returns: the shared key.
def sharedA(receivedMsg, privKey):
    (b_coeffs, c_coeffs) = receivedMsg
    v_coeffs = poly.pointwise(privKey, b_coeffs)
    # v_coeffs = poly.invntt(v_coeffs)
    a_key = poly.rec(v_coeffs, c_coeffs)
    return a_key, v_coeffs
