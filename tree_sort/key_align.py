import hashlib

file = "full_key_mobile_3.txt"
with open(file, 'r+') as f:
    read_data = f.read()

codings = ""
hashes = ""
file = "key_mobile_3.txt"
length = 50
with open(file, 'w+') as f:
    for i in range(0, len(read_data), length):
        hashValue = hashlib.sha256(read_data[i:i + length].encode('utf-8')).hexdigest()
        hashes += bin(int(hashValue, 16))[2:]

    for i in range(0, len(hashes), 25):
        codings += hashes[i:i + 25] + "\n"
    f.write(codings)

# f = open(file, 'r+')
# codings = f.read()
# f.close()
#
# codings = codings.replace(" ", "")
# codings = codings.replace("\n", "")
# alignment = ""
# raw = ""
# for i in range(len(codings)):
#     raw += codings[i]
#     if i != 0 and (i + 1) % 25 == 0:
#         raw = hashlib
#         alignment += "\n"
#
# f = open(file, 'w+')
# f.write(alignment)
# f.close()
#
#
