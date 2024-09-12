import hashlib

name = "a"
file = "key/" + name + ".txt"
lines = []
with open(file, 'r') as f:
    for line in f:
        lines.append(line)

file = "key/align_" + name + ".txt"
with open(file, 'w') as f:
    for line in lines:
        hashValue = hashlib.sha256(line.encode('utf-8')).hexdigest()
        hashes = bin(int(hashValue, 16))[2:]
        codings = hashes[len(hashes) - 25: len(hashes)] + "\n"
        codings += hashes[len(hashes) - 100: len(hashes) - 75] + "\n"
        codings += hashes[len(hashes) - 200: len(hashes) - 175] + "\n"
        codings += hashes[len(hashes) - 300: len(hashes) - 275] + "\n"
        f.write(codings)
