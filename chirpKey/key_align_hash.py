import hashlib

name = "si"
file = "key/" + name + ".txt"
lines = []
with open(file, 'r') as f:
    for line in f:
        lines.append(line)

file = "key/hash_" + name + ".txt"
with open(file, 'w') as f:
    for i in range(len(lines)):
        line += lines[i] + "i"
        hashValue = hashlib.sha256(line.encode('utf-8')).hexdigest()
        hashes = bin(int(hashValue, 16))[2:]
        codings = ""
        for i in range(0, len(hashes), 25):
            if i + 25 >= len(hashes):
                break
            codings += hashes[i:i + 25] + "\n"
        f.write(codings)
