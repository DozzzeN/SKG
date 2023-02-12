import hashlib

file = "full_key_mobile_3.txt"
lines = []
with open(file, 'r') as f:
    for line in f:
        lines.append(line)

file = "origin_key_mobile_3.txt"
with open(file, 'w') as f:
    for line in lines:
        codings = ""
        for i in range(0, len(line), 25):
            if i + 25 >= len(line):
                break
            codings += line[i:i + 25] + "\n"
        f.write(codings)
