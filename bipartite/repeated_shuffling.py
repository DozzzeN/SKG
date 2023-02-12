from itertools import permutations

base = [1, 2, 3, 4, 5, 6, 7]
base1 = [2, 1, 4, 3, 6, 7, 5]
res = []
perms = [_ for _ in permutations(base)]
non_rep = []
for j in range(len(perms)):
    p = list(perms[j])
    repeated = False
    for i in range(len(p)):
        if p[i] == base[i]:
            repeated = True
    for i in range(len(p)):
        if p[i] == base1[i]:
            repeated = True
    if not repeated:
        non_rep.append(p)
print(non_rep)
print(len(non_rep))
