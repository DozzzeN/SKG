def best(N, *a):
    f = [[0] * (N + 1) for _ in range(len(a) + 1)]
    g = [[0] * (N + 1) for _ in range(len(a) + 1)]

    # DP base case
    total_sum = 0
    for i in range(1, len(a) + 1):
        total_sum += a[i - 1]
        f[i][1] = total_sum ** 2
        g[i][1] = 0

    # DP recurrence
    for j in range(2, N + 1):
        for i in range(1, len(a) + 1):
            total_sum = 0
            f[i][j] = f[i][j - 1]
            g[i][j] = i
            for k in reversed(range(i)):
                total_sum += a[k]
                if f[i][j] > f[k][j - 1] + total_sum ** 2:
                    f[i][j] = f[k][j - 1] + total_sum ** 2
                    g[i][j] = k

    # Extract best expansion
    result = []
    i = len(a)
    j = N
    while j:
        k = g[i][j]
        result.insert(0, a[k:i])
        i = k
        j -= 1

    return result

# prints "1, 1, 1, 1, 1, 1, 10, 1"
# prints "4, 1, 1, 1, 1, 6"

# https://stackoverflow.com/questions/2166335/what-algorithm-to-use-to-segment-a-sequence-of-numbers-into-n-subsets-to-minimi
print(best(3, 4, 1, 1, 1, 1, 6))