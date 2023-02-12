import math
import sys

from scipy.spatial import distance
import numpy as np

distri = "normal"
p2 = 0
pk = []
an = [1, 2, 5, 12, 29, 70, 169, 408, 985, 2378, 5741, 13860, 33461, 80782, 195025, 470832, 1136689]

for k in range(2, 10):
    l = 1
    n = l * k

    print("n", k)
    a = 0
    b = 1

    times = 1000000
    matches = 0

    for t in range(times):
        samples1 = []
        samples2 = []
        for i in range(n):
            # measure = np.random.uniform(a, b)
            measure = 0
            if distri == "uniform":
                samples1.append(np.random.uniform(a, b) + measure)
                samples2.append(np.random.uniform(a, b) + measure)
            else:
                samples1.append(np.random.normal(a, b) + measure)
                samples2.append(np.random.normal(a, b) + measure)

        episodes1 = np.array(samples1).reshape(int(n / l), l)
        episodes2 = np.array(samples2).reshape(int(n / l), l)

        min_dist = np.zeros(int(n / l))
        for i in range(int(n / l)):
            tmp = sys.maxsize
            for j in range(int(n / l)):
                if distance.cityblock(episodes1[i], episodes2[j]) < tmp:
                    tmp = distance.cityblock(episodes1[i], episodes2[j])
                    min_dist[i] = j
        min_dist = sorted(min_dist)
        for i in range(1, len(min_dist)):
            if min_dist[i] == min_dist[i - 1]:
                matches += 1
                break
    print("real", 1 - matches / times)
    print("final", math.factorial(n) / math.factorial(2 * n) *
          math.sqrt(2) / 4 * (math.pow(1 + math.sqrt(2), n + 1) - math.pow(1 - math.sqrt(2), n + 1)) /
          np.power(2, n - 1) * math.factorial(n))

    print("upper bound", math.factorial(k) / np.power(k, k))

    print("final lower bound", np.power(2, n) * math.factorial(n) / math.factorial(2 * n) *
          np.power(1 / 2, n - 1) * math.factorial(n))
    if n % 2 == 0:
        print("final upper bound", np.power(2, n) * math.factorial(n) / math.factorial(2 * n) *
              np.power(1 / 2, int((n - 2) / 2)) * math.factorial(n))
    else:
        print("final upper bound", np.power(2, n) * math.factorial(n) / math.factorial(2 * n) *
              np.power(1 / 2, int((n - 1) / 2)) * math.factorial(n))

# n 2
# real 0.41684299999999996
# final 0.41666666666666663
# upper bound 0.5
# final lower bound 0.3333333333333333
# final upper bound 0.6666666666666666
# n 3
# real 0.150694
# final 0.15
# upper bound 0.2222222222222222
# final lower bound 0.1
# final upper bound 0.2
# n 4
# real 0.05179900000000004
# final 0.05178571428571428
# upper bound 0.09375
# final lower bound 0.028571428571428574
# final upper bound 0.1142857142857143
# n 5
# real 0.01749400000000001
# final 0.017361111111111112
# upper bound 0.0384
# final lower bound 0.007936507936507938
# final upper bound 0.03174603174603175
# n 6
# real 0.005803000000000003
# final 0.005715638528138526
# upper bound 0.015432098765432098
# final lower bound 0.0021645021645021645
# final upper bound 0.017316017316017316
# n 7
# real 0.0018089999999999495
# final 0.0018575174825174816
# upper bound 0.006119899021666143
# final lower bound 0.0005827505827505828
# final upper bound 0.004662004662004662
# n 8
# real 0.0005530000000000257
# final 0.000597926379176379
# upper bound 0.00240325927734375
# final lower bound 0.0001554001554001554
# final upper bound 0.0024864024864024864
# n 9
# real 0.00017599999999995397
# final 0.00019105435006170295
# upper bound 0.000936656708416885
# final lower bound 4.1135335252982315e-05
# final upper bound 0.000658165364047717
