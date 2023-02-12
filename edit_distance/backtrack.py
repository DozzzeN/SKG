from alignment import absolute, alignFloatInsDelWithMetrics

metric = absolute

rule = {'=': 0, '+': 1, '-': 2}
a = [[0.91], [0.92], [0.83], [0.88]]
b = [[0.89], [0.82], [0.87], [0.78]]
threshold = 0.03
ruleAB = alignFloatInsDelWithMetrics(rule, a, b, threshold, metric)
print(ruleAB)