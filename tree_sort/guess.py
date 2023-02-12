from math import factorial

n = 64
print(2 ** n)
# print(factorial(n))
# print(factorial(int(n / 2)))

guess = 1
i = 2
while i <= int(n / 4):
    guess *= factorial(i)
    i = i * 2
print(guess)
