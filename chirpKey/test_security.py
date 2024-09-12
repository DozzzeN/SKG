from sympy import symbols, Eq, solve

# 定义未知数
a, b, c = symbols('a b c')
u_11, u_21, u_31 = symbols('u_11 u_21 u_31')
# u_12, u_22, u_32 = symbols('u_12 u_22 u_32')
# u_13, u_23, u_33 = symbols('u_13 u_23 u_33')

# 定义已知数
b_1 = 3
b_2 = 3
b_3 = 3

# 定义方程组
eq1 = Eq(a * u_11 + c * u_21 + b * u_31, b_1)
eq2 = Eq(b * u_11 + a * u_21 + c * u_31, b_2)
eq3 = Eq(c * u_11 + b * u_21 + a * u_31, b_3)

eq4 = Eq(a * a + b * b + c * c - u_11 * u_11 - u_21 * u_21 - u_31 * u_31, 0)
# eq5 = Eq(a * b + a * c + b * c - u_11 * u_12 - u_21 * u_22 - u_31 * u_32, 0)
# eq6 = Eq(a * b + a * c + b * c - u_11 * u_13 - u_21 * u_23 - u_31 * u_33, 0)
# eq7 = Eq(b * b + a * a + c * c - u_12 * u_12 - u_22 * u_22 - u_32 * u_32, 0)
# eq8 = Eq(b * c + a * b + a * c - u_12 * u_13 - u_22 * u_23 - u_32 * u_33, 0)
# eq9 = Eq(c * c + a * a + b * b - u_13 * u_13 - u_23 * u_23 - u_33 * u_33, 0)

# 解方程组
solutions = solve((eq1, eq2, eq3, eq4), (u_11, u_21, u_31, a, b, c))
# solutions = solve((eq1, eq2, eq3, eq4, eq5, eq6, eq7, eq8, eq9),
#                   (u_11, u_21, u_31, u_12, u_22, u_32, u_13, u_23, u_33, a, b, c))
# 打印解
print(solutions)