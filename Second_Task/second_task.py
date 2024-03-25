import numpy as np
import matplotlib.pyplot as plt

from sympy import Symbol, solve, lambdify, Matrix

kp1 = Symbol("kp1")
km1 = Symbol("km1")
kp2 = Symbol("kp2")
kp3 = Symbol("kp3")
km3 = Symbol("km3")

x = Symbol("x")
y = Symbol("y")

eq1 = kp1 * (1 - x - y) - km1 * x - kp2 * ((1 - x - y) ** 2) * x
eq2 = kp3 * (1 - x - y) ** 2 - km3 * (y ** 2)

res = solve([eq1, eq2], x, kp1)
x_eq = res[0][0]
kp1_eq = res[0][1]
# kp1_diff_eq = diff(res[0][1], y)

x_func = lambdify((y, km3, kp3), x_eq)
kp1_func = lambdify((y, km1, kp2, kp3, km3), kp1_eq)
# kp1_diff_func = lambdify((y, km1, kp2, kp3, km3), kp1_diff_eq)

kp1_val = 0.12
km1_val = 0.005
kp2_val = 1.05
kp3_val = 0.0032
km3_val = 0.002

x_f = x_func(np.linspace(0, 1, 1001), km3_val, kp3_val)
func = kp1_func(np.linspace(0, 1, 1001), km1_val, kp2_val, kp3_val, km3_val)
# func_diff = kp1_diff_func(np.linspace(0, 1, 1001), km1_val, kp2_val, kp3_val, km3_val)

plt.plot(np.linspace(0, 1, 1001), x_f)
plt.xlabel("y")
plt.ylabel("x")
plt.show()

plt.plot(np.linspace(0, 1, 1001), func)
plt.xlabel("y")
plt.ylabel("k_1")
plt.show()

A = Matrix([eq1, eq2])
var_vector = Matrix([x, y])
jacA = A.jacobian(var_vector)
det_jacA = jacA.det()
trace_jacA = jacA.trace()

# trace_func = lambdify((x, y, kp1, km1, kp2, kp3, km3) , trace_jacA)

trace_jacA_wo_x = trace_jacA.subs(x, x_eq) # линия нейтральности
det_jacA_wo_x = det_jacA.subs(x, x_eq) # линия кратности

det_jacA_wo_x_func = lambdify((y, kp1, km1, kp2, kp3, km3), det_jacA_wo_x)
det_jacA_wo_x_f = det_jacA_wo_x_func(np.linspace(0, 1, 1001), kp1_val, km1_val, kp2_val, kp3_val, km3_val)

point = 0
for i in range(len(det_jacA_wo_x_f) - 1):
    if det_jacA_wo_x_f[i] < 0 and det_jacA_wo_x_f[i+1] > 0:
        point = i

# plt.plot(np.linspace(0, 1, 1001), det_jacA_wo_x_f)
# plt.xlabel("y")
# plt.ylabel("det A")
# plt.show()

plt.plot(func, x_f, label='x')
plt.plot(func, np.linspace(0, 1, 1001), label='y')
plt.plot(func[point], x_f[point], '*')
plt.plot(func[point], np.linspace(0, 1, 1001)[point], '*')
plt.xlabel("k_1")
plt.legend()
plt.show()

res = solve([trace_jacA_wo_x], kp1)
kp1_again_from_trace = res[kp1]

res = solve([det_jacA_wo_x], kp1)
kp1_again_from_det = res[kp1]

res1 = solve([kp1_again_from_trace - kp1_eq], km1)
km1_again_from_trace_tilda = res1[km1]

res2 = solve([kp1_again_from_det - kp1_eq], km1)
km1_again_from_det_tilda_tilda = res2[km1]

km1_from_trace_func = lambdify((y, kp1, kp2, km3, kp3), km1_again_from_trace_tilda)
km1_from_dim_func = lambdify((y, kp1, kp2, km3, kp3), km1_again_from_det_tilda_tilda)

km1_from_trace_func_f = km1_from_trace_func(np.linspace(0.01, 0.09, 1001), kp1_val, kp2_val, km3_val, kp3_val)
km1_from_dim_func_f = km1_from_dim_func(np.linspace(0.01, 0.09, 1001), kp1_val, kp2_val, km3_val, kp3_val)

plt.plot(np.linspace(0.01, 1, 1001), km1_from_trace_func_f)
plt.xlabel("y")
plt.ylabel("km1_from_trace_func_f")
plt.show()

plt.plot(np.linspace(0.01, 1, 1001), km1_from_dim_func_f)
plt.xlabel("y")
plt.ylabel("km1_from_dim_func_f")
plt.show()

plt.plot(func, km1_from_trace_func_f)
plt.xlabel("kp1")
plt.ylabel("km1_from_trace")
plt.show()

plt.plot(func, km1_from_dim_func_f)
plt.xlabel("kp1")
plt.ylabel("km1_from_dim")
plt.show()
