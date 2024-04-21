import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import odeint
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

x_lambdify = lambdify((y, km3, kp3), x_eq)
kp1_lambdify = lambdify((y, km1, kp2, kp3, km3), kp1_eq)

kp1_val = 0.12
km1_val = 0.005
kp2_val = 1.05
kp3_val = 0.0032
km3_val = 0.002

x_function = x_lambdify(np.linspace(0.01, 1, 1000), km3_val, kp3_val)
kp1_function = kp1_lambdify(np.linspace(0.01, 1, 1000), km1_val, kp2_val, kp3_val, km3_val)

plt.plot(np.linspace(0.01, 1, 1000), x_function, label='x')
plt.plot(np.linspace(0.01, 1, 1000), kp1_function, label='k_1')
plt.title("Зависимости функций x и k_1 от у")
plt.xlabel("y")
plt.legend()
plt.show()

matrix = Matrix([eq1, eq2])
jacobian = matrix.jacobian(Matrix([x, y]))

det_jacobian = jacobian.det()
trace_jacobian = jacobian.trace()

trace_jacobian_wo_x = trace_jacobian.subs(x, x_eq) # линия нейтральности
det_jacobian_wo_x = det_jacobian.subs(x, x_eq) # линия кратности

det_jacobian_wo_x_lambdify = lambdify((y, kp1, km1, kp2, kp3, km3), det_jacobian_wo_x)
det_jacobian_wo_x_function = det_jacobian_wo_x_lambdify(np.linspace(0.01, 1, 1000), kp1_val, km1_val, kp2_val, kp3_val, km3_val)

trace_jacobian_wo_x_lambdify = lambdify((y, kp1, km1, kp2, kp3, km3), trace_jacobian_wo_x)
trace_jacobian_wo_x_function = trace_jacobian_wo_x_lambdify(np.linspace(0.01, 1, 1000), kp1_val, km1_val, kp2_val, kp3_val, km3_val)

point1 = 0
point2 = 0

for i in range(len(det_jacobian_wo_x_function) - 1):
    if det_jacobian_wo_x_function[i] < 0 and det_jacobian_wo_x_function[i+1] > 0:
        point1 = i

    if det_jacobian_wo_x_function[i] > 0 and det_jacobian_wo_x_function[i+1] < 0:
        point2 = i

plt.plot(kp1_function, x_function, label='x')
plt.plot(kp1_function, np.linspace(0.01, 1, 1000), label='y')

plt.plot(kp1_function[point1], x_function[point1], '*')
plt.plot(kp1_function[point1], np.linspace(0.01, 1, 1000)[point1], '*')

plt.plot(kp1_function[point2], x_function[point2], '*')
plt.plot(kp1_function[point2], np.linspace(0.01, 1, 1000)[point2], '*')

x_first = x_function[point1]
y_first = np.linspace(0.01, 1, 1000)[point1]

x_second = x_function[point2]
y_second = np.linspace(0.01, 1, 1000)[point2]

plt.xlabel("k_1")
plt.title("Зависимость стационарных состояний от параметра k_1")
plt.xlim([0, 0.3])
plt.ylim([-0.1, 1])
plt.legend()
plt.show()

def make_system_1(initial_vals, time):
    x = initial_vals[0]
    y = initial_vals[1]
    dx_dt = kp1_function[point1] * (1 - x - y) - km1_val * x - kp2_val * ((1 - x - y) ** 2) * x
    dy_dt = kp3_val * (1 - x - y) ** 2 - km3_val * (y ** 2)
    return [dx_dt, dy_dt]

def make_system_2(initial_vals, time):
    x = initial_vals[0]
    y = initial_vals[1]
    dx_dt = kp1_function[point2] * (1 - x - y) - km1_val * x - kp2_val * ((1 - x - y) ** 2) * x
    dy_dt = kp3_val * (1 - x - y) ** 2 - km3_val * (y ** 2)
    return [dx_dt, dy_dt]

initial_values = [x_first, y_first]
t = np.linspace(0, 5000, 50000)
result = odeint(make_system_1, initial_values, t)

plt.plot(t, result[:, 0], label='x')
plt.plot(t, result[:, 1], label='y')
plt.title("Численное решение при k_1 =" + str(kp1_function[point1]))
plt.xlabel("t")
plt.show()

plt.plot(result[:, 0], result[:, 1])
plt.title("Результат №1 в фазовой плоскости")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

initial_values = [x_second, y_second]
t = np.linspace(0, 5000, 50000)
result = odeint(make_system_2, initial_values, t)

plt.plot(t, result[:, 0], label='x')
plt.plot(t, result[:, 1], label='y')
plt.title("Численное решение при k_1 =" + str(kp1_function[point2]))
plt.xlabel("t")
plt.legend()
plt.show()

plt.plot(result[:, 0], result[:, 1])
plt.title("Результат №2 в фазовой плоскости")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()

res = solve([trace_jacobian_wo_x], kp1)
kp1_again_from_trace = res[kp1]

res = solve([det_jacobian_wo_x], kp1)
kp1_again_from_det = res[kp1]

res1 = solve([kp1_again_from_trace - kp1_eq], km1)
km1_from_trace = res1[km1]

res2 = solve([kp1_again_from_det - kp1_eq], km1)
km1_from_det = res2[km1]

km1_from_trace_lambdify = lambdify((y, kp1, kp2, km3, kp3), km1_from_trace)
km1_from_det_lambdify = lambdify((y, kp1, kp2, km3, kp3), km1_from_det)

km1_from_trace_function = km1_from_trace_lambdify(np.linspace(0.01, 0.9, 1000), kp1_val, kp2_val, km3_val, kp3_val)
km1_from_det_function = km1_from_det_lambdify(np.linspace(0.01, 0.9, 1000), kp1_val, kp2_val, km3_val, kp3_val)

plt.plot(kp1_function, km1_from_trace_function, label='k_(-1) на линии кратности')
plt.plot(kp1_function, km1_from_det_function, label='k_(-1) на линии нейтральности')
plt.title("Параметрический портрет исходной системы")
plt.xlabel("k_1")
plt.xlim([0.118, 0.130])
plt.ylim([0.0, 0.02])
plt.legend()
plt.show()

kp1_point = 0.125
km1_point = 0.005

def make_system_3(initial_vals, time):
    x = initial_vals[0]
    y = initial_vals[1]
    dx_dt = kp1_point * (1 - x - y) - km1_point * x - kp2_val * ((1 - x - y) ** 2) * x
    dy_dt = kp3_val * (1 - x - y) ** 2 - km3_val * (y ** 2)
    return [dx_dt, dy_dt]

initial_values = [x_first, y_first]
t = np.linspace(0, 1000, 10000)
result = odeint(make_system_3, initial_values, t)

plt.plot(t, result[:, 0], label='x')
plt.plot(t, result[:, 1], label='y')
plt.title("Численное решение при k_1 = " + str(kp1_point) + " и k_(-1) =" + str(km1_point))
plt.xlabel("t")
plt.legend()
plt.show()

plt.plot(result[:, 0], result[:, 1])
plt.title("Фазовый портрет системы в области автоколебаний")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
