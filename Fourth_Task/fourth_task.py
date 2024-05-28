from mshr import *
from fenics import *

import imageio
import numpy as np

import matplotlib.tri as tri
import matplotlib.pyplot as plt


def find_numerical_solution(function_space, alpha, f, h, g):
    def boundary(x, on_boundary):
        return on_boundary

    boundary_conditions = DirichletBC(function_space, h, boundary)

    trial_function = TrialFunction(function_space)
    test_function = TestFunction(function_space)

    a = (dot(grad(trial_function), grad(test_function)) + alpha * trial_function * test_function) * dx
    L = f * test_function * dx + g * test_function * ds

    numerical_solution = Function(function_space)
    solve(a == L, numerical_solution, boundary_conditions)

    return numerical_solution

def find_errors(analytical_solution, numerical_solution):
    l2_error = errornorm(analytical_solution, numerical_solution, 'L2')

    analytical_vertex_values = analytical_solution.compute_vertex_values(generated_mesh)
    numerical_vertex_values = numerical_solution.compute_vertex_values(generated_mesh)
    max_error = np.max(np.abs(analytical_vertex_values - numerical_vertex_values))

    return l2_error, max_error

def make_visualization(mesh, solution):
    n = mesh.num_vertices()
    d = mesh.geometry().dim()

    mesh_coordinates = mesh.coordinates().reshape((n, d))
    triangles = np.asarray([cell.entities(0) for cell in cells(mesh)])
    triangulation = tri.Triangulation(mesh_coordinates[:, 0], mesh_coordinates[:, 1], triangles)

    plt.figure(figsize = (6, 5))
    zfaces = np.asarray([solution(cell.midpoint()) for cell in cells(mesh)])
    plt.tripcolor(triangulation, facecolors = zfaces, edgecolors = 'k', cmap = 'jet')

    plt.colorbar()
    plt.plot()
    plt.show()

def make_time_visualization(mesh, i, solution, min, max):
    n = mesh.num_vertices()
    d = mesh.geometry().dim()

    mesh_coordinates = mesh.coordinates().reshape((n, d))
    triangles = np.asarray([cell.entities(0) for cell in cells(mesh)])
    triangulation = tri.Triangulation(mesh_coordinates[:, 0], mesh_coordinates[:, 1], triangles)

    plt.figure(figsize = (6, 5))
    zfaces = np.asarray([solution(cell.midpoint()) for cell in cells(mesh)])
    plt.tripcolor(triangulation, facecolors = zfaces, edgecolors = 'k', cmap = 'jet')

    plt.clim(min, max)
    bounds = np.linspace(min, max, 10)

    plt.colorbar(ticks=bounds, boundaries = bounds)
    plt.plot()

    plt.savefig(f"{i}.png", bbox_inches = "tight")
    plt.close()


# TASK 1

r = Expression("sqrt(x[0] * x[0] + x[1] * x[1])", degree = 2)
phi = Expression("atan2(x[1], x[0])", degree = 2)

circle = Circle(Point(0, 0), 1)
generated_mesh = generate_mesh(circle, 50)
function_space = FunctionSpace(generated_mesh, 'P', 2)

alpha = Constant(1)


# First Case

f = Expression("alpha * r * sin(phi) + alpha", alpha = alpha, r = r, phi = phi, degree = 2)
h = Expression("1 * sin(phi) + 1", phi = phi, degree = 2)
g = Expression("sin(phi)", phi = phi, degree = 2)

analytical_solution = Expression("r * sin(phi) + 1", r = r, phi = phi, degree = 2)

numerical_solution = find_numerical_solution(function_space, alpha, f, h, g)
l2_error, max_error = find_errors(analytical_solution, numerical_solution)

print("L2 error =", l2_error)
print("MAX error =", max_error)

make_visualization(generated_mesh, numerical_solution)
make_visualization(generated_mesh, analytical_solution)


# Second Case

f = Expression("-5*r*cos(2*phi)-3*sin(phi)+alpha * r * r * r * cos(2*phi)+alpha*r*r*sin(phi)", alpha = alpha, r = r, phi = phi, degree = 2)
h = Expression("cos(2*phi)+sin(phi)", phi = phi, degree = 2)
g = Expression("3*cos(2*phi)+2*sin(phi)", phi = phi, degree = 2)

analytical_solution = Expression("r * r * r * cos(2*phi)+r*r*sin(phi)", r = r, phi = phi, degree = 2)

numerical_solution = find_numerical_solution(function_space, alpha, f, h, g)
l2_error, max_error = find_errors(analytical_solution, numerical_solution)

print("L2 error =", l2_error)
print("MAX error =", max_error)

make_visualization(generated_mesh, numerical_solution)
make_visualization(generated_mesh, analytical_solution)


# Third Case

f = Expression("alpha*pow(r,4)* sin(3*phi)*cos(2*phi)-16*r*r*sin(3*phi)*cos(2*phi)+r*r*12*sin(2*phi)*cos(3*phi)+r*r*13*sin(3*phi)*cos(2*phi)", alpha = alpha, r = r, phi = phi, degree = 2)
h = Expression("sin(3*phi)*cos(2*phi)", phi = phi, degree = 2)
g = Expression("4*sin(3*phi)*cos(2*phi)", phi = phi, degree = 2)

analytical_solution = Expression("pow(r,4)* sin(3*phi)*cos(2*phi)", r = r, phi = phi, degree = 2)

numerical_solution = find_numerical_solution(function_space, alpha, f, h, g)
l2_error, max_error = find_errors(analytical_solution, numerical_solution)

print("L2 error =", l2_error)
print("MAX error =", max_error)

make_visualization(generated_mesh, numerical_solution)
make_visualization(generated_mesh, analytical_solution)


# TASK 2

r = Expression("sqrt(x[0] * x[0] + x[1] * x[1])", degree = 2)
phi = Expression("atan2(x[1], x[0])", degree = 2)

circle = Circle(Point(0, 0), 1)
generated_mesh = generate_mesh(circle, 50)
function_space = FunctionSpace(generated_mesh, 'P', 2)

alpha = Constant(1)

time = 5.0
number_of_time_steps = 50
time_step = time / number_of_time_steps


# First Case

start = 0

l2_errors = []
max_errors = []

f = Expression("r * cos(phi)", t = start, r = r, phi = phi, degree = 2)
h = Expression("t * cos(phi) + 1", t = start, phi = phi, degree = 2)
g = Expression("t * cos(phi)", t = start, phi = phi, degree = 2)

analytical_solution = Expression("t * r * cos(phi) + 1", t = start, r = r, phi = phi, degree = 2)
analytical_functions = [Expression("t * r * cos(phi) + 1", t = start, r = r, phi = phi, degree = 2)]

discreteU = interpolate(h, function_space)
numerical_solutions = [discreteU.copy()]

for n in range(number_of_time_steps):
    start += time_step
    analytical_solution.t = start

    f.t = start
    h.t = start
    g.t = start
    
    numerical_solution = find_numerical_solution(function_space, 1.0 / (time_step * alpha), discreteU / (time_step * alpha) + f / alpha, h, g) 
    discreteU.assign(numerical_solution)

    numerical_solutions.append(discreteU.copy())
    analytical_functions.append(Expression("t * r * cos(phi) + 1", t = start, r = r, phi = phi, degree = 2)) 

    l2_error, max_error = find_errors(analytical_solution, numerical_solution)

    l2_errors.append(l2_error)
    max_errors.append(max_error)

max = 0
min = 100

for i, numerical_solution in enumerate(numerical_solutions): 
        n = generated_mesh.num_vertices()
        d = generated_mesh.geometry().dim()
        mesh_coordinates = generated_mesh.coordinates().reshape((n, d))
        zfaces = np.asarray([numerical_solution(cell.midpoint()) for cell in cells(generated_mesh)])
        
        minimum = np.amin(zfaces)
        maximum = np.amax(zfaces)
        if (minimum < min):
            min = minimum
        if (maximum > max):
            max = maximum

for i in range(number_of_time_steps):
    make_time_visualization(generated_mesh, i, numerical_solutions[i], min, max)

with imageio.get_writer("1.1_numerical.gif", mode = 'i', duration = 0.3) as writer:
    for i in range(number_of_time_steps):
        image = imageio.imread(f"{i}.png")
        writer.append_data(image)

max = 0
min = 100

for i, numerical_solution in enumerate(analytical_functions): 
        n = generated_mesh.num_vertices()
        d = generated_mesh.geometry().dim()
        mesh_coordinates = generated_mesh.coordinates().reshape((n, d))
        zfaces = np.asarray([numerical_solution(cell.midpoint()) for cell in cells(generated_mesh)])
        
        minimum = np.amin(zfaces)
        maximum = np.amax(zfaces)
        if (minimum < min):
            min = minimum
        if (maximum > max):
            max = maximum       

for i in range(number_of_time_steps):
    make_time_visualization(generated_mesh, i, analytical_functions[i], min, max)

with imageio.get_writer("1.2_analytical.gif", mode = 'i', duration = 0.3) as writer:
    for i in range(number_of_time_steps):
        image = imageio.imread(f"{i}.png")
        writer.append_data(image)

plt.plot(l2_errors, label='L2')
plt.plot(max_errors, label='MAX')
plt.legend()
plt.show()


# Second Case

start = 0

l2_errors = []
max_errors = []

f = Expression("cos(t)-9*alpha*r*cos(2*phi)+4*alpha*r*r*r*cos(2*phi)", t = start, r = r, phi = phi, alpha = alpha, degree = 2)
h = Expression("cos(2*phi)+sin(t)", t = start, phi = phi, degree = 2)
g = Expression("3*cos(2*phi)", t = start, phi = phi, degree = 2)

analytical_solution = Expression("r*r*r*cos(2*phi)+sin(t)", t = start, r = r, phi = phi, degree = 2)
analytical_functions = [Expression("r*r*r*cos(2*phi)+sin(t)", t = start, r = r, phi = phi, degree = 2)]

discreteU = interpolate(h, function_space)
numerical_solutions = [discreteU.copy()]

for n in range(number_of_time_steps):
    start += time_step
    analytical_solution.t = start

    f.t = start
    h.t = start
    g.t = start

    numerical_solution = find_numerical_solution(function_space, 1.0 / (time_step * alpha), discreteU / (time_step * alpha) + f / alpha, h, g)
    discreteU.assign(numerical_solution)

    numerical_solutions.append(discreteU.copy())
    analytical_functions.append(Expression("r*r*r*cos(2*phi)+sin(t)", t = start, r = r, phi = phi, degree = 2))

    l2_error, max_error = find_errors(analytical_solution, numerical_solution)

    l2_errors.append(l2_error)
    max_errors.append(max_error)

max = 0
min = 100

for i, numerical_solution in enumerate(numerical_solutions): 
        n = generated_mesh.num_vertices()
        d = generated_mesh.geometry().dim()
        mesh_coordinates = generated_mesh.coordinates().reshape((n, d))
        zfaces = np.asarray([numerical_solution(cell.midpoint()) for cell in cells(generated_mesh)])
        
        minimum = np.amin(zfaces)
        maximum = np.amax(zfaces)
        if (minimum < min):
            min = minimum
        if (maximum > max):
            max = maximum

for i in range(number_of_time_steps):
    make_time_visualization(generated_mesh, i, numerical_solutions[i], min, max)

with imageio.get_writer("2.1_numerical.gif", mode = 'i', duration = 0.3) as writer:
    for i in range(number_of_time_steps):
        image = imageio.imread(f"{i}.png")
        writer.append_data(image)

max = 0
min = 100

for i, numerical_solution in enumerate(analytical_functions): 
        n = generated_mesh.num_vertices()
        d = generated_mesh.geometry().dim()
        mesh_coordinates = generated_mesh.coordinates().reshape((n, d))
        zfaces = np.asarray([numerical_solution(cell.midpoint()) for cell in cells(generated_mesh)])
        
        minimum = np.amin(zfaces)
        maximum = np.amax(zfaces)
        if (minimum < min):
            min = minimum
        if (maximum > max):
            max = maximum       

for i in range(number_of_time_steps):
    make_time_visualization(generated_mesh, i, analytical_functions[i], min, max)

with imageio.get_writer("2.2_analytical.gif", mode = 'i', duration = 0.3) as writer:
    for i in range(number_of_time_steps):
        image = imageio.imread(f"{i}.png")
        writer.append_data(image)

plt.plot(l2_errors, label='L2')
plt.plot(max_errors, label='MAX')
plt.legend()
plt.show()


# Third Case

start = 0

l2_errors = []
max_errors = []

f = Expression("-2*pow(r,4)*sin(3*phi)*sin(2*t)-alpha*16*r*r*sin(3*phi)*cos(2*t)+alpha*9*pow(r,4)*sin(3*phi)*cos(2*t)", t = start, r = r, phi = phi, alpha = alpha, degree = 2)
h = Expression("sin(3*phi)*cos(2*t)", t = start, phi = phi, degree = 2)
g = Expression("4*sin(3*phi)*cos(2*t)", t = start, phi = phi, degree = 2)

analytical_solution = Expression("pow(r,4)*sin(3*phi)*cos(2*t)", t = start, r = r, phi = phi, degree = 2)
analytical_functions = [Expression("pow(r,4)*sin(3*phi)*cos(2*t)", t = start, r = r, phi = phi, degree = 2)]

discreteU = interpolate(h, function_space)
numerical_solutions = [discreteU.copy()]

for n in range(number_of_time_steps):
    start += time_step
    analytical_solution.t = start

    f.t = start
    h.t = start
    g.t = start

    numerical_solution = find_numerical_solution(function_space, 1.0 / (time_step * alpha), discreteU / (time_step * alpha) + f / alpha, h, g)  
    discreteU.assign(numerical_solution)

    numerical_solutions.append(discreteU.copy())
    analytical_functions.append(Expression("pow(r,4)*sin(3*phi)*cos(2*t)", t = start, r = r, phi = phi, degree = 2)) 

    l2_error, max_error = find_errors(analytical_solution, numerical_solution)

    l2_errors.append(l2_error)
    max_errors.append(max_error)

max = 0
min = 100

for i, numerical_solution in enumerate(numerical_solutions): 
        n = generated_mesh.num_vertices()
        d = generated_mesh.geometry().dim()
        mesh_coordinates = generated_mesh.coordinates().reshape((n, d))
        zfaces = np.asarray([numerical_solution(cell.midpoint()) for cell in cells(generated_mesh)])
        
        minimum = np.amin(zfaces)
        maximum = np.amax(zfaces)
        if (minimum < min):
            min = minimum
        if (maximum > max):
            max = maximum

for i in range(number_of_time_steps):
    make_time_visualization(generated_mesh, i, numerical_solutions[i], min, max)

with imageio.get_writer("3.1_numerical.gif", mode = 'i', duration = 0.3) as writer:
    for i in range(number_of_time_steps):
        image = imageio.imread(f"{i}.png")
        writer.append_data(image)

max = 0
min = 100

for i, numerical_solution in enumerate(analytical_functions): 
        n = generated_mesh.num_vertices()
        d = generated_mesh.geometry().dim()
        mesh_coordinates = generated_mesh.coordinates().reshape((n, d))
        zfaces = np.asarray([numerical_solution(cell.midpoint()) for cell in cells(generated_mesh)])
        
        minimum = np.amin(zfaces)
        maximum = np.amax(zfaces)
        if (minimum < min):
            min = minimum
        if (maximum > max):
            max = maximum       

for i in range(number_of_time_steps):
    make_time_visualization(generated_mesh, i, analytical_functions[i], min, max)

with imageio.get_writer("3.2_analytical.gif", mode = 'i', duration = 0.3) as writer:
    for i in range(number_of_time_steps):
        image = imageio.imread(f"{i}.png")
        writer.append_data(image)

plt.plot(l2_errors, label='L2')
plt.plot(max_errors, label='MAX')
plt.legend()
plt.show()
