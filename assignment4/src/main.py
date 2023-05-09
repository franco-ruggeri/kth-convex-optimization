#!/usr/bin/env python3

import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
from maxsumlog_solver import MaxSumLogSolver


def generate_problem(m, n):
    A = np.random.rand(n, m)
    b = m * np.random.rand(n)
    c = np.random.rand(m)
    return A, b, c


# Reference implementation, using cvxpy
def solve_cvx(A, b, c):
    x = cp.Variable(c.size)
    objective = cp.Maximize(cp.sum(cp.log1p(cp.multiply(c, x))))
    constraints = [A @ x <= b, x >= 0]
    problem = cp.Problem(objective, constraints)
    problem.solve(solver='SCS')
    y = np.concatenate((constraints[0].dual_value, constraints[1].dual_value))
    return x.value, problem.value, y


def plot_surface(m_grid, n_grid, z_grid, z_label):
    figure, axes = plt.subplots(subplot_kw={'projection': '3d'})
    plot = axes.plot_surface(m_grid, n_grid, z_grid, cmap='summer', rstride=1, cstride=1, alpha=None, antialiased=True)
    axes.set_xlabel('m')
    axes.set_ylabel('n')
    axes.set_zlabel(z_label)
    figure.colorbar(plot, location="left")
    figure.show()


def part_b():
    np.random.seed(5)
    m = 100      # test different values!
    n = 20       # test different values!
    A, b, c = generate_problem(m, n)

    solver = MaxSumLogSolver(verbose=True)
    start = datetime.now()
    x, opt_value, y, n_iterations = solver.solve(A, b, c)
    end = datetime.now()
    elapsed_time = (end-start).total_seconds() * 1000
    print(type(end - start))
    print(f'Solution time: {elapsed_time:.1f} ms')
    print(f'Number of iterations: {n_iterations}')
    print(f'Number of non-zeros in x: {np.count_nonzero(x > 1.0e-5)}')

    start = datetime.now()
    x_cvx, obj_cvx, y_cvx = solve_cvx(A, b, c)
    end = datetime.now()
    elapsed_time = (end - start).total_seconds() * 1000
    print(f'CVX solution time: {elapsed_time:.1f} ms')

    print(f'Relative difference in objective: {np.abs(obj_cvx - opt_value) / obj_cvx:.2g}')
    print(f'Relative difference in x: {np.linalg.norm(x_cvx - x) / np.linalg.norm(x_cvx):.2g}')
    print(f'Relative difference in y: {np.linalg.norm(y_cvx - y) / np.linalg.norm(y_cvx):.2g}')


def part_c():
    np.random.seed(5)
    m_min, m_max = 2, 100
    n_min, n_max = 2, 25

    m_grid, n_grid = np.meshgrid(
        np.arange(m_min, m_max),
        np.arange(n_min, n_max),
        indexing='ij'
    )
    n_non_zeros_grid = np.zeros(m_grid.shape)
    n_iterations_grid = np.zeros(m_grid.shape)

    solver = MaxSumLogSolver()
    progress_bar = tqdm(total=m_grid.shape[0]*m_grid.shape[1])
    for i in range(m_grid.shape[0]):
        for j in range(m_grid.shape[1]):
            m = m_grid[i, j]
            n = n_grid[i, j]
            A, b, c = generate_problem(m, n)
            x, _, _, n_iterations = solver.solve(A, b, c)

            n_non_zeros_grid[i, j] = np.count_nonzero(x > 1.0e-5)
            n_iterations_grid[i, j] = n_iterations

            progress_bar.update(1)
    progress_bar.close()

    plot_surface(m_grid, n_grid, n_iterations_grid, '# iterations')
    plot_surface(m_grid, n_grid, n_non_zeros_grid, '# non-zeros')


def main():
    # part_b()
    part_c()


if __name__ == '__main__':
    main()
