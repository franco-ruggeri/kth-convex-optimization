#!/usr/bin/env python3

import numpy as np
import cvxpy as cp
from datetime import datetime
from maxsumlog_solver import MaxSumLogSolver


# Reference implementation, using cvxpy
def solve_cvx(A, b, c):
    x = cp.Variable(c.size)
    objective = cp.Maximize(cp.sum(cp.log1p(cp.multiply(c, x))))
    constraints = [A @ x <= b, x >= 0]
    problem = cp.Problem(objective, constraints)
    problem.solve()
    y = np.concatenate((constraints[0].dual_value, constraints[1].dual_value))
    return x.value, problem.value, y


def solve_maxsumlog(A, b, c):
    solver = MaxSumLogSolver()
    return solver.solve(A, b, c)


def main():
    np.random.seed(5)

    m = 10      # test different values!
    n = 5       # test different values!

    A = np.array([
        [8   ,  1    , 8   , 14  ,  16  ,  18   ,  2   ,  6    ,20   ,  9],
        [15  ,   3   , 14  ,   8  ,  20  ,   1  ,   8  ,  14   , 15  ,  19],
        [0   ,  7   ,  4   , 11   ,  6   ,  0   , 20  ,  17    , 5  ,   6],
        [6   ,  8   , 18   ,  2   , 14  ,   3  ,  11  ,   0    ,16  ,   6],
        [3   , 11    , 0   ,  4   , 18  ,  18  ,  14   , 15   ,  2   ,  2],
    ])
    b = np.array([1, 14, 5, 6, 10])
    c = np.array([0,6,1,6,7,1,4,7,4,0])

    # A = np.random.rand(n, m)
    # b = m * np.random.rand(n)
    # c = np.random.rand(m)

    start = datetime.now()
    x, obj, y, it = solve_maxsumlog(A, b, c)
    end = datetime.now()
    print(f'Solution time: {end - start}')
    print(f'Number of iterations: {it}')
    print(f'Number of non-zeros in x: %d' % np.count_nonzero(x > 1.0e-5))

    start = datetime.now()
    x_cvx, obj_cvx, y_cvx = solve_cvx(A, b, c)
    end = datetime.now()
    print(f'CVX solution time: {end - start}')

    print(f'Relative difference in objective: {np.abs(obj_cvx - obj) / obj_cvx:.12g}')
    print(f'Relative difference in x: {np.linalg.norm(x_cvx - x) / np.linalg.norm(x_cvx):.12g}')
    print(f'Relative difference in y: {np.linalg.norm(y_cvx - y) / np.linalg.norm(y_cvx):.12g}')


if __name__ == '__main__':
    main()
