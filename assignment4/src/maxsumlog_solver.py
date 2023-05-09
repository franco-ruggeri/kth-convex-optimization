import numpy as np
from tqdm import tqdm


class MaxSumLogSolver:
    def __init__(self, sigma=.1, beta=.8, eps=1e-5):
        """Init maxsumlog solver.

        :param sigma: Reduction factor for barrier parameter. Default: 0.1 (long-shot).
        :param beta: Reduction factor for step length in backtrace search.
        :param eps: Tolerance for duality gap.
        """
        self.sigma = sigma
        self.beta = beta
        self.eps = eps

    @staticmethod
    def _is_strictly_feasible(x, y, A, b):
        primal_strictly_feasible = np.all(A @ x - b < 0) and np.all(x > 0)
        dual_strictly_feasible = np.all(y > 0)
        return primal_strictly_feasible and dual_strictly_feasible

    def solve(self, A, b, c):
        """Solve max sum log optimization problem with primal-dual interior method.

        :param A: n x m matrix with non-negative entries.
        :param b: n x 1 vector with positive entries.
        :param c: m x 1 vector with non-negative entries.
        :return: tuple (x, opt_value, y, n_iterations), where:
            - x is optimal primal solution
            - opt_value is the optimal objective value
            - y is the optimal dual solution corresponding to Ax<=b and x>=0
            - n_iterations is the number of iterations
        """
        n, m = A.shape
        l = n + m
        assert b.shape == (n,)
        assert c.shape == (m,)
        opt_value = None
        n_iterations = 0
        progress_bar = tqdm()
        d_g = np.vstack([-A, np.eye(m)])                # derivative matrix of constraints

        # Initialize with strictly feasible primal-dual point
        x = 1e-5 * np.ones(m)
        y = 1 * np.ones(l)
        assert self._is_strictly_feasible(x, y, A, b)

        # Primal-dual interior method
        while True:
            # Check termination. Since there are no equality constraints, all iterates are primal-dual feasible
            # (guaranteed by line search) and we can just check the duality gap.
            g = np.concatenate([b - A @ x, x])
            eta = g @ y       # (surrogate) duality gap
            if eta < self.eps:
                break

            # Reduce barrier parameter
            mu = self.sigma * eta / l

            # Compute search direction
            d_f = - c / (1 + x * c)                     # gradient of objective
            d2_f = np.diag((c / (1 + x * c)) ** 2)      # hessian of objective
            A_newton = np.block([
                [d2_f, d_g.T],
                [np.diag(y) @ d_g, -np.diag(g)]
            ])
            b_newton = - np.concatenate([
                d_f - d_g.T @ y,
                np.diag(g) @ y - mu
            ])
            delta = np.linalg.solve(A_newton, b_newton)
            delta_x = delta[:m]
            delta_y = -delta[m:]

            # Line search (backtracking search)
            alpha = 1
            while not self._is_strictly_feasible(x + alpha * delta_x, y + alpha * delta_y, A, b):
                alpha *= self.beta
            x += alpha * delta_x
            y += alpha * delta_y

            # Print stats
            n_iterations += 1
            opt_value = np.log(1 + x * c).sum()
            progress_bar.update(n_iterations)
            progress_bar.set_description(
                f'mu: {mu:.2f}'
                f' - surrogate duality gap: {eta:.2f}'
                f' - objective: {opt_value:.2f}'
            )

        progress_bar.close()
        return x, opt_value, y, n_iterations
