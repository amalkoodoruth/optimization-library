import numpy as np
from functools import partial
from optimization_utils import get_step_size,check_cone_condition,get_cost_norm,get_fd_hessian,get_fd_grad

class Problem:
    """
    This class defines an optimization problem
    """
    def __init__(self, cost, grad='fd', grad_step=1e-8,
                 eq_const=None, ineq_const=None):
        """
        Constructor for the optimization problem

        :param cost: function with 2D NumPy array as input
            Objective function of the problem
        :param grad: function with 2D NumPy array as input
            Gradient function of the problem.
            Default = 'fd', finite difference used
        :param grad_step: float
            Step size for finite difference gradient.
            Only required if :param grad= 'fd'.
            Default = 1e-8
        :param eq_const: list of functions
            Equality constraints of the problem
        :param ineq_const: list of functions
            Inequality constraints of the problem
        """
        self._cost = cost

        if grad == 'fd':
            # finite difference method specified
            self._grad_step = grad_step  # default 1e-8
            self._grad = partial(get_fd_grad, self.get_cost, h=self._grad_step)

        else:
            # gradient specified, don't need step size
            self._grad_step = None
            self._grad = grad

        self._eq_const = eq_const
        self._ineq_const = ineq_const

    def get_cost(self, x=None):
        """
        Returns the cost of the problem if x is specified.
        Else, returns the cost function.

        :param x: 2D numpy array
            Point at which the cost is calculated.
            Default = None
        :return: function or float
            If x is None, returns the cost function.
            Else, returns the cost at x.
        """
        if x is not None:
            return self._cost(x)
        else:
            return self._cost

    def get_grad(self, x=None):
        """
        Returns the gradient of the problem if x is specified.
        Else, returns the gradient function.

        :param x: 2D numpy array
            Point at which the gradient is calculated.
            Default = None
        :return: function or float
            If x is None, returns the gradient function.
            Else, returns the gradient at x.
        """
        if x is not None:
            return self._grad(x)
        else:
            return self._grad

    def get_eq_const(self, x=None):
        """
        Equality constraints of the problem.
        If x is None, returns an array of functions.
        Else, returns an array of equality constraints evaluated at x

        :param x: 2D numpy array
            Point at which the equality constraints are evaluated.
        :return: 2D numpy array
            Array of functions if x = None.
            Else, array of equality constraints evaluated at x
        """
        if self._eq_const is not None:
            if x is not None:
                return np.array([[eq(x)] for eq in self._eq_const])
            else:
                return np.array([eq for eq in self._eq_const])
        else:
            return None

    def get_ineq_const(self, x=None):
        """
        Inequality constraints of the problem.
        If x is None, returns an array of functions.
        Else, returns an array of inequality constraints evaluated at x

        :param x: 2D numpy array
            Point at which the inequality constraints are evaluated.
        :return: 2D numpy array
            Array of functions if x = None.
            Else, array of inequality constraints evaluated at x
        """
        if self._ineq_const is not None:
            if x is not None:
                return np.array([[ineq(x)] for ineq in self._ineq_const])
            else:
                return np.array([ineq for ineq in self._ineq_const])
        else:
            return None

    def num_eq_const(self):
        """
        The number of equality constraints.

        :return: int
            The number of equality constraints.
        """
        if self._eq_const is not None:
            return np.max(np.shape(self._eq_const))
        else:
            return 0

    def num_ineq_const(self):
        """
        The number of inequality constraints.

        :return: int
            The number of inequality constraints.
        """
        if self._ineq_const is not None:
            return np.max(np.shape(self._ineq_const))
        else:
            return 0

def steepest_descent(p, x, tol=1e-6, max_iter=999, hist=False):
    """
    Steepest Descent algorithm

    :param p: Problem
        Optimization problem to minimize
    :param x: 2D numpy array
        Initial guess
    :param tol: float
        Tolerance of the algorithm. If norm of gradient is less than tol,
        iterations stop.
        Default = 1e-6
    :param max_iter: int
        Maximum number of iterations before stopping
        Default = 999
    :param hist: bool
        Flag to return history of x or not
        Default = False
    :return: 2D or 3D numpy array
        If hist = False, returns the last value of x.
        Else, returns an array with values of x after each iteration
    """
    i = 0
    x_hist = []
    while np.linalg.norm(p.get_grad(x)) > tol:
        if i > max_iter:
            break
        s = -p.get_grad(x).T # get the gradient at x
        w = get_step_size(p, x, s) # get step size using Armijo algorithm
        x_hist.append(x)
        x = x + w * s # gradient descent
        i += 1 # increment iteration

    return x if not hist else np.array(x_hist)

def conjugate_gradient(p, x, tol=1e-6, rst_iter=99, max_iter=999, hist=False):
    """
    Conjugate gradient algorithm

    :param p: Problem
        Optimization problem to minimize
    :param x: 2D numpy array
        Initial guess
    :param tol: float
        Tolerance of the algorithm. If norm of gradient is less than tol,
        iterations stop.
        Default = 1e-6
    :param rst_iter: int
        Number of iterations before resetting search direction
        and restarting
        Default = 99
    :param max_iter: int
        Maximum total number of iterations
        Default = 999
    :param hist: bool
        Flag to return history of x or not
        Default = False
    :return: 2D or 3D numpy array
        If hist = False, returns the last value of x.
        Else, returns an array with values of x after each iteration
    """
    i = 0
    x_hist = []
    s = -p.get_grad(x).T
    while np.linalg.norm(p.get_grad(x)) > tol:
        if i > rst_iter or not check_cone_condition(p, x, s):
            i = 0
            s = -p.get_grad(x).T # get gradient at x
        # elif not check_cone_condition(p, x, s):
        #     i = 0
        #     s = -p.grad(x).T # get gradient at x
        elif i > max_iter:
            break
        w = get_step_size(p, x, s) # get step size using Armijo algorithm
        x_prv = x
        x_hist.append(x)
        x = x_prv + w * s
        beta = ((p.get_grad(x) - p.get_grad(x_prv)) @ p.get_grad(x).T) \
            / (p.get_grad(x_prv) @ p.get_grad(x_prv).T)
        s = -p.get_grad(x).T + beta * s
        i += 1

    return x if not hist else np.array(x_hist)

def secant(p, x, tol=1e-6, H=None, rst_iter=99, max_iter=999, hist=False):
    """
    Secant optimization algorithm

    :param p: Problem
        Optimization problem to minimize
    :param x: 2D numpy array
        Initial guess
    :param tol: float
        Tolerance of the algorithm. If norm of gradient is less than tol,
        iterations stop.
        Default = 1e-6
    :param H: 2D numpy matrix
        Initial guess at Hessian. If None, set to identity
        Default = None
    :param rst_iter: int
        Number of iterations before resetting search direction
        and restarting
        Default = 99
    :param max_iter: int
        Maximum total number of iterations
        Default = 999
    :param hist: bool
        Flag to return history of x or not
        Default = False
    :return: 2D or 3D numpy array
        If hist = False, returns the last value of x.
        Else, returns an array with values of x after each iteration
    """
    if H is None:
        H = np.eye(np.max(np.shape(x)))

    i = 0
    x_hist = []
    while np.linalg.norm(p.get_grad(x)) > tol:
        s = -H @ p.get_grad(x).T
        if i > rst_iter or not check_cone_condition(p, x, s):
            i = 0
            s = -p.get_grad(x).T # get gradient at x
        # elif not check_cone_condition(p, x, s):
        #     i = 0
        #     s = -p.grad(x).T
        elif i > max_iter:
            break
        w = get_step_size(p, x, s)
        x_prv = x
        x_hist.append(x)
        x = x_prv + w * s
        # Davidon-Fletcher-Powell (DFP) Algorithm
        dx = x - x_prv
        dg = p.get_grad(x) - p.get_grad(x_prv)
        H = H + (dx @ dx.T) / (dx.T @ dg.T) \
            - ((H @ dg.T) @ (H @ dg.T).T) / (dg @ H @ dg.T)
        i += 1

    return x if not hist else np.array(x_hist)

def penalty_function(p, x0, tol=1e-6, tol_cost=1e-4, sigma_max=1e6, hist=False):
    """
    Constrained optimization algorithm using penalty functions
    :param p: Problem
        Constrained optimization problem to minimize
    :param x0: 2D numpy array
        Initial guess
    :param tol: float
        Tolerance of the algorithm. If norm of gradient is less than tol,
        iterations stop.
        Default = 1e-6
    :param tol_cost: float
        Tolerance of cost. If norm of cost is less than tol_cost,
        iteration stops.
        Default = 1e-4
    :param sigma_max: float
        Maximum value of sigma before iteration stops
        Default = 1e6
    :param hist: bool
        Flag to return history of x or not
        Default = False
    :return: 2D or 3D numpy array
        If hist = False, returns the last value of x.
        Else, returns an array with values of x after each iteration
    """

    def phi(p, sigma, x):
        cost = p.get_cost(x) # get cost at x
        if p.get_eq_const() is not None:
            cost = cost + 0.5 * sigma * np.linalg.norm(p.get_eq_const(x))**2
        if p.get_ineq_const() is not None:
            ineq_x = p.get_ineq_const(x)
            c = np.minimum(np.zeros(np.shape(ineq_x)), ineq_x)
            cost = cost + 0.5 * sigma * np.linalg.norm(c)**2
        return cost

    # def cost_norm(x):
    #     cost = 0
    #     if p.eq_const() is not None:
    #         cost = cost + np.linalg.norm(p.eq_const(x))**2
    #     if p.ineq_const() is not None:
    #         ineq_x = p.ineq_const(x)
    #         c = np.minimum(np.zeros(np.shape(ineq_x)), ineq_x)
    #         cost = cost + np.linalg.norm(c)**2
    #     return np.sqrt(cost)

    sigma = 1
    x = x0 # initialize first guess
    x_hist = []

    while get_cost_norm(p, x) > tol_cost:
        up = Problem(partial(phi, p, sigma))
        x_hist.append(x)
        x = steepest_descent(up, x0, tol=tol)
        if sigma >= sigma_max:
            break
        sigma *= 10

    return x if not hist else np.array(x_hist)

def barrier_function(p, x0, tol=1e-6, tol_const=1e-4, sigma_max=1e6,
                     r_min=1e-6, mode='inv', hist=False):
    """
    Constrained optimization algorithm using barrier functions
    :param p: Problem
        Constrained optimization problem to minimize
    :param x0: 2D numpy array
        Initial guess
    :param tol: float
        Tolerance of the algorithm. If norm of gradient is less than tol,
        iterations stop.
        Default = 1e-6
    :param tol_cost: float
        Tolerance of cost. If norm of cost is less than tol_cost,
        iteration stops.
        Default = 1e-4
    :param sigma_max: float
        Maximum value of sigma before iteration stops
        Default = 1e6
    :param r_min: float
        Minimum value of r before iteration stops
        Default = 1e-6
    :param mode: String
        Mode of barrier function. 'inv' for inverse barrier function
        or 'log' for logarithmic barrier function
    :param hist: bool
        Flag to return history of x or not
        Default = False
    :return: 2D or 3D numpy array
        If hist = False, returns the last value of x.
        Else, returns an array with values of x after each iteration
    """
    def phi(p, sigma, r, x):
        cost = p.get_cost(x)
        if p.get_eq_const() is not None:
            cost = cost + 0.5 * sigma * np.linalg.norm(p.get_eq_const(x))**2
        if p.get_ineq_const() is not None:
            ineq_x = p.get_ineq_const(x)
            if mode == 'log':
                cost = cost - r * np.sum(np.log(ineq_x))
            else:
                cost = cost + r * np.sum(np.reciprocal(ineq_x))
        return cost

    # def cost_norm(x):
    #     cost = 0
    #     if p.eq_const() is not None:
    #         cost = cost + np.linalg.norm(p.eq_const(x))**2
    #     if p.ineq_const() is not None:
    #         ineq_x = p.ineq_const(x)
    #         c = np.minimum(np.zeros(np.shape(ineq_x)), ineq_x)
    #         cost = cost + np.linalg.norm(c)**2
    #     return np.sqrt(cost)

    sigma = 1
    r = 1
    x = x0
    x_hist = []

    while get_cost_norm(p, x) > tol_const:
        up = Problem(partial(phi, p, sigma, r))
        x_hist.append(x)
        x = steepest_descent(up, x0, tol=tol)
        if sigma >= sigma_max or r <= r_min:
            break
        sigma *= 10
        r *= 0.1

    return x if not hist else np.array(x_hist)

def augmented_lagrange(p, x0, tol=1e-6, tol_cost=1e-6, sigma_max=1e12, hist=False):
    """
    Constrained optimization algorithm using augmented Lagrange method
    :param p: Problem
        Constrained optimization problem to minimize
    :param x0: 2D numpy array
        Initial guess
    :param tol: float
        Tolerance of the algorithm. If norm of gradient is less than tol,
        iterations stop.
        Default = 1e-6
    :param tol_cost: float
        Tolerance of cost. If norm of cost is less than tol_cost,
        iteration stops.
        Default = 1e-6
    :param sigma_max: float
        Maximum value of sigma before iteration stops
        Default = 1e6
    :param hist: bool
        Flag to return history of x or not
        Default = False
    :return: 2D or 3D numpy array
        If hist = False, returns the last value of x.
        Else, returns an array with values of x after each iteration
    """
    def phi(p, lmb, sgm, x):
        cost = p.get_cost(x)

        n_e = p.num_eq_const()
        n_i = p.num_ineq_const()
        n_c = n_e + n_i

        lmb_e = lmb[0:n_e, :]
        lmb_i = lmb[n_e:n_c, :]
        sgm_e = sgm[0:n_e, :]
        sgm_i = sgm[n_e:n_c, :]

        if p.get_eq_const() is not None:
            c_e = p.get_eq_const(x)
            cost = cost - sum(lmb_e * c_e) + 0.5 * sum(sgm_e * c_e**2)

        if p.get_ineq_const() is not None:
            c_i = p.get_ineq_const(x)
            p_i = np.array([-lmb_i[i] * c_i[i] + 0.5 * sgm_i[i] * c_i[i]**2 \
                            if c_i[i] <= lmb_i[i] / sgm_i[i] \
                            else -0.5 * lmb_i[i]**2 / sgm_i[i] \
                            for i in range(0, n_i)])
            cost = cost + sum(p_i)

        return cost

    x_hist = []

    n_e = p.num_eq_const()
    n_i = p.num_ineq_const()
    n_c = n_e + n_i

    lmb = np.zeros((n_c, 1))
    sgm = np.ones((n_c, 1))

    x = x0
    c = 1e12 * np.ones((n_c, 1))

    while np.linalg.norm(c) > tol_cost:
        # Create new problem to solve, but unconstrained
        up = Problem(partial(phi, p, lmb, sgm))
        x_hist.append(x)
        x = steepest_descent(up, x0, tol=tol)

        # Concatenate costs
        c_prv = c
        c_e = p.get_eq_const(x)
        c_i = p.get_ineq_const(x)
        if c_e is not None and c_i is not None:
            c = np.concatenate((c_e, c_i), axis=0)
        elif c_e is not None:
            c = c_e
        elif c_i is not None:
            c = c_i

        # Make sure sigma is not too big
        if any(sgm >= sigma_max):
            break

        # Update sigma
        if np.linalg.norm(c, np.inf) > 0.25 * np.linalg.norm(c_prv, np.inf):
            for i in range(0, n_c):
                if np.abs(c[i]) > 0.25 * np.linalg.norm(c_prv, np.inf):
                    sgm[i] *= 10
            continue

        lmb = lmb - (sgm * c)

    return x if not hist else np.array(x_hist)

def lagrange_newton(p, x0, tol=1e-6, hist=False):
    """
    Constrained optimization algorithm using augmented Lagrange-Newton method
    :param p: Problem
        Constrained optimization problem to minimize
    :param x0: 2D numpy array
        Initial guess
    :param tol: float
        Tolerance of the algorithm. If norm of gradient is less than tol,
        iterations stop.
        Default = 1e-6
    :param hist: bool
        Flag to return history of x or not
        Default = False
    :return: 2D or 3D numpy array
        If hist = False, returns the last value of x.
        Else, returns an array with values of x after each iteration
    """
    x_hist = []

    n_e = p.num_eq_const()
    n_i = p.num_ineq_const()
    n_c = n_e + n_i

    def W(x, lmb):
        lmb_e = lmb[0:n_e, :]
        lmb_i = lmb[n_e:n_c, :]
        hess_f = get_fd_hessian(p.get_cost, x)
        hess_c_e = - np.sum([lmb_e[i] * get_fd_hessian(p.get_eq_const()[i], x)
            for i in range(0, n_e)])
        hess_c_i = - np.sum([lmb_i[i] * get_fd_hessian(p.get_ineq_const()[i], x)
            for i in range(0, n_i)])
        hess = hess_f + hess_c_e + hess_c_i
        return hess

    def A(x):
        grad_e = np.array([np.squeeze(get_fd_grad(p.get_eq_const()[i], x))
                for i in range (0, n_e)])
        grad_i = np.array([np.squeeze(get_fd_grad(p.get_ineq_const()[i], x))
                for i in range (0, n_i)])
        if n_e != 0 and n_i != 0:
            grad = np.concatenate((grad_e, grad_i), axis=0)
        elif n_e != 0:
            grad = grad_e
        elif n_i != 0:
            grad = grad_i
        return grad

    x = x0
    lmb = np.zeros((n_c, 1))

    # Concatenate costs
    c_e = p.get_eq_const(x)
    c_i = p.get_ineq_const(x)
    if c_e is not None and c_i is not None:
        c = np.concatenate((c_e, c_i), axis=0)
    elif c_e is not None:
        c = c_e
    elif c_i is not None:
        c = c_i

    delta_x = 1e12

    while delta_x  > tol:

        # Compute KKT matrix
        KKT = np.block([
            [W(x, lmb), -A(x).T],
            [-A(x), np.zeros((n_c, n_c))]
        ])

        # Compute gradient augmented with constraints
        if n_e != 0 and n_i != 0:
            f = np.block([
                [-get_fd_grad(p.get_cost, x).T + A(x).T @ lmb],
                [p.get_eq_const(x)],
                [p.get_ineq_const(x)]
            ])
        elif n_e != 0:
            f = np.block([
                [-get_fd_grad(p.get_cost, x).T + A(x).T @ lmb],
                [p.get_eq_const(x)]
            ])
        elif n_i != 0:
            f = np.block([
                [-get_fd_grad(p.get_cost, x).T + A(x).T @ lmb],
                [p.get_ineq_const(x)]
            ])

        x_prv = x
        # Invert KKT matrix to get x and lambda increments
        X = np.linalg.solve(KKT, f)
        dim = np.max(np.shape(x))
        x_hist.append(x)
        # Apply x and lambda increments
        x = x + X[:dim, :]
        lmb = lmb + X[dim:, :]

        c_e = p.get_eq_const(x)
        c_i = p.get_ineq_const(x)

        if c_e is not None and c_i is not None:
            c = np.concatenate((c_e, c_i), axis=0)
        elif c_e is not None:
            c = c_e
        elif c_i is not None:
            c = c_i

        # Check distance from previous x
        delta_x = np.linalg.norm(x - x_prv)

    return x if not hist else np.array(x_hist)