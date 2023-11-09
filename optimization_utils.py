import numpy as np

def get_step_size(p, x, s, gamma=1.5, mu=0.8):
    """
    Determines the step size using Armijo algorithm
    :param p: Problem
        The problem to be optimized
    :param x: numpy 2D array
        The point we are at
    :param s: numpy 2D array
        Search direction
    :param gamma: float
        Parameter for increasing step size
    :param mu: float
        Parameter for decreasing step size
    :return: float
        Step size
    """
    w = 1  # Default step size

    k_g = 0  # Power of gamma
    k_m = 0  # Power of mu

    # Precompute cost and gradient to save time
    vx = p.get_cost(x)
    gx_s = p.get_grad(x) @ s

    def v_bar(w):
        return vx + 0.5 * w * gx_s

    while p.get_cost(x + gamma ** k_g * s) < v_bar(gamma ** k_g):
        k_g += 1
        w = gamma ** k_g

    while p.get_cost(x + mu ** k_m * gamma ** k_g * s) > v_bar(mu ** k_m * gamma ** k_g):
        k_m += 1
        w = mu ** k_m * gamma ** k_g

    return w

def check_cone_condition(p, x, s, theta=89):
    """
    Checks the cone condition at a point
    :param p: Problem
        The problem to be optimized
    :param x: numpy 2D array
        The point we are at
    :param s: numpy 2D array
        Search direction
    :param theta: float
        Acceptable angle with gradient in degrees
    :return: bool
        True if s within theta degrees of gradient at x
    """
    gx = p.get_grad(x)  # get gradient at x
    cos_phi = (-gx @ s) / (np.linalg.norm(s) * np.linalg.norm(gx))
    cos_theta = np.cos(theta * 2 * np.pi / 360)

    return (cos_phi > cos_theta)

def get_cost_norm(p, x):
    """
    Evaluates the cost at d
    :param p: Problem
        The problem to be optimized
    :param x: numpy 2D array
        The point at which we want to get the cost
    :return: float
        The cost at x
    """
    cost = 0
    if p.get_eq_const() is not None:
        cost = cost + np.linalg.norm(p.get_eq_const(x))**2
    if p.get_ineq_const() is not None:
        ineq_x = p.get_ineq_const(x)
        c = np.minimum(np.zeros(np.shape(ineq_x)), ineq_x)
        cost = cost + np.linalg.norm(c)**2
    return np.sqrt(cost)

def get_fd_hessian(f, x, h=1e-8):
    """
    The finite difference approximation of the Hessian at x
    :param f: function of x
        Function whose Hessian we want to evaluate
    :param x: numpy 2D array
        Point at whihc we want to evaluate the Hessian
    :param h: float
        Step size
    :return: 2D numpy matrix
        Hessian at x
    """
    dim = np.max(np.shape(x))
    I = np.eye(dim)
    H = np.zeros((dim, dim))

    for i in range(0, dim):
        for j in range(0, dim):
            H[i, j] = (f(x + h * I[:, [i]] + h * I[:, [j]]) \
                       - f(x + h * I[:, [i]]) - f(x + h * I[:, [j]]) \
                       + f(x)) / h ** 2

    return 0.5 * (H + H.T)

def get_fd_grad(f, x, h=1e-8):
    """
    Finite difference approximation of gradient at x
    :param f: function of x
        function whose gradient we want to evaluate
    :param x: numpy 2D array
        point at which we want to approximate the gradient
    :param h: float
        step size
    :return: numpy 2D array
        gradient at x 
    """
    dim = np.max(np.shape(x))
    grad_gen = ((f(x + h * np.eye(dim)[:, [i]]) - f(x)) / h
                for i in range(0, dim))
    grad = np.expand_dims(np.fromiter(grad_gen, np.float64), axis=0)
    return grad