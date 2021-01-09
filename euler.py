import numpy as np


def dSIR(s, i, r, beta, gamma, dt):
    """Define dS(t), dI(t) and dR(t) in SIR-epidemic model (SIR for short)."""
    
    N = s + i + r
    return np.array([-beta / N * i * s,
                     beta / N * i * s - gamma * i,
                     gamma * i]) * dt


def dSIRD(s, i, r, d, beta, gamma, mu, dt):
    """
    Define dS(t), dI(t), dR(t) and dD(t) in SIR-epidemic model
    with vital dynamics and constant population (SIRD for short).
    """
    
    N = s + i + r + d
    return np.array([-beta / N * i * s,
                     beta / N  * i * s - (gamma+mu) * i,
                     gamma * i,
                     mu * i]) * dt


def euler_step(x0, t0, dx, dt):
    """
    Approximate x(1) by Euler method.

    Parameters
    ----------
    x0 : tuple of type T
        Initial value.
    t0 : float
        Initial time.
    dx : callable, (T, float, float) -> T
        Differential (system of) equation(s).
    dt : float
        Time step.

    Returns
    -------
    x1 : T
        Estimated x(1).

    """
    
    k1 = dx(x0, t0, dt)
    
    x1 = x0 + k1
    
    return x1


def improved_euler_step(x0, t0, dx, dt):
    """
    Approximate x(1) by improved Euler method.

    Parameters
    ----------
    x0 : tuple of type T
        Initial value.
    t0 : float
        Initial time.
    dx : callable, (T, float, float) -> T
        Differential (system of) equation(s).
    dt : float
        Time step.

    Returns
    -------
    x1 : T
        Estimated x(1).

    """
    
    k1 = dx(x0, t0, dt)
    k2 = dx(x0 + k1, t0 + dt, dt)
    
    x1 = x0 + (k1+k2) / 2
    
    return x1


def rk4_step(x0, t0, dx, dt):
    """
    Approximate x(1) by fourth-order Runge–Kutta method.

    Parameters
    ----------
    x0 : tuple of type T
        Initial value.
    t0 : float
        Initial time.
    dx : callable, (T, float, float) -> T
        Differential (system of) equation(s).
    dt : float
        Time step.

    Returns
    -------
    x1 : T
        Estimated x(1).

    """
    
    k1 = dx(x0, t0, dt)
    k2 = dx(x0 + k1/2, t0 + dt/2, dt)
    k3 = dx(x0 + k2/2, t0 + dt/2, dt)
    k4 = dx(x0 + k3, t0 + dt, dt)
    
    x1 = x0 + (k1 + 2*k2 + 2*k3 + k4) / 6
    
    return x1


def iterative_method(x0, dx, t, method='Euler'):
    """
    Approximate x in time range t by iterative method.
    
    Parameters
    ----------
    x0 : tuple of type T
        Initial value.
    dx : callable, (T, float, float) -> T
        Differential (system of) equation(s).
    t : array of float with size n
        Concerned points in time.
    method : string
        Iterative method to use.
        Can recognize 'Euler', 'Improved Euler' and 'Rk4'
        The default is 'Euler'.
        
    Returns
    ------
    x : ndarray of T with size n
        Estimated x values at all concerned points in time.
    
    """
    
    method = method.title()
    
    if method == 'Euler':
        step_func = lambda x, t0, t1: euler_step(x, t0, dx, t1-t0)
    elif method == 'Improved Euler':
        step_func = lambda x, t0, t1: improved_euler_step(x, t0, dx, t1-t0)
    elif method == 'Rk4':
        step_func = lambda x, t0, t1: rk4_step(x, t0, dx, t1-t0)
    else:
        step_func = lambda x, t0, t1: euler_step(x, t0, dx, t1-t0)
    
    x = [x0]
    for t0, t1 in zip(t[:-1], t[1:]):
        x0 = step_func(x0, t0, t1)
        x.append(x0)

    return np.array(x)