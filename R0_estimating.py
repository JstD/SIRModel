import numpy as np
import math
from regionize_data import regionize
from sampling import metropolis_hastings
from gamma_dist import gamma_dist_exp, f_beta_gamma

# KSTN shitty addition
import hyperopt
from hyperopt import fmin, hp, tpe


def estimate_R0_basic():
    """
    Approximate R0

    Returns
    -------
    R0 : float
        Approximated R0.

    """
    
    s, i, r = regionize()
    x = np.array([x + y for x, y in zip(i, r) if x + y != 0])
    
    for _ in range(20):
        samples = metropolis_hastings(np.array([.1, .025]),
                                    lambda x: f_beta_gamma(x[0], x[1]),
                                    10000)
    
        beta = np.array(samples[:, 0])
        gamma = np.array(samples[:, 1])
    
        denom = np.array([sum([gamma_dist_exp(val, b, c) for val in x]) for b, c in zip(beta, gamma)])
        numer = denom + np.array([math.log(b / c) for b, c in zip(beta, gamma)])
    
        max_numer = np.amax(numer)
        max_denom = np.amax(denom)
    
        numer = numer[numer > max_numer - 20]
        denom = denom[denom > max_denom - 20]
    
        numer -= np.array([max_numer - 10] * numer.shape[0])
        denom -= np.array([max_denom - 10] * denom.shape[0])
    
        numer = sum([math.exp(x) for x in numer])
        denom = sum([math.exp(x) for x in denom])
    
        R0 = numer / denom * math.exp(max_numer - max_denom)
        return R0
        

def Ed(beta, gamma, mu, s_c, i_c, r_c, d_c, coef):
    from euler import dSIRD, rk4_step
    
    s0, i0, r0, d0 = s_c[0], i_c[0], r_c[0], d_c[0]
    
    sird = [(s0, i0, r0, d0)]
    for b, c, m in zip(beta[:-1], gamma[:-1], mu[:-1]):
        dsird = lambda x, t, dt: dSIRD(s=x[0],
                                       i=x[1],
                                       r=x[2],
                                       d=x[3],
                                       beta=b,
                                       gamma=c,
                                       mu=m,
                                       dt=dt)
        
        s0, i0, r0, d0 = rk4_step((s0, i0, r0, d0), 0, dsird, 1)
        sird.append((s0, i0, r0, d0))
        
    sird = np.array(sird)
    
    i = sird[:, 1]
    d = sird[:, 3]
    
    log = lambda x: math.log(x) if x > 0 else np.NINF
    
    return coef[0] * sum([(log(a0)-log(a1) if a0 != a1 else 0)**2\
                          + (log(b0)-log(b1) if b0 != b1 else 0)**2
                          for a0, a1, b0, b1 in zip(i, i_c, d, d_c)]) \
        + coef[1] * np.sum((i-i_c)**2 + (d-d_c)**2)

        
def Er(beta, gamma, mu, coef):
    return coef[0] * sum([(beta[i+1] - beta[i])**2 \
                            + (gamma[i+1] - gamma[i])**2 \
                                + 100*(mu[i+1] - mu[i])**2 
                                for i in range(len(beta)-1)]) 


def E0():
    return 0


def L(beta, gamma, mu, s_c, i_c, r_c, d_c):
    max_i_c = np.amax(i_c)
    log_max_i_c = math.log(max_i_c)
    max_alpha = max([beta[0], gamma[0], mu[0]])
    
    return Ed(beta, gamma, mu, s_c, i_c, r_c, d_c, coef = (1, 0.01*log_max_i_c/max_i_c)) \
        + Er(beta, gamma, mu, coef = (100*log_max_i_c/max_alpha,)) \
        + E0()
        
    
def obj_f(params):
    beta = params['beta']
    gamma = params['gamma']
    mu = params['mu']
    
    return L(beta, gamma, mu, obj_f.s_c, obj_f.i_c, obj_f.r_c, obj_f.d_c)


def estimate_R0_ML():
    s, i, r, d = regionize(model='SIRD')
    s, i, r, d = s[60:90], i[60:90], r[60:90], d[60:90]
    
    space = {
        'beta': [hp.uniform('beta', 0.0, 10.0)] * i.shape[0],
        'gamma': [hp.uniform('gamma', 0, 1)] * i.shape[0],
        'mu': [hp.uniform('mu', 0, 0.1)] * i.shape[0],
    }
    
    obj_f.s_c = s
    obj_f.i_c = i
    obj_f.r_c = r
    obj_f.d_c = d
    
    return fmin(fn=obj_f, space=space, algo=tpe.rand.suggest, max_evals=1000)
    

if __name__ == '__main__':
    print(estimate_R0_basic())