from scipy import special
import math


def gamma_dist_exp(x, alpha, beta):
    return alpha * math.log(beta) - special.loggamma(alpha) \
            + (alpha - 1) * math.log(x) - beta * x if x > 0 else -math.inf


def gamma_dist_var(x, alpha, beta):
    return math.exp((alpha - 1) * math.log(x) - beta * x) if x > 0 else 0


def gamma_dist(x, alpha, beta):
    return math.exp(gamma_dist_exp(x, alpha, beta))


def f_beta_gamma(beta,
                  gamma,
                  lambda_beta=0.27,
                  v_beta=2.47,
                  lambda_gamma=0.19,
                  v_gamma=7.58):
    return gamma_dist_var(beta, lambda_beta, v_beta) \
            * gamma_dist_var(gamma, lambda_gamma, v_gamma)


def pi_beta_gamma(beta,
                  gamma,
                  lambda_beta=0.27,
                  v_beta=2.47,
                  lambda_gamma=0.19,
                  v_gamma=7.58):
    return gamma_dist(beta, lambda_beta, v_beta) \
        * gamma_dist(gamma, lambda_gamma, v_gamma)
    

if __name__ == "__main__":
    mode_beta = 0.0085
    mode_gamma = 0.0176
    
    mean_beta = 0.27 / 2.47
    mean_gamma = 0.19 / 7.58
    
    print(pi_beta_gamma(0.0212, 3.39))
    print(pi_beta_gamma(mode_beta, mode_gamma))
    print(pi_beta_gamma(mean_beta, mean_gamma))