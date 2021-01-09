import numpy as np  # ver 1.19.0
# import scipy.stats as st  # ver 1.5.1
# import seaborn as sns  # ver 0.10.1
# from matplotlib import pyplot as plt  # ver 3.2.2
# from gamma_dist import f_beta_gamma
# import sys


def metropolis_hastings(x, f, nIter):
    """
    Metropolis algorithm to draw samples with known probability distribution.

    Parameters
    ----------
    x : ndarry with size n (type T)
        Initial state.
    f : callable, (T) -> float
        Function proportional to the desired probability distribution.
    nIter : int
        Size of sample set.

    Returns
    ------
    samples : ndarray with size nIter x n
         Samples with desired probability distribution.
    """
    
    # Draw samples
    samples = np.zeros((nIter, x.shape[0]))
    for idx in range(nIter):
        # Generate a candidate
        x_star = np.random.multivariate_normal(mean=x, cov=np.diag([1] * x.shape[0]))
        
        # Accept ratio
        r = f(x_star) / f(x)
        
        # Accept new state
        if np.random.uniform(0.0, 1.0) < r:
            x = x_star

        # "Append" new state
        samples[idx] = np.array(x)

    return samples


# if __name__ == "__main__":
#     from gamma_dist import gamma_dist_var
    
#     if len(sys.argv) > 4:
#         # Interpret input
#         lambda_beta, v_beta, lambda_gamma, v_gamma = map(float, sys.argv[1:])
#     else:
#         lambda_beta,v_beta = 0.27,2.47
#         lambda_gamma,v_gamma = 0.19,7.58    
#     # f is proportional to pi(beta, gamma)
#     f = lambda x: gamma_dist_var(x[0], lambda_beta, v_beta) \
#         * gamma_dist_var(x[1], lambda_gamma, v_gamma)
    

#     # Draw samples
#     samples = metropolis_hastings(np.array([lambda_beta / v_beta, lambda_gamma / v_gamma]), f, 100000)
#     beta_samples = samples[:, 0]
#     gamma_samples = samples[:, 1]
    
#     # Plot
#     h = sns.jointplot(samples[:, 0], samples[:, 1], kind='kde')
#     h.set_axis_labels(r'$\beta$', r'$\gamma$')
#     plt.suptitle('Lấy mẫu bằng Metropolis-Hastings')
    
#     # Output capture
#     plt.savefig('sampling.png')
    
#     # Compare means and variance
#     print('=' * 73)
    
#     print('beta')
#     print('actual : mean: {:.4f}\t variance: {:.4f}'\
#             .format(lambda_beta / v_beta, lambda_beta / v_beta**2))
#     print('samples: mean: {:.4f}\t variance: {:.4f}'\
#             .format(np.mean(beta_samples), np.var(beta_samples)))

#     print('gamma')
#     print('actual : mean: {:.4f}\t variance: {:.4f}'\
#             .format(lambda_gamma / v_gamma, lambda_gamma / v_gamma**2))
#     print('samples: mean: {:.4f}\t variance: {:.4f}'\
#             .format(np.mean(gamma_samples), np.var(gamma_samples)))
    
#     print('=' * 73)

#     # Show plot
#     plt.show()