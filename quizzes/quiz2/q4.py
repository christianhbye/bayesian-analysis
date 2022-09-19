import numpy as np

"""
This is essentially asking to compute the mean and variance of the 
Gaussian distribution that is the product of two Gaussians. This is obviously
done analytically. We compute it for our case.
"""

def sig_prod(sig1, sig2):
    num = sig1**2 * sig2**2
    den = sig1**2 + sig2**2
    return np.sqrt(num/den)

def mean_prod(mean1, mean2, sig1, sig2):
    num = mean1 * sig2**2 + mean2 * sig1 ** 2
    den = sig1 ** 2 + sig2 ** 2
    return num/den

if __name__=="__main__":
    # prior:
    s1 = 40
    m1 = 180

    # likelihood
    """
    This actually has unknown mean but evaluated at a specific value
    since the Gaussian is symmetric in x and mu, we can pretend the mean
    is known
    """
    n = 10
    s2 = 20 / np.sqrt(n)  # standard deviation
    m2 = 150  # sample mean, pretended to be mean of likelihood

    print(f"The mean is {mean_prod(m1, m2, s1, s2)}.")
    print(f"The standard deviation is {sig_prod(s1, s2)}.")

