import matplotlib.pyplot as plt
import numpy as np
from scipy.special import comb  # n choose k
import time


def binomial(nb, fb, N):
    """
    Binomial distribution.

    Parameters
    ----------
    nb : int
        Number of black balls that are drawn.
    fb : float
        The fraction of balls in the urn that are black.
    N : int
       Number of draws.

    Returns
    -------
    p : float
        The probability of drawing nb black balls from the given urn.

    """
    return comb(N, nb) * fb**nb * (1 - fb) ** (N - nb)


def flat_prior(N_urns):
    return np.full(N_urns, 1 / N_urns)


if __name__ == "__main__":
    N_urns = 7  # number of urns
    nb_urns = np.arange(N_urns)  # list of black balls per urn
    balls_per_urn = 6
    fb = nb_urns / balls_per_urn
    prior1 = flat_prior(N_urns)

    plt.ion()
    plt.figure()
    plt.scatter(nb_urns, prior1, label="Prior")
    plt.legend()

    # first round: 3/6 balls drawn are black
    ll1 = binomial(3, fb, 6)  # likelihood for each urn
    post1 = prior1 * ll1
    post1 /= post1.sum()  # normalize by evidence

    plt.scatter(nb_urns, post1, label="First draw")
    plt.legend()

    # second round: use first round posterior as prior, 5/6 balls are black
    prior2 = post1
    ll2 = binomial(5, fb, 6)
    post2 = prior2 * ll2
    post2 /= post2.sum()

    plt.scatter(nb_urns, post2, label="Second draw")
    plt.legend()
    plt.show(block=True)
