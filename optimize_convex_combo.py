import numpy as np
from scipy.optimize import minimize
import nashpy as nash


def nash_welfare(u1, u2, d1=0, d2=0):
    return np.log(u1 - d1) + np.log(u2 - d2)


def egalitarian_welfare(u1, u2, d1=0, d2=0):
    return np.min((u1 - d1, u2 - d2))


def convex_combo(alpha, u11, u12, u21, u22):
    u1 = alpha * u11 + (1 - alpha) * u12
    u2 = alpha * u21 + (1 - alpha) * u22
    return u1, u2


def optimize_welfare(welfare, u11, u12, u21, u22, d1=0, d2=0):
    # optimize welfare functions along segment [(u11, u21), (u12, u22)]

    def objective(alpha):
        u1, u2 = convex_combo(alpha, u11, u12, u21, u22)
        w = welfare(u1, u2, d1=d1, d2=d2)
        return w

    alphas = np.linspace(0, 1, 100)
    welfares = np.array([objective(a) for a in alphas])
    best_alpha = alphas[np.argmax(welfares)]
    best_payoffs = convex_combo(best_alpha, u11, u12, u21, u22)
    return best_payoffs


if __name__ == "__main__":
    # u11, u21 = 3.5, 1
    # u12, u22 = 1, 3
    # nash = optimize_welfare(nash_welfare, u11, u12, u21, u22)
    # egalitarian = optimize_welfare(egalitarian_welfare, u11, u12, u21, u22)
    # print(f'nash: {nash} egal: {egalitarian}')

    U1 = np.array([[3.5, 1., 3.5], [1., 1.5, 1.5], [3.5, 1.5, 2.5]])
    U2 = np.array([[1., 0.3, 1.], [0.3, 1.4, 1.4], [1., 1.4, 1.2]])
    G = nash.Game(U1, U2)
    eqs = list(G.support_enumeration())
    for eq in eqs:
        print(eq, G[eq])






