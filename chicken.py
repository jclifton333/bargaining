import numpy as np
import nashpy as nash
from nash_unif import get_nash_welfare_optimal_eq, get_welfare_optimal_eq
import copy


CRASH = -10
G_1 = np.array([[0, -1], [2, CRASH]])
G_2 = np.array([[0, 2], [-1, CRASH]])


def draw_chicken_game(sd=0.5):
    G_1_draw = copy.copy(G_1)
    G_2_draw = copy.copy(G_2)
    G_1_draw[1, 0] += np.random.normal(scale=sd)
    G_2_draw[1, 0] += np.random.normal(scale=sd)

    return G_1_draw, G_2_draw


def combine_reports(G_1_rep, G_2_rep, sd=0.5):
    diff = np.abs(G_1_rep[0, 1] - G_2_rep[0, 1])
    if diff < 2*sd:
       return True, (G_1_rep + G_2_rep) / 2
    else:
       return False, None


def evaluate_reporting_policy_profile(distort_1, distort_2, sd=0.5, nrep=100):
    v1_mean = 0.
    v2_mean = 0.
    v1_default_mean = 0.
    v2_default_mean = 0.
    true_game = nash.Game(G_1, G_2)
    for rep in range(nrep):
        G_1_private, G_2_private = draw_chicken_game(sd=sd)
        G_1_rep = copy.copy(G_1_private)
        G_2_rep = copy.copy(G_2_private)
        G_1_rep[0, 1] += distort_1
        G_2_rep[0, 1] += distort_2
        combine, combined_game = combine_reports(G_1_rep, G_2_rep)

        a1_default, _, _ = get_welfare_optimal_eq(nash.Game(G_1_rep, G_2))
        _, a2_default, _ = get_welfare_optimal_eq(nash.Game(G_2_rep, G_2))
        v1_default, v2_default = true_game[(a1_default, a2_default)]
        v1_default_mean += v1_default / nrep
        v2_default_mean += v2_default / nrep

        if combine:
            a1, a2, _ = get_welfare_optimal_eq(nash.Game(combined_game, G_2))
        else:
            a1, a2 = a1_default, a2_default
        v1, v2 = true_game[(a1, a2)]
        v1_mean += v1 / nrep
        v2_mean += v2 / nrep
    return v1_mean, v2_mean, v1_default_mean, v2_default_mean


if __name__ == "__main__":
    v1_mean, v2_mean, v1_default_mean, v2_default_mean = evaluate_reporting_policy_profile(0, 0)
    print(f'{v1_mean}, {v2_mean}\n{v1_default_mean}, {v2_default_mean}')

