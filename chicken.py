import numpy as np
import nashpy as nash
from nash_unif import get_nash_welfare_optimal_eq, get_welfare_optimal_eq
import copy
import pdb


CRASH = -5.
G_1 = np.array([[0., -1.], [2., CRASH]])
G_2 = np.array([[0., 2.], [-1., CRASH]])


def draw_chicken_game(sd=0.5):
    G_1_draw = copy.copy(G_1)
    G_2_draw = copy.copy(G_2)
    perturbation_1 = np.array([[0., 0.], [np.random.normal(scale=sd), 0]])
    perturbation_2 = np.array([[0., 0.], [np.random.normal(scale=sd), 0]])
    G_1_draw += perturbation_1
    G_2_draw += perturbation_2
    return G_1_draw, G_2_draw


def combine_reports(G_1_rep, G_2_rep, sd=0.5):
    diff = np.abs(G_1_rep[1, 0] - G_2_rep[1, 0])
    if diff < 8*sd:
        return True, (G_1_rep + G_2_rep) / 2
    else:
        return False, None


def evaluate_reporting_policy_profile(distort_1, distort_2, sd=0.5, nrep=100):
    v1_mean = 0.
    v2_mean = 0.
    v1_default_mean = 0.
    v2_default_mean = 0.
    true_game = nash.Game(G_1, G_2)
    prop_agree = 0.
    v1_list = []
    v2_list = []
    for rep in range(nrep):
        G_1_private, G_2_private = draw_chicken_game(sd=sd)
        G_1_rep = copy.copy(G_1_private)
        G_2_rep = copy.copy(G_2_private)
        G_1_rep[1, 0] += distort_1
        G_2_rep[1, 0] += distort_2
        combine, combined_game = combine_reports(G_1_rep, G_2_rep, sd=sd)
        prop_agree += combine / nrep

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
        v1_list.append(v1)
        v2_list.append(v2)
    v1_se = np.std(v1_list) / nrep
    v2_se = np.std(v2_list) / nrep
    return v1_mean, v2_mean, v1_default_mean, v2_default_mean, prop_agree, v1_se, v2_se


if __name__ == "__main__":
    sd = 0.5
    payoffs_1 = np.zeros((4, 4))
    payoffs_2 = np.zeros((4, 4))
    se_1 = np.zeros((4, 4))
    se_2 = np.zeros((4, 4))
    for i, distort_1 in enumerate(np.linspace(0, sd/2, 4)):
        for j, distort_2 in enumerate(np.linspace(0, sd / 4, 4)):
            v1, v2, v1_default, v2_default, prop_agree, v1_se, v2_se = \
                evaluate_reporting_policy_profile(distort_1, -distort_2, nrep=1000)
            payoffs_1[i, j] = v1
            payoffs_2[i, j] = v2
            se_1[i, j] = v1_se
            se_2[i, j] = v2_se
            if i == 0 and j == 0:
               print(f'default: {v1_default}, {v2_default}')
    g = nash.Game(payoffs_1, payoffs_2)
    a1, a2, _ = get_welfare_optimal_eq(g)
    print(payoffs_1)
    print(payoffs_2)
    print(a1, a2)
    print(g[(a1, a2)])
    final_se_1 = np.sqrt(np.dot(a2, np.dot(se_1 ** 2, a1)))
    final_se_2 = np.sqrt(np.dot(a1, np.dot(se_2 ** 2, a2)))
    print(final_se_1)
    print(final_se_2)
