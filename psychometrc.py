import numpy as np
import pdb
from sklearn import decomposition
from scipy.stats import pearsonr


def generate_tc_cross_pd_matrix(social_dilemma_weight=1., tc_bonus=1., tc_byproduct=0.):
  tc_equal_1 = np.array([[0., 0.], [0., 1.]]) * social_dilemma_weight
  tc_equal_2 = np.array([[0., 0.], [0., 1.]]) * social_dilemma_weight

  tc_less_1 = tc_byproduct * np.ones((2, 2))
  tc_less_2 = tc_byproduct * np.ones((2, 2))

  tc_greater_1 = tc_bonus * np.ones((2, 2))
  tc_greater_2 = tc_bonus * np.ones((2, 2))

  u1 = np.column_stack((
    np.vstack((tc_equal_1 + tc_greater_1 + tc_byproduct, tc_less_1)),
    np.vstack((tc_greater_1, tc_equal_1))
  ))

  u2 = np.column_stack((
    np.vstack((tc_equal_2 + tc_greater_2 + tc_byproduct, tc_greater_2)),
    np.vstack((tc_less_2, tc_equal_2))
  ))

  return u1, u2


def get_type1_policy_payoffs(u1, u2, pi1, pi2):
  # Policies parameterized by (prob high task competence, prob cooperate)
  phigh1, pcoop1 = pi1
  phigh2, pcoop2 = pi2
  plow1 = 1 - phigh1
  plow2 = 1 - phigh2
  pdefect1 = 1 - pcoop1
  pdefect2 = 1 - pcoop2

  s1 = np.array([phigh1 * pdefect1, phigh1 * pcoop1, plow1 * pdefect1, plow1 * pcoop1])
  s2 = np.array([phigh2 * pdefect2, phigh2 * pcoop2, plow2 * pdefect2, plow2 * pcoop2])

  ev1 = np.dot(s1.T, np.dot(u1, s2))
  ev2 = np.dot(s2.T, np.dot(u2, s1))

  return ev1, ev2


if __name__ == "__main__":
  np.random.seed(3)

  # Generate policies
  pi1_list = np.random.beta(0.2, 0.2, size=(20, 2))
  pi2_list = np.random.beta(0.2, 0.2, size=(20, 2))

  # Two clusters: small tc and large cc component; large tc and small cc component
  num_envs_to_generate = 6
  u1_list = []
  u2_list = []

  for _ in range(num_envs_to_generate // 2):
    social_dilemma_weight = np.random.gamma(shape=1, scale=1)
    tc_bonus = np.random.gamma(shape=0.01, scale=1)
    tc_byproduct = np.random.gamma(shape=0.01, scale=1)
    u1, u2 = generate_tc_cross_pd_matrix(social_dilemma_weight=social_dilemma_weight, tc_bonus=tc_bonus,
                                         tc_byproduct=tc_byproduct)
    u1_list.append(u1)
    u2_list.append(u2)

  for _ in range(num_envs_to_generate // 2):
    social_dilemma_weight = np.random.gamma(shape=0.01, scale=1)
    tc_bonus = np.random.gamma(shape=1, scale=1)
    tc_byproduct = np.random.gamma(shape=1, scale=1)
    u1, u2 = generate_tc_cross_pd_matrix(social_dilemma_weight=social_dilemma_weight, tc_bonus=tc_bonus,
                                         tc_byproduct=tc_byproduct)
    u1_list.append(u1)
    u2_list.append(u2)

  # Construct feature matrix
  X = np.zeros((0, 2*num_envs_to_generate))
  phigh1 = np.array([])
  phigh2 = np.array([])
  pcoop1 = np.array([])
  pcoop2 = np.array([])
  for pi1 in pi1_list:
    for pi2 in pi2_list:
      x = np.zeros(0)
      for u1, u2 in zip(u1_list, u2_list):
        ev1, ev2 = get_type1_policy_payoffs(u1, u2, pi1, pi2)
        x = np.hstack((x, [ev1, ev2]))
      X = np.vstack((X, x))
      phigh1 = np.append(phigh1, pi1[0])
      phigh2 = np.append(phigh2, pi2[0])
      pcoop1 = np.append(pcoop1, pi1[1])
      pcoop2 = np.append(pcoop2, pi2[1])

  pca = decomposition.PCA(n_components=4)
  pca.fit(X)
  Y = pca.transform(X)


