import numpy as np
import nashpy as nash


# ToDo: what to do immediately after defection?
def util_policy(a_prev, p_defect):
  # Player 1
  a1_prev, a2_prev = a_prev
  if a1_prev == 0 and a2_prev == 0:
    return 0, 0
  elif a1_prev == 0 and a2_prev == 1:
    if np.random.random() < p_defect:
      return 2, 2
    else:
      return 0, 0
  elif a1_prev == 2 or a2_prev == 2:
    return 0, 0
  else:
    return 2, 2


def egal_policy(a_prev, p_defect):
  # Player 2
  a1_prev, a2_prev = a_prev
  if a1_prev == 0 and a2_prev == 0:
    return 1, 1
  elif a1_prev == 1 and a2_prev == 1:
    return 0, 0
  elif a1_prev == 0 and a2_prev == 1:
    if np.random.random() < p_defect:
      return 2, 2
    else:
      return 1, 1
  elif a1_prev == 2 or a2_prev == 2:
    return 1, 1
  else:
    return 2, 2





if __name__ == "__main__":
  # Payoffs for conditional deployment game in asymmetric BOTSPD
  # Using disagreement point = 1.1, 0.9 (from preliminary pop-based training experiments)
  # Players submit one of {{egal}, {util}, {egal, util}} and randomize between the intersection

  # p2_defect solves p*-1 + (1-p)*3.5 = 2 => -p(1 + 3.5) = -1.5 => p = 1.5 / 4.5
  # p1_defect solves p*-1 + (1-p)*2 = 1 => -p(1 + 2) = -1 => p = 1 / 3

  # p1_mix = 0.5*2. + 0.5*3.5
  # p2_mix = 0.5*2. + 0.5*1.
  # v1 = np.array([[2, 1.1, 2],
  #       [1.1, 3.5, 3.5],
  #       [2, 3.5, p1_mix]])
  # v2 = np.array([[2, 0.9, 2],
  #               [0.9, 1, 1],
  #               [2, 1, p2_mix]])
  # game = nash.Game(v1, v2)
  # eqs = list(game.support_enumeration())
  payout_mat_1 = np.array([[3.5, 0, -3], [0, 1.0, -3],
                               [2, 2, -1]])
  payout_mat_2 = np.array([[1.0, 0.0, 2], [0, 3, 2],
                               [-3, -3, -1]])

  n_rep = 100
  a_prev = 0, 0
  payoffs_1 = np.zeros(n_rep)
  payoffs_2 = np.zeros(n_rep)
  for i in range(n_rep):
    a1_1, a1_2 = util_policy(a_prev, 1/3)
    a2_1, a2_2 = egal_policy(a_prev, 1.5/4.5)
    a_prev = a1_1, a2_2
    payoffs_1[i] = payout_mat_1[a_prev]
    payoffs_2[i] = payout_mat_2[a_prev]
  mean_1 = payoffs_1.mean()
  mean_2 = payoffs_2.mean()





