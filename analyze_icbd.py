import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
import pymc3 as pm
import matplotlib.pyplot as plt
import pdb


def compute_strategic_regret(X, y, ind, n_feature_subsets=5):
  x_test = X.iloc[0, :]

  # ToDo: Implement strategic model sharing

  # Fit ground truth (prob of player A winning)
  lm_true = RFECV(LogisticRegression())
  lm_true.fit(X, y)

  # Players only observe subset of features
  ind_1 = np.random.choice(ind, 5)
  ind_2 = np.random.choice(ind, 5)

  # Player A search
  best_features_1 = None
  best_prob_1 = 0.
  for _ in range(n_feature_subsets):
    features_1 = np.random.choice(ind_1, 3)
    X_1 = X.loc[:, features_1]
    lm_1 = LogisticRegression()
    lm_1.fit(X_1, y)
    p_1 = lm_1.predict_proba(np.array(x_test.loc[features_1]).reshape(1, -1))[0, 1]
    if p_1 > best_prob_1:
      best_prob_1 = p_1
      best_features_1 = features_1

  # Fit player B
  best_features_2 = None
  best_prob_2 = 1.
  for _ in range(n_feature_subsets):
    features_2 = np.random.choice(ind_2, 3)
    X_2 = X.loc[:, features_2]
    lm_2 = LogisticRegression()
    lm_2.fit(X_2, y)
    p_2 = lm_2.predict_proba(np.array(x_test.loc[features_2]).reshape(1, -1))[0, 1]
    if p_2 < best_prob_2:
      best_prob_2 = p_2
      best_features_2 = features_2

  # Fit combined models
  combined_features = np.union1d(best_features_1, best_features_2)
  X_combined = X.loc[:, combined_features]
  lm_combined = LogisticRegression()
  lm_combined.fit(X_combined, y)

  coop_1_strat_2 = np.union1d(ind_1, best_features_2)
  X_coop_1_strat_2 = X.loc[:, coop_1_strat_2]
  lm_coop_1_strat_2 = LogisticRegression()
  lm_coop_1_strat_2.fit(X_coop_1_strat_2, y)

  strat_1_coop_2 = np.union1d(ind_2, best_features_1)
  X_strat_1_coop_2 = X.loc[:, strat_1_coop_2]
  lm_strat_1_coop_2 = LogisticRegression()
  lm_strat_1_coop_2.fit(X_strat_1_coop_2, y)

  # # Fit full private models
  # X_1_private = X.loc[:, ind_1]
  # X_2_private = X.loc[:, ind_2]
  # lm_1_private = LogisticRegression()
  # lm_2_private = LogisticRegression()
  # lm_1_private.fit(X_1_private, y)
  # lm_2_private.fit(X_2_private, y)

  # Regrets from cooperative play
  x_test_combined = np.array(x_test.loc[combined_features]).reshape(1, -1)
  x_test_coop_1_strat_2 = np.array(x_test.loc[coop_1_strat_2]).reshape(1, -1)
  x_test_strat_1_coop_2 = np.array(x_test.loc[strat_1_coop_2]).reshape(1, -1)
  p_combined = lm_combined.predict_proba(x_test_combined)[0, 1]
  p_coop_1_strat_2 = lm_coop_1_strat_2.predict_proba(x_test_coop_1_strat_2)[0, 1]
  p_strat_1_coop_2 = lm_strat_1_coop_2.predict_proba(x_test_strat_1_coop_2)[0, 1]

  regret_1 = p_combined - p_coop_1_strat_2
  regret_2 = (1 - p_combined) - (1 - p_strat_1_coop_2)
  return regret_1


if __name__ == "__main__":
  tau = 0.05
  cost = -0.2
  n_feature_subsets = 5  # Number of feature subsets to search over in strategic model sharing

  df = pd.read_csv('icb2v13.csv')
  drop = ['icb1', 'crisno', 'crisname', ]

  # drop first 49 columns, except for outcom

  # Only keep sevvio in [4] (full-scale war)
  # Only keep outcom in [1, 4] (victory, defeat); this is dependent var
  # Only keep gravity in [1, 3, 4, 5, 6]
  # (limited military threat, territorial threat, threat to influence,
  #  threat of grave damage, threat to existence)

  # independent variables: cols 64-86
  # dependent var: 44

  ind = ['territ', 'allycap', 'issue', 'pethin', 'powdis', 'actloc', 'geog', 'cractloc']
  dep = ['outcom']
  keep = ind + dep
  df = df.loc[df.sevvio == 4]
  df = df.loc[df.outcom.isin([1., 4.])]
  df = df.loc[df.gravty.isin([1, 3, 4, 5, 6])]
  df = df.loc[:, keep]
  df.dropna(axis='rows', inplace=True)
  df.outcom = df.outcom.map({4.: 1, 1.: 0})

  X = df.loc[:, ind]
  y = df.loc[:, dep]

  mc_reps = 20
  regrets = []
  for _ in range(mc_reps):
    regret_1 = compute_strategic_regret(X, y, ind)
    regrets.append(regret_1)
  print(np.mean(regrets))
