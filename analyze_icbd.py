import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
import pymc3 as pm
import matplotlib.pyplot as plt

if __name__ == "__main__":
  tau = 0.05
  cost = -0.2

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

  X = df.loc[:, ind]
  y = df.loc[:, dep]

  # Fit ground truth (prob of player A winning)
  lm_true = RFECV(LogisticRegression())
  lm_true.fit(X, y)

  # Fit player A
  features_1 = np.random.choice(ind, 4)
  X_1 = X.loc[:, features_1]
  lm_1 = LogisticRegression()
  lm_1.fit(X_1, y)

  # Fit player B
  features_2 = np.random.choice(ind, 4)
  X_2 = X.loc[:, features_2]
  lm_2 = LogisticRegression()
  lm_2.fit(X_2, y)

  p_true = lm_true.predict_proba(X)[:, 1]
  p_1 = lm_1.predict_proba(X_1)[:, 1]
  p_2 = lm_2.predict_proba(X_2)[:, 1]
  p_coop = (p_1 + p_2) / 2

  compromise_ev_1 = p_coop
  compromise_ev_2 = 1 - p_coop

  war_ev_1 = p_true + cost
  war_ev_2 = 1 - p_true + cost

  war_regret_1 = compromise_ev_1 - war_ev_1
  war_regret_2 = compromise_ev_2 - war_ev_2
  print(war_regret_1, war_regret_2)
