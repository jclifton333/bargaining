import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB

if __name__ == "__main__":
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
  df = df.loc[df.outcom.isin([1, 4])]
  df = df.loc[df.gravty.isin([1, 3, 4, 5, 6])]
  df.outcome = (df.outcom == 1)
  df = df.loc[:, keep]
  df.dropna(axis='rows', inplace=True)

  X = df.loc[:, ind]
  y = df.loc[:, dep]

  lm = LogisticRegression()
  nb = BernoulliNB()
  lm.fit(X, y)
  nb.fit(X, y)

  phat_lm = lm.predict_proba(X)[:, -1]
  phat_nb = nb.predict_proba(X)[:, -1]




