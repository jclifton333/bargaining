import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
import pymc3 as pm
import matplotlib.pyplot as plt

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
  df.loc[df.outcom == 4, 'outcom'] = 1
  df.loc[df.outcom == 1, 'outcom'] = 0
  df = df.loc[:, keep]
  df.dropna(axis='rows', inplace=True)

  X = df.loc[:, ind]
  y = df.loc[:, dep]

  priors_1 = {'powdis': pm.Normal.dist(mu=0., sigma=1.)}
  priors_2 = {'powdis': pm.Normal.dist(mu=0., sigma=10.)}

  # ToDo: posterior probs of winning
  with pm.Model() as model:
    pm.glm.GLM.from_formula('outcom ~ powdis', df, family=pm.glm.families.Binomial(),
                            priors=priors_1)
    fitted = pm.fit(method='fullrank_advi')
    trace_big_1 = fitted.sample(2000)

  with pm.Model() as model:
    pm.glm.GLM.from_formula('outcom ~ powdis', df, family=pm.glm.families.Binomial(),
                            priors=priors_2)
    fitted = pm.fit(method='fullrank_advi')
    trace_big_2 = fitted.sample(2000)




