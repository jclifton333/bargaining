import numpy as np
import pandas as pd


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

  keep = ['outcom', 'territ', 'allycap', 'issue', 'pethin', 'powdis',
          'actloc', 'geog', 'cractloc']
  df = df.loc[df.sevvio == 4]
  df = df.loc[df.outcom in [1, 4]]
  df = df.loc[df.gravity in [1, 3, 4, 5, 6]]
  df.outcome = (df.outcome == 1)
