import numpy as np
import pystan as ps


def big_model(actions, splits):
  code = """
    data {
      int<lower=0> N;
      int<lower=0, upper=1> a[N];
      vector[N] s;
    }
    parameters {
      real<lower=0> u;
    }
    model {
      vector[N] p;
      u ~ gamma(1, 1);
      for (i in 1:N) {
        p[i] = exp(u*s[i]) / (1 + exp(u*s[i]));
        a[i] ~ bernoulli(p[i]);
      }
    }
  """
  data = {'a': actions, 's': splits, 'N': len(splits)}
  sm= ps.StanModel(model_code=code)
  fit = sm.sampling(data=data, iter=1000, chains=2)
  return fit


if __name__ == "__main__":
  N = 100
  splits = np.random.uniform(size=N)
  actions = np.random.binomial(1, p=0.5, size=N)
  fit = big_model(actions, splits)
