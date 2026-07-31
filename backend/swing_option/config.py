import numpy as np
import pandas as pd

# model parameters
T = 1
N = 252
sigma = 12.750
r = 0.04
kappa = 29.223
S_0 = 3.21

# alpha(t) under Q (pricing)
a0 = 3.485041
b_trend = -0.169307
a1 = -0.445194
b1 = 0.087682
a2 = 0.395931
b2 = 0.011217

def alpha(t):  # seasonal mean reversion
    return (
        a0 + b_trend * t
        + a1 * np.cos(2 * np.pi * t) + b1 * np.sin(2 * np.pi * t)
        + a2 * np.cos(4 * np.pi * t) + b2 * np.sin(4 * np.pi * t)
    )

# Option contract parameters
K = a0 + (b_trend / 2)  # strike price is the mean of alpha(t)
C_max = 1260
C_min = 1050
q_max = 10