import pandas as pd
def mean_policy(arr):
    # return np.median(arr)
    return pd.Series(arr).ewm(span=5).mean().iloc[-1]