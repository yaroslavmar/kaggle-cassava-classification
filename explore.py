import pandas as pd

train = pd.read_csv('./data/train.csv')

train.groupby('label').size().sort_values()

"""
0     1087
1     2189
2     2386
4     2577
3    13158
"""
