import matplotlib.pyplot as pl
import pandas as pd

df = pd.read_csv('oqdm_formation.csv')

print(df)
fig, ax = pl.subplots()
ax.scatter(range(df.shape[0]), df['delta_e'], marker='.')
ax.set_xlabel('index')
ax.set_ylabel('delta_e')
pl.show()
