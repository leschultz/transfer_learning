from pymatgen.core import Composition
import pandas as pd

def name_convention(x):
    
    x = Composition(x)
    x = x.alphabetical_formula
    x = str(x)
    x = x.replace(' ', '')

    return x

df = 'oqmd.csv'
df = pd.read_csv(df)

df = df.drop_duplicates()
df = df.dropna()

df['name'] = df['name'].apply(name_convention)
df = df.sort_values(by=['stability', 'delta_e', 'name', 'entry_id'])

# Keep most stable compositions
df = df.drop_duplicates(subset='name', keep='first')

stability = df[['name', 'stability']]
formation = df[['name', 'delta_e']]

# Sort to make pretty
stability = stability.sort_values(by=['stability', 'name'])
formation = formation.sort_values(by=['delta_e', 'name'])

'''
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.scatter(range(formation.shape[0]), formation['delta_e'], marker='.')
ax.set_xlabel('index')
ax.set_ylabel('delta_e')
ax.set_yscale('log')
plt.show()
'''

print(df)
print(stability)
print(formation)

stability.to_csv('oqdm_stability.csv', index=False)
formation.to_csv('oqdm_formation.csv', index=False)
