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

#df = df.sample(frac=0.01)  # for debugging

df = df.drop_duplicates()
df = df.dropna()

df['name'] = df['name'].apply(name_convention)
df = df.sort_values(by=['stability', 'delta_e', 'name', 'entry_id'])

# Keep most stable compositions
df = df.drop_duplicates(subset='name', keep='first')

# Remove some large formation energies by a cutoff
df = df[df['delta_e'] <= 3.0]

stability = df[['name', 'stability']]
formation = df[['name', 'delta_e']]

# Sort to make pretty
stability = stability.sort_values(by=['stability', 'name'])
formation = formation.sort_values(by=['delta_e', 'name'])

print(df)
print(stability)
print(formation)

stability.to_csv('oqmd_stability.csv', index=False)
formation.to_csv('oqmd_formation.csv', index=False)
