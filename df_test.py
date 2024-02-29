import pandas as pd

data = {'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]}
df = pd.DataFrame(data)

new_column_values = [10, 11, 12]
new_column_name = 'NewColumn'

df.insert(len(df.columns)-1, new_column_name, new_column_values)

df['NewColumn'] = df["A"] + df["B"]

print(df)