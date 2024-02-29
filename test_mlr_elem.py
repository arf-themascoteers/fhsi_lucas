import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from torch.utils.data import DataLoader
from soil_dataset import SoilDataset
from ds_manager import DSManager
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("data/lucas_s2.csv")
df = df[["b4","b8","oc"]]

for column in df.columns:
    scaler = MinMaxScaler()
    scaled_column = scaler.fit_transform(df[[column]])
    df[column] = scaled_column.flatten()

data = df.to_numpy()
deno = data[:,0]+data[:,1]
deno[deno==0]=0.00001
ndvi = (data[:,0]-data[:,1])/(deno)
ndvi = ndvi.reshape(-1,1)
data = np.concatenate((ndvi, data), axis=1)

r2s = []

kf = KFold(n_splits=10)
for i, (train_index, test_index) in enumerate(kf.split(data)):
    train_data = data[train_index]
    test_data = data[test_index]
    train_x = train_data[:, 0:-1]
    train_y = train_data[:, -1]
    test_x = test_data[:, 0:-1]
    test_y = test_data[:, -1]
    model_instance = LinearRegression()
    model_instance = model_instance.fit(train_x, train_y)
    r2 = model_instance.score(test_x, test_y)
    print(r2)
    r2s.append(r2)

r2s_p = []
for r in r2s:
    if r > 0:
        r2s_p.append(r)
print(sum(r2s_p)/len(r2s_p))


