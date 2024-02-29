from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("data/mangrove.csv")
#df = df[["c","Band_4","Band_8"]]

# for column in df.columns:
#     scaler = MinMaxScaler()
#     scaled_column = scaler.fit_transform(df[[column]])
#     df[column] = scaled_column.flatten()

data = df.to_numpy()

r2s = []

kf = KFold(n_splits=10)
for i, (train_index, test_index) in enumerate(kf.split(data)):
    train_data = data[train_index]
    test_data = data[test_index]
    train_x = train_data[:, 1:]
    train_y = train_data[:, 0]
    test_x = test_data[:, 1:]
    test_y = test_data[:, 0]
    model_instance = RandomForestRegressor(n_estimators=100, max_depth=10)
    model_instance = model_instance.fit(train_x, train_y)
    y_hat = model_instance.predict(test_x)
    score = r2_score(test_y, y_hat)
    r2s.append(score)

print(r2s)
print(sum(r2s)/len(r2s))


