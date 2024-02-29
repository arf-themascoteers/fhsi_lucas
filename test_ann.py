from ann_simple import ANNSimple
from ds_manager import DSManager

dm = DSManager("data/full.csv")
r2s = []

for fold_number, (train_x, train_y, test_x, test_y, validation_x, validation_y) in enumerate(dm.get_k_folds()):
    ann = ANNSimple(train_x, train_y, test_x, test_y, validation_x, validation_y)
    r2, rmse, pc = ann.run()
    print(f"r2={r2:.3f}, rmse={rmse:.3f}, pc={pc:.3f}")

print(r2s)
print(sum(r2s)/len(r2s))


