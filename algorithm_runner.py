import torch
from sklearn.metrics import mean_squared_error, r2_score
from ann_simple import ANNSimple


class AlgorithmRunner:
    @staticmethod
    def calculate_score(train_x, train_y,
                        test_x, test_y,
                        validation_x,
                        validation_y,
                        algorithm
                        ):
        print(f"Train: {len(train_y)}, Test: {len(test_y)}, Validation: {len(validation_y)}")
        clazz = None
        if algorithm == "ann_simple":
            clazz = ANNSimple
        model_instance = clazz(train_x, train_y, test_x, test_y, validation_x, validation_y)
        r2, rmse, pc = model_instance.run()
        return max(r2,0), rmse, pc