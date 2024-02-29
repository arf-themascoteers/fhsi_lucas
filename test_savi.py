from evaluator import Evaluator

if __name__ == "__main__":
    c = Evaluator(
        prefix="savi",
        folds=3,
        algorithms=[
            "ann_savi_learnable"
        ],
        feature_sets= [
            []
        ]
    )
    c.process()