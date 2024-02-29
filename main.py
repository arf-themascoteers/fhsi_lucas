from evaluator import Evaluator

if __name__ == "__main__":
    c = Evaluator(
        prefix="mixed",
        folds=3,
        algorithms=[
            "ann_simple"
        ],
        features = [

        ]
    )
    c.process()