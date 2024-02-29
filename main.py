from evaluator import Evaluator

if __name__ == "__main__":
    c = Evaluator(
        prefix="mixed",
        folds=3,
        algorithms=[
            "ann_simple"
        ],
        feature_sets= [
            ["ndvi"],
            ["b4","b8"],
            ["b4","b8","ndvi"]
        ]
    )
    c.process()