from evaluator import Evaluator
import utils

if __name__ == "__main__":
    folds = 10
    if utils.is_test():
        folds = 3
    c = Evaluator(
        prefix="savi",
        folds=folds,
        algorithms=[
            "ann_savi",
            "ann_savi_learnable",
            "ann_savi_skip",
            "ann_savi_skip_learnable",
            "ann_savi_skip_all",
            "ann_savi_skip_all_learnable",
            "ann_savi_bands_only",
            "ann_savi_learnable_fn"
        ],
        feature_sets= [
            []
        ]
    )
    c.process()