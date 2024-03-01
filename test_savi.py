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
            "ann_savi_learnable_half",

            "ann_savi_skip",
            "ann_savi_skip_learnable",
            "ann_savi_skip_learnable_half",

            "ann_savi_skip_all",
            "ann_savi_skip_all_learnable",
            "ann_savi_skip_all_learnable_half",

            "ann_savi_learnable_fn",
            "ann_savi_learnable_fn_all",
            "ann_savi_learnable_bi",

            "ann_savi_learnable_fn_skip",
            "ann_savi_learnable_fn_all_skip",
            "ann_savi_learnable_bi_skip",

            "ann_savi_bands_only"
        ],
        feature_sets= [
            []
        ]
    )
    c.process()