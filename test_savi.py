from evaluator import Evaluator

if __name__ == "__main__":
    c = Evaluator(
        prefix="savi",
        folds=10,
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