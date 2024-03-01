from reporter import Reporter
from ds_manager import DSManager
import utils
from ann_simple import ANNSimple

from ann_savi_bands_only import ANNSAVIBandsOnly

from ann_savi import ANNSAVI
from ann_savi_skip import ANNSAVISkip
from ann_savi_skip_all import ANNSAVISkipAll

from ann_savi_learnable import ANNSAVILearnable
from ann_savi_skip_learnable import ANNSAVISkipLearnable
from ann_savi_skip_all_learnable import ANNSAVISkipAllLearnable

from ann_savi_learnable_fn import ANNSAVILearnableFn
from ann_savi_learnable_fn_all import ANNSAVILearnableFnAll
from ann_savi_learnable_bi import ANNSAVILearnableBI

from ann_savi_learnable_fn_skip import ANNSAVILearnableFnSkip
from ann_savi_learnable_fn_all_skip import ANNSAVILearnableFnAllSkip
from ann_savi_learnable_bi_skip import ANNSAVILearnableBISkip

from ann_savi_learnable_half import ANNSAVILearnableHalf
from ann_savi_skip_learnable_half import ANNSAVISkipLearnableHalf
from ann_savi_skip_all_learnable_half import ANNSAVISkipAllLearnableHalf



class Evaluator:
    def __init__(self, prefix="", verbose=False, repeat=1, folds=10, algorithms=None, feature_sets=None):
        self.repeat = repeat
        self.folds = folds
        self.verbose = verbose
        self.algorithms = algorithms

        if self.algorithms is None:
            self.algorithms = ["ann_simple"]

        if feature_sets is None:
            feature_sets = [utils.get_all_features()]

        self.feature_sets = []

        for feature_set in feature_sets:
            if len(feature_set) == 0:
                feature_set = utils.get_all_features()
            self.feature_sets.append(feature_set)

        self.reporter = Reporter(prefix, self.feature_sets, self.algorithms, self.repeat, self.folds)

    def process(self):
        for repeat_number in range(self.repeat):
            self.process_repeat(repeat_number)

    def process_repeat(self, repeat_number):
        for index_algorithm, algorithm in enumerate(self.algorithms):
            self.process_algorithm(repeat_number, index_algorithm)

    def process_algorithm(self, repeat_number, index_algorithm):
        for index_config in range(len(self.feature_sets)):
            config = self.feature_sets[index_config]
            print("Trying", f"{repeat_number}:{self.algorithms[index_algorithm]} - {config}")
            self.process_config(repeat_number, index_algorithm, index_config)

    def process_config(self, repeat_number, index_algorithm, index_config):
        algorithm = self.algorithms[index_algorithm]
        feature_set = self.feature_sets[index_config]
        ds = DSManager(self.folds, feature_set)
        for fold_number, (train_ds, test_ds, validation_ds) in enumerate(ds.get_k_folds()):
            r2, rmse, pc = self.reporter.get_details(index_algorithm, repeat_number, fold_number, index_config)
            if rmse != 0:
                print(f"{repeat_number}-{fold_number} done already")
                continue
            else:
                print("Start", f"{repeat_number}:{self.algorithms[index_algorithm]} - fold {fold_number}")
                r2, rmse, pc = Evaluator.calculate_score(train_ds, test_ds, validation_ds, algorithm)
            if self.verbose:
                print(f"{r2} - {rmse} - {pc}")
            self.reporter.set_details(index_algorithm, repeat_number, fold_number, index_config, r2, rmse, pc)
            self.reporter.write_details()
            self.reporter.update_summary()

    @staticmethod
    def calculate_score(train_ds, test_ds, validation_ds,algorithm):
        print(f"Train: {len(train_ds.y)}, Test: {len(test_ds.y)}, Validation: {len(validation_ds.y)}")
        clazz = None
        if algorithm == "ann_simple":
            clazz = ANNSimple
        elif algorithm == "ann_savi_bands_only":
            clazz = ANNSAVIBandsOnly
        elif algorithm == "ann_savi":
            clazz = ANNSAVI
        elif algorithm == "ann_savi_skip":
            clazz = ANNSAVISkip
        elif algorithm == "ann_savi_skip_all":
            clazz = ANNSAVISkipAll
        elif algorithm == "ann_savi_learnable":
            clazz = ANNSAVILearnable
        elif algorithm == "ann_savi_skip_learnable":
            clazz = ANNSAVISkipLearnable
        elif algorithm == "ann_savi_skip_all_learnable":
            clazz = ANNSAVISkipAllLearnable
        elif algorithm == "ann_savi_learnable_fn":
            clazz = ANNSAVILearnableFn
        elif algorithm == "ann_savi_learnable_fn_all":
            clazz = ANNSAVILearnableFnAll
        elif algorithm == "ann_savi_learnable_bi":
            clazz = ANNSAVILearnableBI

        elif algorithm == "ann_savi_learnable_fn_skip":
            clazz = ANNSAVILearnableFnSkip
        elif algorithm == "ann_savi_learnable_fn_all_skip":
            clazz = ANNSAVILearnableFnAllSkip
        elif algorithm == "ann_savi_learnable_bi_skip":
            clazz = ANNSAVILearnableBISkip

        elif algorithm == "ann_savi_learnable_half":
            clazz = ANNSAVILearnableHalf
        elif algorithm == "ann_savi_skip_learnable_half":
            clazz = ANNSAVISkipLearnableHalf
        elif algorithm == "ann_savi_skip_all_learnable_half":
            clazz = ANNSAVISkipAllLearnableHalf

        model_instance = clazz(train_ds, test_ds, validation_ds)
        r2, rmse, pc = model_instance.run()
        return max(r2,0), rmse, pc