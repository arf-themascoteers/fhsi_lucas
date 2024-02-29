from reporter import Reporter
from ds_manager import DSManager
import utils
from ann_simple import ANNSimple


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
            print("Start", f"{repeat_number}:{self.algorithms[index_algorithm]} - {config}")
            self.process_config(repeat_number, index_algorithm, index_config)

    def process_config(self, repeat_number, index_algorithm, index_config):
        algorithm = self.algorithms[index_algorithm]
        feature_set = self.feature_sets[index_config]
        ds = DSManager(self.folds, feature_set)
        for fold_number, (train_ds, test_ds, validation_ds) in enumerate(ds.get_k_folds()):
            r2, rmse, pc = self.reporter.get_details(index_algorithm, repeat_number, fold_number, index_config)
            if r2 != 0:
                print(f"{repeat_number}-{fold_number} done already")
                continue
            else:
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
        model_instance = clazz(train_ds, test_ds, validation_ds)
        r2, rmse, pc = model_instance.run()
        return max(r2,0), rmse, pc