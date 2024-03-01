from ann_savi_learnable import ANNSAVILearnable


class ANNSAVILearnableHalf(ANNSAVILearnable):
    def __init__(self, train_ds, test_ds, validation_ds):
        super().__init__(train_ds, test_ds, validation_ds)
        self.L.requires_grad = False

    def after_epoch(self, epoch):
        if epoch == self.num_epochs/2:
            self.L.requires_grad = True
