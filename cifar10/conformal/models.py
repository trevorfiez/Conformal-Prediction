import os
from keras.models import load_model as lm


class Models:
    def __init__(self):
        self.modals_paths = 'models'


class Cifar10(Models):
    def __init__(self, soft_max = True):
        self.modal_path = os.path.join(Models.modals_paths, 'cifar10_cnn.h5')
        self.soft_max = soft_max

    def load_modal(self):
        if self.soft_max:
            return lm(self.modal_path)
