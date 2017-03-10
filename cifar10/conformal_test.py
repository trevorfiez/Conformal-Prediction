from __future__ import division
from __future__ import print_function
from conformal import ConformalPrediction
from conformal.measures import SoftMax, Binary, Diff
from keras.datasets import cifar10
from keras.models import load_model
from keras.utils import np_utils
from keras.models import Model
import os


def __ensure_directory_exits(directory_path):
    """creates directory if path doesn't exist"""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    return directory_path


if __name__ == '__main__':
    models_path = __ensure_directory_exits(os.path.join(os.getcwd(), 'models'))
    conformal_data_path = __ensure_directory_exits(os.path.join(os.getcwd(), 'conformal_data'))
    histogram_path = __ensure_directory_exits(os.path.join(conformal_data_path, 'histogram'))

    # initial_configuration
    batch_size = 32
    nb_classes = 10
    nb_epoch = 30

    # input image dimensions
    img_rows, img_cols = 32, 32
    # the CIFAR10 images are RGB
    img_channels = 3

    # the data, shuffled and split between train and test sets
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    # without soft_max layer
    epsilon = 5
    model = load_model(os.path.join(models_path, 'cifar10_cnn_model_step_1_after_dropout_removal.h5'))
    pre_sigmoid_layer_model = Model(input=model.input,
                                     output=model.layers[-2].output)
    prediction = pre_sigmoid_layer_model.predict(X_test, batch_size=batch_size)

    cf_binary = ConformalPrediction(prediction.copy(), Y_test, 5, measure=Binary())
    cf_prediction = cf_binary.predict(prediction.copy())
    cf_accuracy = cf_binary.evaluate(cf_prediction, Y_test)
    cf_label_histogram = cf_binary.label_histogram(cf_prediction, os.path.join(histogram_path, 'with_binary_no_sm'),
                                                   'Conformal Prediction with Binary - epsilon:' + str(
                                                       epsilon) + ' ( without soft_max layer)')
    print('Accuracy of Conformal Prediction with Binary Measure:', cf_accuracy)
    print('Labels Histogram:', cf_label_histogram)

    epsilon = 5
    model = load_model(os.path.join(models_path, 'cifar10_cnn_model_step_1_after_dropout_removal.h5'))
    pre_sigmoid_layer_model = Model(input=model.input,
                                     output=model.layers[-2].output)
    prediction = pre_sigmoid_layer_model.predict(X_test, batch_size=batch_size)
    cf_softmax = ConformalPrediction(prediction, Y_test, 5, measure=SoftMax())
    cf_prediction = cf_softmax.predict(prediction)
    cf_accuracy = cf_softmax.evaluate(cf_prediction, Y_test)
    cf_label_histogram = cf_softmax.label_histogram(cf_prediction, os.path.join(histogram_path, 'with_softmax_no_sm'),
                                                    'Conformal Prediction with Softmax - epsilon:' + str(
                                                        epsilon) + '( without soft_max layer)')
    print('Accuracy of  Conformal Prediction with Softmax Measure:', cf_accuracy)
    print('Labels Histogram:', cf_label_histogram)

    epsilon = 5
    model = load_model(os.path.join(models_path, 'cifar10_cnn_model_step_1_after_dropout_removal.h5'))
    pre_sigmoid_layer_model = Model(input=model.input,
                                     output=model.layers[-2].output)
    prediction = pre_sigmoid_layer_model.predict(X_test, batch_size=batch_size)
    cf_diff = ConformalPrediction(prediction, Y_test, 5, measure=Diff())
    cf_prediction = cf_diff.predict(prediction)
    cf_accuracy = cf_diff.evaluate(cf_prediction, Y_test)
    cf_label_histogram = cf_diff.label_histogram(cf_prediction, os.path.join(histogram_path, 'with_diff_no_sm'),
                                                 'Conformal Prediction with Diff - epsilon:' + str(
                                                     epsilon) + '( without soft_max layer)')
    print('Accuracy of Conformal Prediction with Diff Measure:', cf_accuracy)
    print('Labels Histogram:', cf_label_histogram)

    # with soft_max
    epsilon = 5
    model = load_model(os.path.join(models_path, 'cifar10_cnn_model_step_1_after_dropout_removal.h5'))
    prediction = model.predict(X_test, batch_size=batch_size)
    cf_binary = ConformalPrediction(prediction, Y_test, 5, measure=Binary())
    cf_prediction = cf_binary.predict(prediction)
    cf_accuracy = cf_binary.evaluate(cf_prediction, Y_test)
    cf_label_histogram = cf_binary.label_histogram(cf_prediction, os.path.join(histogram_path, 'with_binary_and_sm'),
                                                   'Conformal Prediction with Binary - epsilon:' + str(
                                                       epsilon) + '( with soft_max layer)')
    print('Accuracy of Conformal Prediction with Binary Measure:', cf_accuracy)
    print('Labels Histogram:', cf_label_histogram)

    epsilon = 5
    model = load_model(os.path.join(models_path, 'cifar10_cnn_model_step_1_after_dropout_removal.h5'))
    prediction = model.predict(X_test, batch_size=batch_size)
    cf_softmax = ConformalPrediction(prediction, Y_test, 5, measure=SoftMax())
    cf_prediction = cf_softmax.predict(prediction)
    cf_accuracy = cf_softmax.evaluate(cf_prediction, Y_test)
    cf_label_histogram = cf_softmax.label_histogram(cf_prediction, os.path.join(histogram_path, 'with_softmax_and_sm'),
                                                    'Conformal Prediction with Softmax - epsilon:' + str(
                                                        epsilon) + '( with soft_max layer)')
    print('Accuracy of Conformal Prediction with Softmax Measure:', cf_accuracy)
    print('Labels Histogram:', cf_label_histogram)

    epsilon = 5
    model = load_model(os.path.join(models_path, 'cifar10_cnn_model_step_1_after_dropout_removal.h5'))
    prediction = model.predict(X_test, batch_size=batch_size)
    cf_diff = ConformalPrediction(prediction, Y_test, 5, measure=Diff())
    cf_prediction = cf_diff.predict(prediction)
    cf_accuracy = cf_diff.evaluate(cf_prediction, Y_test)
    cf_label_histogram = cf_diff.label_histogram(cf_prediction, os.path.join(histogram_path, 'with_diff_and_sm'),
                                                 'Conformal Prediction with Diff - epsilon:' + str(
                                                     epsilon) + ' ( with soft_max layer)')
    print('Accuracy of Conformal Prediction with Diff Measure:', cf_accuracy)
    print('Labels Histogram:', cf_label_histogram)