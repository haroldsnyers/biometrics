import datetime
from os.path import isfile, join, exists
from os import mkdir

import attr
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from tensorflow.keras import Model

from plotly.offline import iplot
from plotly.subplots import make_subplots
from icecream import ic

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.python.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.python.keras.utils.vis_utils import plot_model
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau

from tensorflow.keras.applications import vgg16, inception_v3, resnet, resnet_v2, inception_resnet_v2


class TransferModels:
    vgg16 = vgg16.VGG16
    inception_V3 = inception_v3.InceptionV3
    inception_resnet_V2 = inception_resnet_v2.InceptionResNetV2
    resnet = resnet.ResNet50
    resnet_v2 = resnet_v2.ResNet50V2


@attr.s
class IrisModel:
    input_size = attr.ib(default=(200, 500, 3))
    n_classes = attr.ib(default=7)
    optimiser = attr.ib(default='adam')
    hist = attr.ib(default=None)
    _model = attr.ib(default=None, type=Sequential)
    _train_loss = attr.ib(default=None)
    _train_acc = attr.ib(default=None)
    _test_loss = attr.ib(default=None)
    _test_acc = attr.ib(default=None)

    SAVE_DIRECTORY = '../model/'

    @property
    def get_model_summary(self):
        return self._model.summary()

    @property
    def get_model(self):
        return self._model

    def set_model(self, model):
        self._model = model

    def __attrs_post_init__(self):
        pass

    def compute_model(self, transfer_model):
        self._model = self._build_model(transfer_model)
        self.compile_model(opt=self.optimiser)

    def _build_model(self, transfer_model, trainable_layers=None):
        """
        Model architecture
        :param input_shape: shape of image
        :param num_classes: number of different persons you want to identify
        :param model_n: model name - default : 'imagenet', other options: resnet50, vgg16, senet50
        :param dropout: added layers architecture : default: 'dropout', other options: dropout_simple or other
        :param global_avg: last layer extraction default: True, other False
        :param trainable_l: layer number from which the transfer model fine tune its weights based on new data set
                default None, other int
        :return: (Model) fined tuned transfer model
        """

        # create the base pre-trained model
        base_model = transfer_model(
                weights='imagenet', include_top=False, input_shape=self.input_size)

        base_model = self.activate_training_layers(base_model, trainable_layers)

        print(base_model.summary())
        # add a global spatial average pooling layer
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        # let's add a fully-connected layer
        x = Dense(1024, activation='relu')(x)
        # and a logistic layer -- let's say we have 200 classes
        outputs = Dense(self.n_classes, activation='softmax')(x)

        return Model(base_model.inputs, outputs)

    @staticmethod
    def activate_training_layers(tf_model, trainable_l=None):
        """
        Activate the learninng of certain layers for a given model
        :param tf_model: transfer learning model from which to regulate the training of its layers weights
        :param trainable_l: layer number from which the transfer model fine tune its weights based on new data set
        :return: transfer model with training parameters for layers
        """
        ic(len(tf_model.layers))
        # Freeze all transfer model layers
        if trainable_l is None:
            tf_model.trainable = False
        else:
            # Freeze transfer model layers until trainable_l
            for layer in tf_model.layers[:trainable_l]:
                layer.trainable = False

        return tf_model

    def compile_model(self, opt):
        self._model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    def fit(self, training_set, val_set, epochs=20, model_choice=None):
        log_dir = "logs/" + model_choice + '/'
        if not exists(log_dir):
            mkdir(log_dir)

        log_dir = log_dir + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        csv_logger = CSVLogger(log_dir, append=False)
        # early_stop = EarlyStopping('val_loss', patience=25)
        trained_models_path = "neural_network/model/" + model_choice
        reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1, patience=int(25/2), verbose=1)
        model_names = trained_models_path + '.{epoch:02d}-{val_accuracy:.2f}.hdf5'
        model_checkpoint = ModelCheckpoint(model_names, 'val_loss', verbose=1,save_best_only=True)
        callbacks = [model_checkpoint, csv_logger, reduce_lr]
        # tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1,)

        self.hist = self._model.fit(x=training_set,
                                    validation_data=val_set,
                                    epochs=epochs,
                                    callbacks=callbacks,
                                    shuffle=True
                                    # use_multiprocessing=True,
                                    # workers=8
                                    )

    def evaluate_model(self, training_set, test_set):
        self._train_loss, self._train_acc = self.compute_accuracy_and_loss(training_set)
        self._test_loss, self._test_acc = self.compute_accuracy_and_loss(test_set)

        print("final train accuracy = {:.2f} , validation accuracy = {:.2f}".format(self._train_acc * 100,
                                                                                    self._test_acc * 100))
        print("final train loss = {:.2f} , validation loss = {:.2f}".format(self._train_loss * 100,
                                                                            self._test_loss * 100))

    def generate_model_plot(self, filename):
        plot_model(self._model, to_file=filename + '.png', show_shapes=True, show_layer_names=True)

    def save_model(self, filename):
        self._model.save(filename)
        print('[+] Model trained and saved at ' + filename)

    def load_model(self, filename):
        filename_dir = filename + '.txt'
        if isfile(join(self.SAVE_DIRECTORY, filename_dir)):
            self._model.load(join(self.SAVE_DIRECTORY, filename_dir))
            print('[+] Model loaded from ' + filename_dir)

    def compute_accuracy_and_loss(self, eval_set):
        return self._model.evaluate(eval_set)

    def plot_accuracy_and_loss_plt(self):
        plt.figure(figsize=(14, 5))
        plt.subplot(1, 2, 2)
        plt.plot(self.hist.history['accuracy'])
        plt.plot(self.hist.history['val_accuracy'])
        plt.title('Model Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend(['train', 'test'], loc='upper left')
        plt.subplot(1, 2, 1)
        plt.plot(self.hist.history['loss'])
        plt.plot(self.hist.history['val_loss'])
        plt.title('model Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

    def plot_accuracy_and_loss_plotly(self):
        fig = make_subplots(rows=1, cols=2, subplot_titles=['Model Accuracy', 'Model Loss'])
        fig.add_trace(go.Scatter(
            self.hist.history['accuracy'],
            mode='lines',
            name='train',
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            self.hist.history['val_accuracy'],
            mode='lines',
            name='test',
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            self.hist.history['loss'],
            mode='lines',
            name='train',
        ), row=1, col=2)

        fig.add_trace(go.Scatter(
            self.hist.history['val_loss'],
            mode='lines',
            name='test',
        ), row=1, col=2)

        iplot(fig)