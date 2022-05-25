import os

import attr
import matplotlib as plt
import pandas as pd
from icecream import ic
from keras_vggface.vggface import VGGFace
from tensorflow.keras import Model
from tensorflow.keras.applications import vgg16, vgg19, inception_v3, resnet, resnet_v2, inception_resnet_v2, nasnet
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import plot_model


class TransferModels:
    vgg16_ = vgg16.VGG16
    vgg19_ = vgg19.VGG19
    inception_V3 = inception_v3.InceptionV3
    inception_resnet_V2 = inception_resnet_v2.InceptionResNetV2
    resnet50 = resnet.ResNet50
    resnet_50v2 = resnet_v2.ResNet50V2
    resnet_101 = resnet.ResNet101
    resnet_101v2 = resnet_v2.ResNet101V2
    nasnet_large = nasnet.NASNetLarge
    resnet50_vggface = 'resnet50'
    vgg16_vggface = 'vgg16'
    senet50_vggface = 'senet50'


@attr.s
class ClassificationModel:
    input_size = attr.ib(default=(47, 47, 3))
    n_classes = attr.ib(default=7)
    optimiser = attr.ib(default='adam')
    hist = attr.ib(default=None)
    _model = attr.ib(default=None, type=Sequential)
    _train_loss = attr.ib(default=None)
    _train_acc = attr.ib(default=None)
    _test_loss = attr.ib(default=None)
    _test_acc = attr.ib(default=None)

    LASTLAYER = {
        TransferModels.resnet50_vggface: 'avg_pool',
        TransferModels.senet50_vggface: 'avg_pool',
        TransferModels.vgg16_vggface: 'pool5'
    }

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

    def compute_model(self, transfer_model, pretrained=True, vggface=True):
        self._model = self._build_model(transfer_model, pretrained=pretrained, vggface=vggface)
        self.compile_model(opt=self.optimiser)

    def _build_model(self, transfer_model, vggface=False, trainable_layers=None, pretrained=True):
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
        if pretrained:
            if vggface:
                base_model = VGGFace(model=transfer_model, include_top=False, input_shape=self.input_size,
                                     pooling='avg')
            else:
                base_model = transfer_model(
                    weights='imagenet', include_top=False, input_shape=self.input_size)

            base_model = self.activate_training_layers(base_model, trainable_layers)
        else:
            base_model = transfer_model(
                weights=None, include_top=False, input_shape=self.input_size)

        print(base_model.summary())
        # add a global spatial average pooling layer
        if vggface:
            last_layer = base_model.get_layer(self.LASTLAYER[transfer_model]).output
            x = Flatten()(last_layer)
        else:
            last_layer = base_model.output
            x = GlobalAveragePooling2D()(last_layer)

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
        self._model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

    def fit(self, training_set, val_set, epochs=20, model_choice=None, fine_tuning=False):

        weights_filename = 'best_weights_fine_' if fine_tuning else 'best_weights_'
        weights_filename = weights_filename + '{epoch:02d}-{val_accuracy:.2f}.hdf5'
        print(weights_filename)
        # Work around to check if we are working locally or in kaggle environment
        if os.path.exists('/kaggle/'):
            # create directory for saving results
            ckpt_path = '/kaggle/working/ckpts/output/'
            if not os.path.exists(ckpt_path):
                os.mkdir(ckpt_path)
            self.path_weights = ckpt_path + model_choice + '/' + weights_filename
        else:
            self.path_weights = 'output/' + model_choice + '/' + weights_filename

        reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1, patience=int(25 / 2), verbose=1)
        model_checkpoint = ModelCheckpoint(self.path_weights, 'val_loss', verbose=1, save_best_only=True)
        callbacks = [model_checkpoint, reduce_lr]

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

    def load_model(self, filename, model_choice):
        if os.path.exists('/kaggle/'):
            # create directory for saving results
            ckpt_path = '/kaggle/working/ckpts/output/'
            if not os.path.exists(ckpt_path):
                os.mkdir(ckpt_path)
            path_weights = ckpt_path + model_choice + '/' + filename
        else:
            path_weights = 'output/' + model_choice + '/' + filename
        if os.path.isfile(path_weights):
            self._model.load_weights(os.path.join(path_weights))
            print('[+] Model loaded from ' + model_choice + '/' + filename)

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

    def predict(self, test_data):
        predictions = self._model.predict(test_data)

        data = pd.DataFrame(predictions, dtype=float)

        return data.values
