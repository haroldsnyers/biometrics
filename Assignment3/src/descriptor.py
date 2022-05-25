from enum import Enum

import attr
import localmodules.siamese as siamese
import numpy as np
from icecream import ic
from localmodules.local_binary_patterns import LBP
from scipy.spatial.distance import euclidean
from scipy.stats import chisquare
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from tensorflow.keras.optimizers import Adam
# To visualize your model structure:
from tensorflow.keras.utils import plot_model


class FeatureDescriptor(Enum):
    LBP = 'LBP'
    PCA = 'PCA'
    LDA = 'LDA'
    DL = 'DL'
    TL = 'TL'
    VGG = 'VGG'


@attr.s
class FacialDescriptor:
    num_components = attr.ib(default=35)
    faces = attr.ib(default=None)
    n_samples = attr.ib(default=None)
    n_features = attr.ib(default=None)
    n_classes = attr.ib(default=None)
    imshape = attr.ib(default=None)

    def extract_face_representation(self, DESC, holdout_split=None, transfer_system=None):
        print(DESC == FeatureDescriptor.DL)
        if DESC == FeatureDescriptor.PCA:
            # Compute a PCA (eigenfaces) on the face dataset
            num_components = min(self.num_components, min(self.n_samples, self.n_features))
            print("num_components {n}".format(n=num_components))
            desc = PCA(n_components=num_components, svd_solver='randomized', whiten=True).fit(self.faces.data)
            X_pca = desc.transform(self.faces.data)
            embedded = X_pca

            dist_metric = euclidean

        if DESC == FeatureDescriptor.LDA:
            num_components = min(self.num_components, min(self.n_classes - 1, self.n_features))
            desc = LinearDiscriminantAnalysis(n_components=num_components).fit(self.faces.data, self.faces.target)
            X_lda = desc.fit_transform(self.faces.data, self.faces.target)
            embedded = X_lda
            dist_metric = euclidean

        if DESC == FeatureDescriptor.LBP:
            ic(DESC)
            desc = LBP(numPoints=8, radius=1, grid_x=7, grid_y=7)
            embedded = desc.describe_list(self.faces.images[..., 0])
            dist_metric = chisquare

            # if np.isnan(dist_metric).all():
            #     dist_metric = self.CHI2

        if DESC == FeatureDescriptor.DL:
            x_train, x_test, y_train, y_test = holdout_split(
                *siamese.get_siamese_paired_data(self.faces.images, self.faces.target))

            model, encoder = self.get_model()

            rms = Adam()
            model.compile(
                loss=siamese.contrastive_loss,
                optimizer=rms,
                metrics=[siamese.accuracy],
                run_eagerly=True)

            epochs = 10
            model.fit([x_train[:, 0], x_train[:, 1]], y_train,
                      validation_split=0.2,
                      batch_size=32, verbose=2, epochs=epochs)

            test_scores = model.predict([x_test[:, 0], x_test[:, 1]])
            test_acc = accuracy_score(y_test, test_scores > 0.5)
            print("Accuracy on the test set: {}".format(test_acc))
            embedded = encoder(self.faces.images.astype(np.float32)).numpy()

            dist_metric = euclidean

        if DESC == FeatureDescriptor.TL:
            model = transfer_system['model']
            encoder = transfer_system['encoder']

            x_train, x_test, y_train, y_test = holdout_split(
                *siamese.get_siamese_paired_data(self.faces.images, self.faces.target))

            rms = Adam()
            model.compile(
                loss=siamese.contrastive_loss,
                optimizer=rms,
                metrics=[siamese.accuracy],
                run_eagerly=True)

            epochs = 10
            model.fit([x_train[:, 0], x_train[:, 1]], y_train,
                      validation_split=0.2,
                      batch_size=32, verbose=2, epochs=epochs)

            test_scores = model.predict([x_test[:, 0], x_test[:, 1]])
            test_acc = accuracy_score(y_test, test_scores > 0.5)
            print("Accuracy on the test set: {}".format(test_acc))
            embedded = encoder(self.faces.images.astype(np.float32)).numpy()

            dist_metric = euclidean

        return embedded, dist_metric

    def get_model(self):
        encoder, model = siamese.create_siamese_model(self.imshape)
        model.summary()

        plot_model(model, to_file='results/model.png', show_shapes=True, show_layer_names=True)

        return model, encoder
