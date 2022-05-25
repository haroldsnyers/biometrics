import attr
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.utils.np_utils import to_categorical


@attr.s
class DataPreProcessor:
    images_train_x = attr.ib(default=None)
    images_train_y = attr.ib(default=None)
    images_test_x = attr.ib(default=None)
    images_test_y = attr.ib(default=None)
    n_classes = attr.ib(default=None)
    batch_size = attr.ib(default=16)
    _train_datagen = attr.ib(default=None)
    _val_datagen = attr.ib(default=None)
    _test_datagen = attr.ib(default=None)

    def __attrs_post_init__(self):
        self.generate_train_data_gen()
        self.generate_test_data_gen()
        self._train_set, self._val_set = self.generate_train_data_set(self.images_train_x, self.images_train_y)
        self._test_set = self.generate_test_data_set(self.images_test_x, self.images_test_y)

    @property
    def get_train_set(self):
        return self._train_set

    @property
    def get_val_set(self):
        return self._train_set

    @property
    def get_test_set(self):
        return self._test_set

    @property
    def get_total_images_count(self):
        return len(self.images_train_x) + len(self.images_test_x)

    def generate_train_data_gen(self):
        self._train_datagen = ImageDataGenerator(
            rescale=1. / 255,  # feature scaling (like normalization -> put every pixel between 0 and 1
            zoom_range=0.1,
            rotation_range=10,
            brightness_range=(0.3, 1.3),
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            validation_split=0.1)

    # def generate_val_data_gen(self):
    #     self._val_datagen = ImageDataGenerator(rescale=1. / 255)

    def generate_test_data_gen(self):
        self._test_datagen = ImageDataGenerator(rescale=1. / 255)

    def generate_train_data_set(self, x, y):
        y = to_categorical(y, self.n_classes)
        train_data = self._train_datagen.flow(x=x, y=y, batch_size=self.batch_size, shuffle=True, subset='training')
        validation_data = self._train_datagen.flow(x=x, y=y, batch_size=self.batch_size, shuffle=True,
                                                   subset='validation')

        return train_data, validation_data

    def generate_test_data_set(self, x, y):
        test_data = self._test_datagen.flow(x=x, y=y, batch_size=self.batch_size, shuffle=False)
        return test_data
