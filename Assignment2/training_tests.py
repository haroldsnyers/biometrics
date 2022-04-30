from datetime import datetime
import os
import glob
import shutil
import sys
from src.neural_network.data_preprocessing import DataPreProcessor
from src.neural_network.model_class import TransferModels, IrisModel
import tensorflow as tf
import src.image_processing as image_processor
import src.plotting_functions as plotter
import src.matrix_construction as matrix_constructor
import pickle
import cv2
import numpy as np

input_folder = "data/dataset/"
output_folder = "data/datasetv2/"
imagePaths = sorted(os.listdir(input_folder))
if not os.path.exists(output_folder):
    os.mkdir(output_folder)
    for id in imagePaths:
        for side, n in zip(['L', 'R'], [1, 2]):
            src_dir = input_folder + id + '/' + side
            dst_dir = output_folder + id + str(n)
            os.mkdir(dst_dir)
            for jpgfile in glob.iglob(os.path.join(src_dir, "*.jpg")):
                shutil.copy(jpgfile, dst_dir)

folders = list(os.walk(output_folder))[1:]

# remove empty folders and folders with not enough data
for folder in folders:
    # folder example: ('FOLDER/3', [], ['file'])
    if len(folder[2]) < 4:
        for imagePaths in os.listdir(folder[0]):
            os.remove(folder[0] + '/' + imagePaths)
        os.rmdir(folder[0])


### Split the data into train, val and test folders

import splitfolders

# Split with a ratio.
# To only split into training and validation set, set a tuple to `ratio`, i.e, `(.8, .2)`.
out_folder = "data/datasetv2_prepped/"
if not os.path.exists(out_folder):
    splitfolders.fixed(output_folder, output=out_folder,
        seed=1337, fixed=(2, 1), group_prefix=None, move=False) # default values

train_images, train_labels, train_labels_ids, train_ids = image_processor.read_iris_casia_db(out_folder +'\\train\\')
val_images, val_labels, val_labels_ids, val_ids = image_processor.read_iris_casia_db(out_folder + '\\val\\')
test_images, test_labels, test_labels_ids, test_ids = image_processor.read_iris_casia_db(out_folder + '\\test\\')
print(len(train_images), len(val_images), len(test_images))

try:
    with open("data/iris_mask_dataset_v2_db_train.pkl", "rb") as f:
        enhanced_iris_train, mask_iris_train = pickle.load(f)
except:
    enhanced_iris_train, mask_iris_train = image_processor.enhance_and_segment_irises(train_images, radial_res=200, angular_res=500)
    with open("data/iris_mask_dataset_v2_db_train.pkl", "wb") as f:
        pickle.dump([enhanced_iris_train, mask_iris_train], f)

try:
    with open("data/iris_mask_dataset_v2_db_val.pkl", "rb") as f:
        enhanced_iris_val, mask_iris_val = pickle.load(f)
except:
    enhanced_iris_val, mask_iris_val = image_processor.enhance_and_segment_irises(val_images, radial_res=200, angular_res=500)
    with open("data/iris_mask_dataset_v2_db_val.pkl", "wb") as f:
        pickle.dump([enhanced_iris_val, mask_iris_val], f)

try:
    with open("data/iris_mask_dataset_v2_db_test.pkl", "rb") as f:
        enhanced_iris_test, mask_iris_test = pickle.load(f)
except:
    enhanced_iris_test, mask_iris_test = image_processor.enhance_and_segment_irises(test_images, radial_res=200, angular_res=500)
    with open("data/iris_mask_dataset_v2_db_test.pkl", "wb") as f:
        pickle.dump([enhanced_iris_test, mask_iris_test], f)

n=2
train_segmented_db = np.array([a*b for a,b in zip(enhanced_iris_train, mask_iris_train)])
val_segmented_db = np.array([a*b for a,b in zip(enhanced_iris_val, mask_iris_val)])
test_segmented_db = np.array([a*b for a,b in zip(enhanced_iris_test, mask_iris_test)])

plotter.plot_image_sequence(enhanced_iris_train.tolist()[1:n] + mask_iris_train.tolist()[1:n] + train_segmented_db.tolist()[1:n], 3*(n-1), 3,  cmap='gray')

train_set = np.array([cv2.cvtColor(gray,cv2.COLOR_GRAY2RGB) for gray in train_segmented_db])
val_set = np.array([cv2.cvtColor(gray,cv2.COLOR_GRAY2RGB) for gray in val_segmented_db])
test_set = np.array([cv2.cvtColor(gray,cv2.COLOR_GRAY2RGB) for gray in test_segmented_db])

data_processor = DataPreProcessor(images_train_x=train_set, images_train_y=train_labels_ids,
                                  images_val_x=val_set, images_val_y=val_labels_ids,
                                  images_test_x=test_set, images_test_y=test_labels_ids,
                                  image_shape=train_set[0].shape)

n = 7
plotter.plot_image_sequence(train_images, n)

# Markdown("""
# Let's have a look at the dataset... It contains {} images and the size of the images is {}×{}. It has {} classes. Let's visualise the first 7 images:
# """.format(data_processor.get_total_images_count, data_processor.image_shape[0], data_processor.image_shape[1], data_processor.get_iris_count))

training_set = data_processor.get_train_set
val_set = data_processor.get_val_set
test_set = data_processor.get_test_set

model = IrisModel(optimiser='adam', input_size=train_set[0].shape, n_classes=data_processor.get_iris_count)
model.compute_model(transfer_model=TransferModels.vgg16)

print(model.get_model_summary)

model.fit(training_set, val_set, model_choice='vgg16', epochs=80)

model_vgg16 = model.get_model
model_vgg16 = model.activate_training_layers(model_vgg16, 17)
model.set_model(model_vgg16)
model.compile_model(opt=tf.keras.optimizers.Adam(1e-5)) ## slow learning

model.fit(training_set, val_set, model_choice='vgg16', epochs=20)

model.plot_accuracy_and_loss_plt()
