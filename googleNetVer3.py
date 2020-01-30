import logging
logging.basicConfig()
import sys
import os
import argparse
import random
import time
import datetime
from collections import Counter
import numpy as np
import pandas as pd
import shutil
from tqdm import tqdm
import inspect
import gc
import re
from PIL import Image
import cv2
import keras
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras import models
from keras.models import Model
from keras.models import Sequential
from keras.layers import Conv2D, Activation, MaxPooling2D, Dropout, GlobalAveragePooling1D, GlobalAveragePooling2D, Flatten, BatchNormalization, Dense
from keras.applications.inception_v3 import InceptionV3
from keras.constraints import maxnorm
from keras import optimizers
from keras.optimizers import Adam, SGD, RMSprop
from keras import backend as K
K.set_image_dim_ordering('tf')
from keras.callbacks import ModelCheckpoint,EarlyStopping,TensorBoard,ReduceLROnPlateau
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from mlxtend.plotting import plot_confusion_matrix
import tensorflow as tf
from IPython.display import display
import seaborn as sns
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt



# Creates directory, if directory exists removes if remove parameter is set to True
def create_directory(directory_path, remove=False):
    if remove and os.path.exists(directory_path):
        try:
            shutil.rmtree(directory_path)
            os.mkdir(directory_path)
        except OSError:
            print("Could not remove directory : ", directory_path)
            return False
    else:
        try:
            os.mkdir(directory_path)
        except OSError:
            print("Could not create directory: ", directory_path)
            return False

    return True


# Removes directory, if directory exists
def remove_directory(directory_path):
    if os.path.exists(directory_path):
        try:
            shutil.rmtree(directory_path)
        except OSError:
            print("Could not remove directory : ", directory_path)
            return False

    return True


def clear_directory(directory_path):
    dirs_files = os.listdir(directory_path)

    for item in dirs_files:
        #         item_path = os.path.join(directory_path, item)
        item_path = directory_path + item

        try:
            if os.path.isfile(item_path):
                os.unlink(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
        except Exception as e:
            print(e)

    return True


def remove_empty_folders(path, remove_root=True):
    if not os.path.isdir(path):
        return

    # remove empty subfolders
    files = os.listdir(path)

    if len(files):
        for f in files:
            fullpath = os.path.join(path, f)

            if os.path.isdir(fullpath):
                remove_empty_folders(fullpath)

    # if folder empty, delete it
    files = os.listdir(path)

    if len(files) == 0 and remove_root:
        print("Removing empty folder:", path)
        os.rmdir(path)


def dir_file_count(directory):
    return sum([len(files) for r, d, files in os.walk(directory)])


# print date and time for given type of representation
def date_time(x):
    if x == 1:
        return 'Timestamp: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now())
    if x == 2:
        return 'Timestamp: {:%Y-%b-%d %H:%M:%S}'.format(datetime.datetime.now())
    if x == 3:
        return 'Date now: %s' % datetime.datetime.now()
    if x == 4:
        return 'Date today: %s' % datetime.date.today()


# prints a integer for degugging
def debug(x):
    print("-"*40, x, "-"*40)


# Removes everything except alphabetical and selected characters from name string
def name_correct(name):
    return re.sub(r'[^a-zA-Z,:]', ' ', name).title()


def get_reset_subplot_params(n_rows, n_cols, dpi):
    subplot_params = dict()
    subplot_params["n_rows"] = n_rows
    subplot_params["n_cols"] = n_cols

    subplot_params["fig_size_col"] = subplot_params["n_cols"] * 2.5
    subplot_params["fig_size_row"] = subplot_params["n_rows"] * 2.5
    subplot_params["dpi"] = dpi
    subplot_params["face_color"] = 'w'
    subplot_params["edge_color"] = 'k'
    subplot_params["subplot_kw"] = {'xticks': [], 'yticks': []}
    subplot_params["axes.title_size"] = 'small'
    subplot_params["h_space"] = 0.5
    subplot_params["w_space"] = 0.3

    return subplot_params


def get_reset_plot_params(fig_size=(15,15), title="", x_label="", y_label="", legends=[],
                          title_font_size=18, label_font_size=14, image_file_name="", save=False,
                          dpi=100, update_image=True):
    plot_params = dict()

    plot_params["fig_size"] = fig_size
    plot_params["title"] = title
    plot_params["x_label"] = x_label
    plot_params["y_label"] = y_label
    plot_params["legends"] = legends
    plot_params["title_font_size"] = title_font_size
    plot_params["axes.title_size"] = "small"
    plot_params["label_font_size"] = label_font_size
    plot_params["image_file_name"] = image_file_name
    plot_params["save"] = save
    plot_params["update_image"] = update_image
    plot_params["subplot"] = None
    return plot_params


# building list of all the file paths in the directory according to the class
def select_image_by_category(image_dir, image_count_per_category):
    classes = os.listdir(image_dir)
    # figure out what is image_count_per_category!!!!!!!!
    # In the directory each folder contain one type of class, and the name of the folder is the name of the class.
    class_count = len(classes)
    image_file_paths = dict()

    for i in range(class_count):
        subdir_path = image_dir + "/" + classes[i]
        subdir_files = os.listdir(subdir_path)
        # Return the names of all the files in the folder of one class
        subdir_file_count = len(subdir_files)
        subdir_file_mem = dict()

        subdir_file_index = -1

        image_file_paths[classes[i]] = []

        for j in range(image_count_per_category):
            while subdir_file_index in subdir_file_mem:
                subdir_file_index = random.randint(0, subdir_file_count-1)
                # Getting random index from the class folder
                # the purpose of the  while loop is to chek that i will not repeat on index.

            subdir_file_mem[subdir_file_index] = 1
            # at the first loop subdir_file_mem is empty and because of that
            # subdir_file_index stays = -1

            subdir_file_name = subdir_files[subdir_file_index]
            subdir_file_path = subdir_path + "/" + subdir_file_name
            image_file_paths[classes[i]].append(subdir_file_path)

    return image_file_paths


def get_fig_axs(subplot_params):
    fig, axs = plt.subplots(
        nrows=subplot_params["n_rows"], ncols=subplot_params["n_cols"],
        figsize=(subplot_params["fig_size_col"], subplot_params["fig_size_row"]),
        dpi=subplot_params["dpi"], facecolor=subplot_params["face_color"],
        edgecolor=subplot_params["edge_color"], subplot_kw=subplot_params["subplot_kw"])

    return fig, axs


def plot_sample_image(image_file_paths, plot_params, subplot_params, update_image=True):
    fig, axs = get_fig_axs(subplot_params)

    plt.rcParams.update({'axes.titlesize': plot_params["axes.title_size"]})
    plt.subplots_adjust(hspace=subplot_params["h_space"], wspace=subplot_params["w_space"])

    i = 0
    for img_file_path in image_file_paths:
        img = cv2.imread(img_file_path, 1)
        plt.title(img_file_path.split("/")[-1])
        plt.subplot(subplot_params["n_rows"], subplot_params["n_cols"], i + 1)
        plt.imshow(img)

        plt.xticks([])
        plt.yticks([])

        i = i + 1

    if plot_params["update_image"] and os.path.exists(plot_params["image_file_name"]):
        os.remove(plot_params["image_file_name"])
    if plot_params["save"]:
        fig.savefig(plot_params["image_file_name"], dpi=plot_params["dpi"])

    plt.tight_layout()
    plt.show()


def show_class_sample_images(directory, image_count_per_category=5, save=False, dpi=100, update_image=False):
    class_count = len(os.listdir(directory))
    print("Number of Class: ", class_count)
    sample_img_by_class = select_image_by_category(directory, image_count_per_category)
    for class_name in sample_img_by_class:
        plot_params = get_reset_plot_params(image_file_name="img.png", save = save, dpi=dpi, update_image=update_image)
        subplot_params = get_reset_subplot_params(n_rows=1, n_cols=image_count_per_category, dpi=dpi)
        print("%s%s%s"%("-"*55, name_correct(class_name), "-"*55))
        plot_sample_image(sample_img_by_class[class_name], plot_params, subplot_params)
        print("")
    print("%s%s%d%s" % ("-"*55, "All Class Printed:", class_count, "-"*55))


# count number of files on each subdirectory of a directory
# Saving the names of the subdirectory in subdirectory_names
def subdirectory_file_count(master_directory):
    subdirectories = os.listdir(master_directory)
    subdirectory_count = len(subdirectories)

    subdirectory_names = []
    subdirectory_file_counts = []

    for subdirectory in subdirectories:
        current_directory = os.path.join(master_directory, subdirectory)
        file_count = len(os.listdir(current_directory))
        subdirectory_names.append(subdirectory)
        subdirectory_file_counts.append(file_count)
    return subdirectory_names, subdirectory_file_counts


# show barplot
def bar_plot(x, y, plot_property):
    if plot_property['subplot']:
        plt.subplot(plot_property['subplot'])
    sns.barplot(x=x, y=y)
    plt.title(plot_property['title'], fontsize=plot_property['title_font_size'])
    plt.xlabel(plot_property['x_label'], fontsize=plot_property['label_font_size'])
    plt.ylabel(plot_property['y_label'], fontsize=plot_property['label_font_size'])
    plt.xticks(range(len(x)), x)


# show bar plot for count of labels in subdirectory of a directory
def count_bar_plot(master_directory, plot_property):
    subdirectory_names, subdirectory_file_counts = subdirectory_file_count(master_directory)
    x = [name_correct(i) for i in subdirectory_names]
    # x = dir_name
    y = subdirectory_file_counts
    bar_plot(x, y, plot_property)


# show bar plot for count of labels in subdirectory of a training, validation, testing directory
def show_train_val_test(training_dir, validation_dir, testing_dir, plot_property):
    plt.figure(figsize=plot_property['fig_size'])

    title = plot_property['title']
    plot_property['title'] = title + " (Training)"
    subplot_no = plot_property['subplot']

    count_bar_plot(training_dir, plot_property)

    plot_property['title'] = title + " (Validation)"
    plot_property['subplot'] = subplot_no + 1
    count_bar_plot(validation_dir, plot_property)

    plot_property['title'] = title + " (Testing)"
    plot_property['subplot'] = subplot_no + 2
    count_bar_plot(testing_dir, plot_property)

    plt.show()


# reset tensorflow graph tp free up memory and resource allocation
def reset_graph(model=None):
    if model:
        try:
            del model
        except IOError:
            return False

    tf.reset_default_graph()
    K.clear_session()
    gc.collect()
    return True


# reset callbacks
def reset_callbacks(checkpoint=None, reduce_lr=None, early_stopping=None, tensor_board=None):
    checkpoint = None
    reduce_lr = None
    early_stopping = None
    tensor_board = None


# PREPROCESSING

reset_graph()
reset_callbacks()

# Configure input/ output directory
# Configure training, validation, testing directory

input_directory = r"data3/input/"
output_directory = r"data3/output/"

training_dir = input_directory + r"train"
validation_dir = input_directory + r"val"
testing_dir = input_directory + r"test"

figure_directory = r"data3/output/figures"
figure_directory = "data3/output/figures"

if not os.path.exists(figure_directory):
    os.mkdir(figure_directory)

file_name_prediction_batch = figure_directory+r"/result"
file_name_prediction_sample = figure_directory+r"/sample"

show_class_sample_images(training_dir, image_count_per_category=5, save=False, dpi=100, update_image=False)


plot_params = get_reset_plot_params()

plot_params['fig_size'] = (18,4)

plot_params['title_font_size'] = 13
plot_params['label_font_size'] = 10

plot_params['title'] = "Number of Cases"

plot_params['subplot'] = 131

show_train_val_test(training_dir, validation_dir, testing_dir, plot_params)
classes = os.listdir(training_dir)
classes = [name_correct(i) for i in classes]


# Image Preprocessing/Augmetntation/Tranformation for training, validtation,
# testing and Dataset

# batch size = 32
# target_size = (299,299)
# color mode = 'rgb'

rescale = 1./255
# comment 1
target_size = (150, 150)
# comment 2
batch_size = 163
class_mode = "categorical"
# class_mode = "binary"


# Generate batches of tensor image data with real-time data augmentation.
# The data will be looped over (in batches).
# I need to see where I will do the contrast to the image.
train_data_gen = ImageDataGenerator(
    #comment 3
    featurewise_center=True,
    #samplewise_center=True,
    featurewise_std_normalization=True,
    #samplewise_std_normalization=True,
    rescale=rescale,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

# from the class ImageDataGenerator class:
# flow_from_directory method:
# Takes the path to a directory & generates batches of augmented data.
# return A DirectoryIterator yielding tuples of (x, y)
# where x is a numpy array containing a batch of images with shape
# (batch_size, *target_size, channels)
# and y is a numpy array of corresponding labels.
train_generator = train_data_gen.flow_from_directory(
    training_dir,
    target_size=target_size,
    class_mode=class_mode,
    batch_size=batch_size,
    shuffle=True)
# צריך למרכז נתונים\לנרמל אותם בהתאם להתפלגות של מדגם המבחן!!!!
validation_data_gen = ImageDataGenerator(rescale=rescale)

validation_generator = validation_data_gen.flow_from_directory(
    validation_dir,
    target_size=target_size,
    class_mode=class_mode,
    # I using all the image in the directory for the batch value.
    batch_size=dir_file_count(validation_dir),
    shuffle=False)
# צריך למרכז נתונים\לנרמל אותם בהתאם להתפלגות של מדגם המבחן!!!!
test_data_gen = ImageDataGenerator(rescale=rescale)

# flow_from_directory Takes the path to a directory and generates batches of augmented data

test_generator = test_data_gen.flow_from_directory(
    testing_dir,
    target_size=target_size,
    class_mode=class_mode,
    batch_size=dir_file_count(testing_dir),
    shuffle=False)


from sklearn.utils import class_weight
# Estimate class weights for unbalanced datasets.
#return Array with class_weight_vect[i] the weight for i-th class


def get_weight(y):
    class_weight_current = class_weight.compute_class_weight(
        'balanced', np.unique(y), y)
    return class_weight_current

# TRAINING FILES CONFIGURATION


class_weight = get_weight(train_generator.classes)


print(class_weight)


main_model_dir = output_directory + r"models/"
main_log_dir = output_directory + r"logs/"

clear_directory(main_log_dir)
remove_empty_folders(main_model_dir, False)

model_dir = main_model_dir + time.strftime('%Y-%m-%d %H-%M-%S') + "/"
log_dir = main_log_dir + time.strftime('%Y-%m-%d %H-%M-%S')

create_directory(model_dir, remove=True)
create_directory(log_dir, remove=True)

model_file = model_dir + "{epoch:02d}-val_acc-{val_acc:.2f}-val_loss-{val_loss:.2f}.hdf5"

#Callbacks
#Abstract base class used to build new callbacks.
reset_graph()
reset_callbacks()

print("Setting Callbacls at", date_time(1))

# ModelCheckpoint save the model after every epoch
# comment
checkpoint = ModelCheckpoint(
    model_file,
    monitor='val_acc',
    save_best_only=True)

# EarlyStopping stop training when a monitored quantity has stopped improving
# comment
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    verbose=1,
    restore_best_weights=True)

# TensorBoard basic visualization.
# this callback writes a log for TensorBoard, which allows you to visualize
# dynamic graphs of your training and test metrics, as well as activation
# histograms for the different layers in my model
tensor_board = TensorBoard(
    log_dir=log_dir,
    batch_size=batch_size,
    update_freq='batch')

# ReduceLROnPlateau reduce learning rate when metric has stopped improving
# comment
reduce_lr = ReduceLROnPlateau(
    monitor='val_los',
    patience=5,
    cooldown=2,
    min_lr=0.0000000001,
    verbose=1)

callbacks = [checkpoint, reduce_lr, early_stopping, tensor_board]
print("Set Callbacks at " , date_time(1))


#LOAD AND CONFIGURE MODEL InceptionV3

def get_model():
    # configure 1
    # base_model = InceptionV3(weights=None, include_top=False)
    #THE SCALING DOES'T MATCH THE SIZE OF THE INPUT TO THE MODEL!!!!!!
    base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(150, 150, 3), classes=2)
    #comment
    x = base_model.output
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)
    predictions = Dense(2, activation='softmax')(x)
    model = Model(input=base_model.input, outputs=predictions)

    for layer in base_model.layers:
        layer.trainable=False

    model.summary()

    return model

# Training/Fine-Tuning Base Model-InceptionV3 for Fine-Tuning
# with New Class Label


print("Getting Base Model",date_time(1))
model = get_model()


print("Starting Trainning Model", date_time(1))

step_per_epoch = len(train_generator)
validation_steps = len(validation_generator)

optimizer = optimizers.Adam()
loss = 'categorical_crossentropy'
metrics = ['accuracy']
epochs = 5

model.compile(optimizer, loss=loss, metrics=metrics)
#comment
history = model.fit_generator(
    train_generator,
    steps_per_epoch=step_per_epoch,
    epochs=epochs,
    verbose=2,
    callbacks=callbacks,
    validation_data=validation_generator,
    validation_steps=validation_steps,
    class_weight=class_weight)

# class weight:Optional dictionary mapping class indices (integers)
# to a weight (float) value, used for weighting the loss function
# (during training only). This can be useful to tell the model
# to "pay more attention" to samples from an under-represented class.

print("Completed Model Trainning", date_time(1))

# Model performance visualization over the epochs

xlabel = 'Epoch'
legends = ['Trainig', 'Validation']

ylim_pad = [0.01,0.1]

plt.figure(figsize=(15,15))

# plot training & validation Accuracy values

# history = model.fit_generator
y1 = history.history['acc']
y2 = history.history['val_acc']

min_y = min(min(y1), min(y2)) - ylim_pad[0]
max_y = max(max(y1), max(y2)) - ylim_pad[0]

plt.subplot(121)

plt.plot(y1)
plt.plot(y2)

plt.title('Model Accuracy', fontsize=17)
plt.xlabel(xlabel, fontsize=15)
plt.ylabel('Accuracy', fontsize=15)
plt.ylim(min_y, max_y)
plt.legend(legends, loc='upper left')
plt.grid()

plt.show()

# TEST SAVED MODELS

dir_name = r"data3/output/models/"
# every time, when I running  the code, it will save me a new model in dir_name
dirs = os.listdir(dir_name)
for i in range(len(dirs)):
    print(i, dirs[i])

#בחרתי את המודל הראשון בתיקיה
cur_dir = dir_name + dirs[0] + "/"
model_names = os.listdir(cur_dir)
# printing the model name after few epochs
for i in range(len(model_names)):
    print(i, model_names[i])

model_file = cur_dir + model_names[0]

model = keras.models.load_model(model_file)

print ('results')
# test_generator is the uploader  of the test image.
# evaluate_generator evakuate the model on a data generator
result = model.evaluate_generator(test_generator, steps=len(test_generator), verbose=1)

print("%s%.2f  "% ("Loss     : ", result[0]))
print("%s%.2f%s"% ("Accuracy : ", result[1]*100, "%"))

print('results')
y_pred = model.predict_generator(test_generator, steps=len(test_generator), verbose=1)
y_pred = y_pred.argmax(axis=1)
y_true = test_generator.classes

image_file_name_CM = figure_directory + "/CM"

title = model_file.split("/")

model_title = "/".join([i for i in title[3:]])

precision = precision_score(y_true, y_pred)
recall = recall_score(y_true,y_pred)
f1 = f1_score(y_true,y_pred)

print("-"*90)
print("Derived Report")
print("-"*90)
print("%s%.2f%s"% ("Precision     : ", precision*100, "%"))
print("%s%.2f%s"% ("Recall        : ", recall*100,    "%"))
print("%s%.2f%s"% ("F1-Score      : ", f1*100,        "%"))
print("-"*90)
print("\n\n")

CM = confusion_matrix(y_true,y_pred)

fig, ax = plot_confusion_matrix(conf_mat=CM, figsize=(10, 8), hide_ticks=True, cmap=plt.cm.Blues)
plt.xticks(range(len(classes)), classes, fontsize=12)
plt.yticks(range(len(classes)), classes, fontsize=12)
plt.title("Confusion Matrix for Model File (Test Dataset): \n" + model_title, fontsize=11)
fig.savefig(image_file_name_CM, dpi=100)
plt.show()

cls_report_print = classification_report(y_true, y_pred, target_names=classes)

cls_report = classification_report(y_true, y_pred, target_names=classes, output_dict=True)

print("\n\n")
print("-" * 90)
print("Report for Model File: ", model_title)
print("-" * 90)
print(cls_report_print)
print("-" * 90)



