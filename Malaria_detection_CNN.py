
import os
import glob
import numpy as np
import pandas as pd
import cv2
from concurrent import futures
import threading
from sklearn.model_selection import train_test_split
from collections import Counter

print('1')
# In[2]:


# Data loading
root = '/media/song/新加卷/Malaria_detection_ANN/Data'
# reading dataset

base_dir = os.path.join(root)
cell_dir = os.path.join(base_dir, 'cell_images')

parasitized_dir = os.path.join(cell_dir, 'Parasitized')
uninfected_dir = os.path.join(cell_dir, 'Uninfected')

parasitized_files = glob.glob(parasitized_dir+'/*.png')
uninfected_files = glob.glob(uninfected_dir+'/*.png')

print(len(parasitized_files), len(uninfected_files))


# In[3]:
print('2')

# checking top 5 rows
np.random.seed(42)
files_df = pd.DataFrame({
    'filename': parasitized_files+ uninfected_files,
    'label': ['Parasitized']* len(parasitized_files) + ['Uninfected'] * len(uninfected_files)
}).sample(frac=1, random_state=42).reset_index(drop=True)

print(files_df.head())


# In[4]:


print('3')
# Data Preprocessing

# Data Split, split the dataset into training(60),testing(30),validation(10)
train_files, test_files, train_labels, test_labels = train_test_split(files_df['filename'].values,
                                                                      files_df['label'].values,
                                                                      test_size=0.3, random_state=42)
train_files, val_files, train_labels, val_labels = train_test_split(train_files,
                                                                    train_labels,
                                                                    test_size=0.1, random_state=42)

print(train_files.shape, val_files.shape, test_files.shape)
print('Train:', Counter(train_labels), '\nVal:', Counter(val_labels), '\nTest:', Counter(test_labels))


# In[5]:

print('4')
# Check image Dimensions
def get_img_shape_parallel(idx, img, total_imgs):
    if idx % 5000 == 0 or idx == (total_imgs - 1):
        print('{}: working on img num: {}'.format(threading.current_thread().name,
                                                  idx))
    return cv2.imread(img).shape


ex = futures.ThreadPoolExecutor(max_workers=None)
data_inp = [(idx, img, len(train_files)) for idx, img in enumerate(train_files)]
print('Starting Img shape computation:')
train_img_dims_map = ex.map(get_img_shape_parallel,
                            [record[0] for record in data_inp],
                            [record[1] for record in data_inp],
                            [record[2] for record in data_inp])
train_img_dims = list(train_img_dims_map)
print('Min Dimensions:', np.min(train_img_dims, axis=0))
print('Avg Dimensions:', np.mean(train_img_dims, axis=0))
print('Median Dimensions:', np.median(train_img_dims, axis=0))
print('Max Dimensions:', np.max(train_img_dims, axis=0))


# In[6]:


print('5')
# Watershed segmentation
def segmentation(image):
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Performing Otsu's Binarization : This means that if the value of the pixel exceeds the threshold, it would be considered as 1. Else, 0
    ret, thresh = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # print("Threshold limit: " + str(ret))

    # Specifying the Background and Foreground after Noise Removal
    # noise removal
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # Performing Distance Transfrom : In distance transfrom, the gray level intensities of the points inside the foreground
    # are changed to distance their respective distances from the closest 0 value

    # sure foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Connected Components
    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    # Applying Watershed Segmentation
    markers = cv2.watershed(image, markers)
    image[markers == -1] = [0, 255, 0]
    return image


# In[ ]:

print('6')

# Image Resizing and Watershed Segmentation
IMG_DIMS = (70,70)
INPUT_SHAPE = (70,70, 3)

def get_img_data_parallel(idx, img, total_imgs):
    if idx % 5000 == 0 or idx == (total_imgs - 1):
        print('{}: working on img num: {}'.format(threading.current_thread().name,
                                                  idx))
    img = cv2.imread(img)
    #img = cv2.bilateralFilter(img, 15, 75, 75)
    img = cv2.resize(img, dsize=IMG_DIMS, interpolation=cv2.INTER_CUBIC)
    img = segmentation(img)
    img = np.array(img, dtype=np.float32)
    return img

ex = futures.ThreadPoolExecutor(max_workers=None)
train_data_inp = [(idx, img, len(train_files)) for idx, img in enumerate(train_files)]
val_data_inp = [(idx, img, len(val_files)) for idx, img in enumerate(val_files)]
test_data_inp = [(idx, img, len(test_files)) for idx, img in enumerate(test_files)]

print('Loading Train Images:')
train_data_map = ex.map(get_img_data_parallel,
                        [record[0] for record in train_data_inp],
                        [record[1] for record in train_data_inp],
                        [record[2] for record in train_data_inp])
train_data = np.array(list(train_data_map))

print('\nLoading Validation Images:')
val_data_map = ex.map(get_img_data_parallel,
                        [record[0] for record in val_data_inp],
                        [record[1] for record in val_data_inp],
                        [record[2] for record in val_data_inp])
val_data = np.array(list(val_data_map))

print('\nLoading Test Images:')
test_data_map = ex.map(get_img_data_parallel,
                        [record[0] for record in test_data_inp],
                        [record[1] for record in test_data_inp],
                        [record[2] for record in test_data_inp])
test_data = np.array(list(test_data_map))

print(train_data.shape, val_data.shape, test_data.shape)


# In[ ]:

print('7')

import matplotlib.pyplot as plt
# %matplotlib inline

plt.figure(1 , figsize = (8 , 8))
n = 0
for i in range(16):
    n += 1
    r = np.random.randint(0 , train_data.shape[0] , 1)
    plt.subplot(4 , 4 , n)
    plt.subplots_adjust(hspace = 0.5 , wspace = 0.5)
    plt.imshow(train_data[r[0]]/255.)
    plt.title('{}'.format(train_labels[r[0]]))
    plt.xticks([]) , plt.yticks([])


# In[ ]:

print('8')
# Normalization of data
train_imgs_scaled = train_data / 255.
val_imgs_scaled = val_data / 255.
test_imgs_scaled = test_data / 255.

# encode text category labels
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(train_labels)
train_labels_t = le.transform(train_labels)
val_labels_t = le.transform(val_labels)
test_labels_t = le.transform(test_labels)

# change version of tensorflow
# %get_ipython().run_line_magic('tensorflow_version', '2.x')
# Parasitized = 0 ,  Uninfected =1

print(train_labels[:6], train_labels_t[:6])


# In[ ]:

print('9')
import tensorflow as tf

# Load the TensorBoard notebook extension (optional)
# %get_ipython().run_line_magic('load_ext', 'tensorboard')

#tf.random.set_random_seed(42)
np.random.seed(42)
print(tf.__version__)


# In[ ]:


print('10')
#Applying Data augmentation to images
BATCH_SIZE = 8
NUM_CLASSES = 2
EPOCHS = 10
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255,
                                                                zoom_range=0.05,
                                                                rotation_range=25,

                                                                shear_range=0.05, horizontal_flip=True,
                                                                fill_mode='nearest')

val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
# check augmented images
img_id = 0
sample_generator = train_datagen.flow(train_data[img_id:img_id+1], train_labels[img_id:img_id+1],
                                      batch_size=1)
sample = [next(sample_generator) for i in range(0,5)]

fig, ax = plt.subplots(1,5, figsize=(16, 6))
print('Labels:', [item[1][0] for item in sample])
l = [ax[i].imshow(sample[i][0][0]) for i in range(0,5)]

# build image augmentation generators
train_generator = train_datagen.flow(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_generator = val_datagen.flow(val_data, batch_size=BATCH_SIZE, shuffle=False)


# In[ ]:
print('11')

# from keras.optimizers import Adam
#Fully Connected Neural Network

# Define the network model and its arguments.
# Set the number of neurons/nodes for each layer:

inp = tf.keras.layers.Input(shape=INPUT_SHAPE)

conv1 = tf.keras.layers.Conv2D(32, kernel_size=(3, 3),
                               activation='relu', padding='same')(inp)
pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)
conv2 = tf.keras.layers.Conv2D(64, kernel_size=(3, 3),
                               activation='relu', padding='same')(pool1)
pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)
conv3 = tf.keras.layers.Conv2D(64, kernel_size=(3, 3),
                               activation='relu', padding='same')(pool2)
pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)

flat = tf.keras.layers.Flatten()(pool3)

hidden1 = tf.keras.layers.Dense(256, activation='relu')(flat)
drop1 = tf.keras.layers.Dropout(rate=0.3)(hidden1)
hidden2 = tf.keras.layers.Dense(128, activation='relu')(drop1)
drop2 = tf.keras.layers.Dropout(rate=0.3)(hidden2)
# we will use Dense(1, activation='sigmoid') for binary and Dense(4, activation='softmax') for multi-classificaiton
out = tf.keras.layers.Dense(1, activation='sigmoid')(drop2)

model = tf.keras.Model(inputs=inp, outputs=out)
# we'll use loss='binary_crossentropy' for binry classificaiton and loss='categorical_crossentropy' for multi-classification
# model.compile(optimizer='SGD', loss='categorical_crossentropy', metrics=['accuracy'])
# we can use 'adam',  'SGD', 'RMSProp'
model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy'])
model.summary()



print('12')
# In[ ]:


import datetime
logdir = os.path.join('/media/song/新加卷/Malaria_detection_ANN/Model', datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                              patience=2, min_lr=0.000001)

callbacks = [reduce_lr, tensorboard_callback]
train_steps_per_epoch = train_generator.n // train_generator.batch_size
val_steps_per_epoch = val_generator.n // val_generator.batch_size
# history = model.fit_generator(train_generator, steps_per_epoch=train_steps_per_epoch, epochs=EPOCHS,
#                               validation_data=val_generator, validation_steps=val_steps_per_epoch,
#                               verbose=1)
# history = model.fit_generator(train_generator, steps_per_epoch=train_steps_per_epoch, epochs=EPOCHS, validation_data=val_generator, validation_steps=val_steps_per_epoch)
history = model.fit(train_imgs_scaled, train_labels_t, validation_data=(val_imgs_scaled, val_labels_t), epochs=15, batch_size=8 )
model.save('malaria_detection_CNN_model_1.h5')
print('13')
# In[ ]:
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
t = f.suptitle('SqueezeNet model', fontsize=12)
f.subplots_adjust(top=0.85, wspace=0.3)

max_epoch = len(history.history['accuracy'])+1
epoch_list = list(range(1,max_epoch))
ax1.plot(epoch_list, history.history['accuracy'], label='Train Accuracy')
ax1.plot(epoch_list, history.history['val_accuracy'], label='Validation Accuracy')
ax1.set_xticks(np.arange(1, max_epoch, 5))
ax1.set_ylabel('Accuracy Value')
ax1.set_xlabel('Epoch')
ax1.set_title('Accuracy')
l1 = ax1.legend(loc="best")

ax2.plot(epoch_list, history.history['loss'], label='Train Loss')
ax2.plot(epoch_list, history.history['val_loss'], label='Validation Loss')
ax2.set_xticks(np.arange(1, max_epoch, 5))
ax2.set_ylabel('Loss Value')
ax2.set_xlabel('Epoch')
ax2.set_title('Loss')
l2 = ax2.legend(loc="best")

# In[ ]:

model.save('./Model/KidneyStoneDetector_squeezenet.h5')

print('15')
# In[ ]:


#scale test data
test_imgs_scaled.shape, test_labels_t.shape


# In[ ]:

print('16')
# Model evaluated on test data
score = model.evaluate(test_imgs_scaled, test_labels_t, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

print('17')

