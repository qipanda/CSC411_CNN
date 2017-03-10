'''This script goes along the blog post
"Building powerful image classification models using very little data"
from blog.keras.io.

It uses data that can be downloaded at:
https://www.kaggle.com/c/dogs-vs-cats/data

In our setup, we:
- created a data/ folder
- created train/ and validation/ subfolders inside data/
- created cats/ and dogs/ subfolders inside train/ and validation/
- put the cat pictures index 0-999 in data/train/cats
- put the cat pictures index 1000-1400 in data/validation/cats
- put the dogs pictures index 12500-13499 in data/train/dogs
- put the dog pictures index 13500-13900 in data/validation/dogs

So that we have 1000 training examples for each class, and 400 validation examples for each class.

In summary, this is our directory structure:
```
data/
    train/
        dogs/
            dog001.jpg
            dog002.jpg
            ...
        cats/
            cat001.jpg
            cat002.jpg
            ...
    validation/
        dogs/
            dog001.jpg
            dog002.jpg
            ...
        cats/
            cat001.jpg
            cat002.jpg
            ...
```
'''
import os
import h5py
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as bk
from keras.regularizers import l2, activity_l2
import glob
import scipy as sp
import matplotlib.pyplot as plt

bk.set_image_dim_ordering('th')

# path to the model weights file.
weights_path = 'vgg16_weights.h5'
top_model_weights_path = 'bottleneck_fc_model.h5'
# dimensions of our images.
img_width, img_height = 128, 128

train_data_dir = 'data/train'
validation_data_dir = 'data/validation'
test_data_dir = 'public_test'
private_data_dir = 'private_test'
nb_train_samples = 15560
nb_validation_samples = 3888
nb_test_samples = 970
nb_private_samples = 2000
nb_epoch = 10


def save_bottlebeck_features():
    datagen = ImageDataGenerator(rescale=1./255)

    # build the VGG16 network
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(3, img_width, img_height)))

    model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # load the weights of the VGG16 networks
    # (trained on ImageNet, won the ILSVRC competition in 2014)
    # note: when there is a complete match between your model definition
    # and your weight savefile, you can simply call model.load_weights(filename)
    assert os.path.exists(weights_path), 'Model weights not found (see "weights_path" variable in script).'
    f = h5py.File(weights_path)
    for k in range(f.attrs['nb_layers']):
        if k >= len(model.layers):
            # we don't look at the last (fully-connected) layers in the savefile
            break
        g = f['layer_{}'.format(k)]
        weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
        model.layers[k].set_weights(weights)
    f.close()
    print('Model loaded.')

    # generator = datagen.flow_from_directory(
    #         train_data_dir,
    #         target_size=(img_width, img_height),
    #         batch_size=32,
    #         class_mode=None,
    #         shuffle=False)
    # bottleneck_features_train = model.predict_generator(generator, nb_train_samples)
    # import pdb; pdb.set_trace()
    # np.save(open('bottleneck_features_train.npy', 'wb'), bottleneck_features_train)
    # print('train generater done')
    #
    # generator = datagen.flow_from_directory(
    #         validation_data_dir,
    #         target_size=(img_width, img_height),
    #         batch_size=32,
    #         class_mode=None,
    #         shuffle=False)
    # bottleneck_features_validation = model.predict_generator(generator, nb_validation_samples)
    # np.save(open('bottleneck_features_validation.npy', 'wb'), bottleneck_features_validation)
    # print('val generater done')
    #
    # generator = datagen.flow_from_directory(
    #         test_data_dir,
    #         target_size=(img_width, img_height),
    #         batch_size=32,
    #         class_mode=None,
    #         shuffle=False)
    # bottleneck_features_test = model.predict_generator(generator, nb_test_samples)
    # np.save(open('bottleneck_features_test.npy', 'wb'), bottleneck_features_test)
    # print('test generater done')

    generator = datagen.flow_from_directory(
            private_data_dir,
            target_size=(img_width, img_height),
            batch_size=32,
            class_mode=None,
            shuffle=False)
    bottleneck_features_private = model.predict_generator(generator, nb_private_samples)
    np.save(open('bottleneck_features_private.npy', 'wb'), bottleneck_features_private)
    print('private test generater done')

def train_top_model():
    import pdb; pdb.set_trace()
    train_data = np.load(open('bottleneck_features_train.npy', 'rb'))
    train_labels = np.identity(8, dtype=np.int64)
    train_labels = np.repeat(train_labels, nb_train_samples/8, axis=0)

    validation_data = np.load(open('bottleneck_features_validation.npy', 'rb'))
    validation_labels = np.identity(8, dtype=np.int64)
    validation_labels = np.repeat(validation_labels, nb_validation_samples/8, axis=0)

    test_data = np.load(open('bottleneck_features_test.npy', 'rb'))
    private_data = np.load(open('bottleneck_features_private.npy', 'rb'))
    import pdb; pdb.set_trace()
    #append public and private test sets
    full_test = np.append(test_data, private_data, axis=0)

    l2_lam = 0.0001
    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, W_regularizer=l2(l2_lam), activation='relu'))
    model.add(Dense(256, W_regularizer=l2(l2_lam), activation='relu'))
    model.add(Dense(256, W_regularizer=l2(l2_lam), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(8, W_regularizer=l2(l2_lam), activation='softmax'))

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    # model.load_weights(top_model_weights_path)
    history = model.fit(train_data, train_labels,
              nb_epoch=nb_epoch, batch_size=32,
              validation_data=(validation_data, validation_labels))
    import pdb; pdb.set_trace()
    #evaluate on public test set
    # test_images = np.array(
    #     insertImageToDataframe('public_test'), dtype='float32')

    # predictions = model.predict(test_data)
    # predictions = np.argmax(predictions, axis=1)+1
    # saveCSV(predictions)

    predictions = model.predict(full_test)
    predictions = np.argmax(predictions, axis=1)+1

    # plt.plot(history.history['loss'], 'b', label='train')
    # plt.plot(history.history['val_loss'], 'g', label='validation')
    # plt.xlabel('epoch')
    # plt.ylabel('cross entropy')
    # plt.title('Training CNN w ImageNet VGG16 + Fine Tuned NN on Augmented Images')
    # plt.show()
    #
    # plt.plot(history.history['acc'], 'b', label='train')
    # plt.plot(history.history['val_acc'], 'g', label='validation')
    # plt.xlabel('epoch')
    # plt.ylabel('accuracy')
    # plt.title('Training CNN w ImageNet VGG16 + Fine Tuned NN on Augmented Images')
    # plt.show()

    #save model weights
    # model.save_weights('bottleneck_fc_5model_catce_0001.h5')

    saveCSV_full(predictions)



def saveCSV(predictions):
    filler = np.zeros(2000)
    predictions = np.concatenate((predictions, filler))
    prediction_labels = np.arange(970+2000)+1
    prediction_outtxt = np.concatenate((prediction_labels.reshape(-1,1), predictions.reshape(-1,1)), axis=1)
    prediction_outtxt = prediction_outtxt.astype(int)
    np.savetxt('trial7.csv', prediction_outtxt, comments='',
        delimiter=',', header='Id, Prediction', fmt='%.i')

def saveCSV_full(predictions):
    prediction_labels = np.arange(970+2000)+1
    prediction_outtxt = np.concatenate((prediction_labels.reshape(-1,1), predictions.reshape(-1,1)), axis=1)
    prediction_outtxt = prediction_outtxt.astype(int)
    np.savetxt('cat_ce_3_full_test_0001.csv', prediction_outtxt, comments='',
        delimiter=',', header='Id, Prediction', fmt='%.i')

'''import all the images into a list of ndarrays'''
def insertImageToDataframe(image_folder):
    image_list = []
    for filename in glob.glob(image_folder + '/*.jpg'):
        im = sp.misc.imread(filename)
        image_list.append(im)

    return image_list


# save_bottlebeck_features()
train_top_model()
