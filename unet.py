import numpy as np
import tensorflow as tf
from tensorflow.python.keras.models     import Model
from tensorflow.python.keras.layers     import (Input,
                                                Conv2D,
                                                MaxPooling2D,
                                                UpSampling2D,
                                                concatenate,
                                                Conv2DTranspose,
                                                BatchNormalization,
                                                Dropout)
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.callbacks  import (CSVLogger,
                                                ModelCheckpoint,
                                                EarlyStopping,
                                                ReduceLROnPlateau,
                                                TensorBoard)

from patches import DEFAULT_PATCH_SIZE


"""
U-Net model (https://arxiv.org/abs/1505.04597)
"""
def unet_model(nb_classes=5, img_size=DEFAULT_PATCH_SIZE, nb_channels=8, nb_filters_start=32,
              growth_factor=2, upconv=True, class_weights=[0.2, 0.3, 0.1, 0.1, 0.3],
              droprate=0.25):
    nb_filters = nb_filters_start
    inputs = Input(shape=(img_size, img_size, nb_channels))
    conv1 = Conv2D(nb_filters, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(nb_filters, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    nb_filters *= growth_factor
    pool1 = BatchNormalization()(pool1)
    conv2 = Conv2D(nb_filters, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(nb_filters, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    pool2 = Dropout(droprate)(pool2)

    nb_filters *= growth_factor
    pool2 = BatchNormalization()(pool2)
    conv3 = Conv2D(nb_filters, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(nb_filters, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    pool3 = Dropout(droprate)(pool3)

    nb_filters *= growth_factor
    pool3 = BatchNormalization()(pool3)
    conv4_0 = Conv2D(nb_filters, (3, 3), activation='relu', padding='same')(pool3)
    conv4_0 = Conv2D(nb_filters, (3, 3), activation='relu', padding='same')(conv4_0)
    pool4_1 = MaxPooling2D(pool_size=(2, 2))(conv4_0)
    pool4_1 = Dropout(droprate)(pool4_1)

    nb_filters *= growth_factor
    pool4_1 = BatchNormalization()(pool4_1)
    conv4_1 = Conv2D(nb_filters, (3, 3), activation='relu', padding='same')(pool4_1)
    conv4_1 = Conv2D(nb_filters, (3, 3), activation='relu', padding='same')(conv4_1)
    pool4_2 = MaxPooling2D(pool_size=(2, 2))(conv4_1)
    pool4_2 = Dropout(droprate)(pool4_2)

    nb_filters *= growth_factor
    conv5 = Conv2D(nb_filters, (3, 3), activation='relu', padding='same')(pool4_2)
    conv5 = Conv2D(nb_filters, (3, 3), activation='relu', padding='same')(conv5)
    
    
    # this is the bottom part of "U"
        
    
    nb_filters //= growth_factor
    if upconv:
        up6_1 = concatenate([Conv2DTranspose(nb_filters, (2, 2), strides=(2, 2), padding='same')(conv5), conv4_1])
    else:
        up6_1 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4_1])
        
    up6_1 = BatchNormalization()(up6_1)
    conv6_1 = Conv2D(nb_filters, (3, 3), activation='relu', padding='same')(up6_1)
    conv6_1 = Conv2D(nb_filters, (3, 3), activation='relu', padding='same')(conv6_1)
    conv6_1 = Dropout(droprate)(conv6_1)
    
    nb_filters //= growth_factor
    if upconv:
        up6_2 = concatenate([Conv2DTranspose(nb_filters, (2, 2), strides=(2, 2), padding='same')(conv6_1), conv4_0])
    else:
        up6_2 = concatenate([UpSampling2D(size=(2, 2))(conv6_1), conv4_0])
    up6_2 = BatchNormalization()(up6_2)
    conv6_2 = Conv2D(nb_filters, (3, 3), activation='relu', padding='same')(up6_2)
    conv6_2 = Conv2D(nb_filters, (3, 3), activation='relu', padding='same')(conv6_2)
    conv6_2 = Dropout(droprate)(conv6_2)
    
    nb_filters //= growth_factor
    if upconv:
        up7 = concatenate([Conv2DTranspose(nb_filters, (2, 2), strides=(2, 2), padding='same')(conv6_2), conv3])
    else:
        up7 = concatenate([UpSampling2D(size=(2, 2))(conv6_2), conv3])
    up7 = BatchNormalization()(up7)
    conv7 = Conv2D(nb_filters, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(nb_filters, (3, 3), activation='relu', padding='same')(conv7)
    conv7 = Dropout(droprate)(conv7)
    
    nb_filters //= growth_factor
    if upconv:
        up8 = concatenate([Conv2DTranspose(nb_filters, (2, 2), strides=(2, 2), padding='same')(conv7), conv2])
    else:
        up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2])
    up8 = BatchNormalization()(up8)
    conv8 = Conv2D(nb_filters, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(nb_filters, (3, 3), activation='relu', padding='same')(conv8)
    conv8 = Dropout(droprate)(conv8)

    nb_filters //= growth_factor
    if upconv:
        up9 = concatenate([Conv2DTranspose(nb_filters, (2, 2), strides=(2, 2), padding='same')(conv8), conv1])
    else:
        up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1])
    conv9 = Conv2D(nb_filters, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(nb_filters, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(nb_classes, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=inputs, outputs=conv10)
    
    # define loss function
    def weighted_binary_crossentropy(y_true, y_pred):
        class_loglosses = tf.reduce_mean(tf.keras.backend.binary_crossentropy(y_true, y_pred), axis=[0, 1, 2])
        return tf.reduce_sum(class_loglosses * tf.constant(class_weights))
    
    model.compile(optimizer=Adam(), loss=weighted_binary_crossentropy)
    
    print(model.summary())

    return model