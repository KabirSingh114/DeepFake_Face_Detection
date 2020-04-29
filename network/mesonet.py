import numpy as np
from keras import backend as K

from keras.layers import Input, Dense, Flatten, GlobalAveragePooling2D, Activation, Conv2D, MaxPooling2D, BatchNormalization, Lambda, Dropout, MaxPool2D
from keras.layers import SeparableConv2D, Add
from keras.models import Model


def mesonet(img_height=64, img_width=64, dropout_rate=0.2, include_top=True):
    #K.set_image_dim_ordering('th')


    img_input = Input(shape=(img_height, img_width, 3))

    x = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(img_input)
    x = BatchNormalization()(x)
    #x = MaxPool2D(pool_size=(2,2))(x)
    
    x = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPool2D(pool_size=(2,2))(x)
    
    x = Conv2D(filters=128, kernel_size=(5, 5), activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPool2D(pool_size=(2,2))(x)
    
    x = Conv2D(filters=256, kernel_size=(5, 5), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPool2D(pool_size=(2,2))(x)

    
    #x = Dense(1024, activation='relu')(x)
    #x = Dense(16, activation='relu')(x)
    
    if include_top == True:
        x = GlobalAveragePooling2D()(x)
        x = Dense(units=2, activation='softmax')(x)
        #x = Dropout(dropout_rate)(x)

    model = Model(img_input, x)
    return model

