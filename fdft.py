from utils import *
from fdft import att 
from fdft import block
from network import *
import os
import argparse
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import numpy as np
from keras.models import Input
from keras.layers import Input, Dense, Flatten, GlobalAveragePooling2D, Activation, Conv2D, MaxPooling2D, BatchNormalization, Lambda, Dropout
from keras.layers import SeparableConv2D, Add, Convolution2D, concatenate, Layer, ReLU, DepthwiseConv2D, Reshape, Multiply, InputSpec
from keras.regularizers import l2
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tqdm import tqdm, trange
from sklearn import metrics
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, classification_report
from scipy.optimize import brentq
from scipy.interpolate import interp1d

parser = argparse.ArgumentParser(description='Pretrain the models')
parser.add_argument('-network', required=True, type=str, help='select the backbone network')
parser.add_argument('-model', required=True, type=str, help='pretrained backbone model')
parser.add_argument('-ft_dir', required=True, type=str, help='fine tune image directory')
parser.add_argument('-val_dir', required=True, type=str, help='validation image directory')
parser.add_argument('-test_dir', required=False, type=str, help='fine tune image directory')
args = parser.parse_args()

img_height = 64
img_width = 64
es_patieance = 10
reduce_factor = .1

batch_size = 128

ft_dir = args.ft_dir
validation_dir = args.val_dir
train_gen_aug = ImageDataGenerator(shear_range=0,
                               zoom_range=0,
                               rotation_range=0.5,
                               width_shift_range=2.,
                               height_shift_range=2.,
                               horizontal_flip=True,
                               zca_whitening=False,
                               fill_mode='nearest',
                               preprocessing_function=cutout) 
"""
train_gen_aug = ImageDataGenerator(shear_range=0,
                               zoom_range=0,
                               rotation_range=0.5,
                               width_shift_range=2.,
                               height_shift_range=2.,
                               horizontal_flip=True,
                               zca_whitening=False,
                               fill_mode='nearest')
"""
test_datagen = ImageDataGenerator(rescale=1./255)

ft_gen = train_gen_aug.flow_from_directory(ft_dir, target_size=(img_height, img_width), batch_size=batch_size, shuffle=True, 
                                            class_mode='categorical')


validation_generator = test_datagen.flow_from_directory(validation_dir, target_size=(img_height, img_width), batch_size=batch_size, shuffle=False,
                                                        class_mode='categorical')

model_ft = tf.keras.models.load_model(args.model)
print(model_ft)

for i in range(2):
    model_ft.layers.pop()
im_in = Input(shape=(img_width, img_height, 3))

if args.network == 'resnetV2':
    base_model = resNetV2(img_width, img_height, include_top=False)
elif args.network == 'squeezenet':
    base_model = squeezeNet(img_width, img_height, .2, include_top=False)
elif args.network == 'vgg16':
    base_model = vgg16(img_width, img_height, include_top=False)
elif args.network == 'mesonet':
    base_model = mesonet(img_width, img_height, include_top=False)
elif args.network == 'denseNet':
    base_model = denseNet(img_width, img_height, include_top=False)
    
base_model.set_weights(model_ft.get_weights())

pt_output = base_model(im_in)

mb = block(shape=tf.Tensor.get_shape(pt_output)[1:])
ftt = att(shape=(img_width, img_height, 3))

mb_output = mb(pt_output)
ftt_output = ftt(im_in)
######## final addition #########

x2 = Add()([mb_output, ftt_output])
x2 = Dense(2, kernel_regularizer=l2(1e-5))(x2)
x2 = Activation('softmax')(x2)

model_top = Model(inputs=im_in, outputs=x2)
model_top.summary()

# optimizer = SGD(lr=1e-3, momentum=0.9, nesterov=True)
optimizer = Adam()
model_top.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['acc'])
callback_list = [EarlyStopping(monitor='val_acc', patience=es_patieance), ReduceLROnPlateau(monitor='loss', factor=reduce_factor, cooldown=0, patience=5, min_lr=0.5e-5)]
output = model_top.fit_generator(ft_gen, steps_per_epoch=200, epochs=100,
                                  validation_data=validation_generator, validation_steps=len(validation_generator), callbacks=callback_list)
model_top.save('fdftnet')


test_dir = args.test_dir

test_generator = test_datagen.flow_from_directory(test_dir, target_size=(64, 64), batch_size=128, shuffle=False, class_mode='categorical')

output = model_top.predict(test_generator, steps=len(test_generator), verbose=1)
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
print(output)

output_score = []
output_class = []
answer_class = []
answer_class_1 =[]

for i in trange(len(test_generator)):
    output = model_top.predict_on_batch(test_generator[i][0])
    output_score.append(output)
    answer_class.append(test_generator[i][1])
    
output_score = np.concatenate(output_score)
answer_class = np.concatenate(answer_class)
output_class = np.argmax(output_score, axis=1)
answer_class_1 = np.argmax(answer_class, axis=1)

print(output_class)
print(answer_class_1)

cm = confusion_matrix(answer_class_1, output_class)
report = classification_report(answer_class_1, output_class)

recall = cm[0][0] / (cm[0][0] + cm[0][1])
fallout = cm[1][0] / (cm[1][0] + cm[1][1])

fpr, tpr, thresholds = roc_curve(answer_class_1, output_score[:, 1], pos_label=1.)
eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
thresh = interp1d(fpr, thresholds)(eer)

print(report)
print(cm)
print("AUROC: %f" %(roc_auc_score(answer_class_1, output_score[:, 1])))
print(thresh)
print('test_acc: ', len(output_class[np.equal(output_class, answer_class_1)]) / len(output_class))
