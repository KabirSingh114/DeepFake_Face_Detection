from network import *
import argparse
import os
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam, SGD
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

# argparse
parser = argparse.ArgumentParser(description='Pretrain the models')

parser.add_argument('-network', required=True, type=str, help='select the backbone network')
parser.add_argument('-train_dir', required=True, type=str, help='train image directory')
parser.add_argument('-val_dir', required=True, type=str, help='validation image directory')
parser.add_argument('-batch_size', required=True, type=int, help='batch_size')
parser.add_argument('-reduce_patience', required=True, type=int, help='reduce patience')
parser.add_argument('-step', required=True, type=int, help='steps per epoch')
parser.add_argument('-epochs', type=int, default=300, help='epochs')
parser.add_argument('-gpu_ids', type=str, default='0', help='select the GPU to use')

args = parser.parse_args()

root_dir = os.getcwd()
weight_save_dir = os.path.join(root_dir, 'weights')

img_height = 64
img_width = 64
es_patience = 20
reduce_factor = .1
dropout_rate = .2

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids

# model selection
if args.network == 'resnetV2':
    model = resNetV2(img_height, img_width)
elif args.network == 'squeezenet':
    model = squeezeNet(img_height, img_width, dropout_rate)
elif args.network == 'mesonet':
    model = mesonet(img_height, img_width, dropout_rate)
elif args.network == 'vgg16':
    model = vgg16(img_height, img_width, dropout_rate)
elif args.network == 'denseNet':
    model = denseNet(img_height, img_width, dropout_rate)

model.summary()



model.compile(optimizer=Adam(),loss='categorical_crossentropy',metrics=['accuracy'])

print(len(model.trainable_weights))

datagenerator = ImageDataGenerator(rescale=1./255,)

train_generator = datagenerator.flow_from_directory(args.train_dir, target_size=(img_height, img_width), batch_size=args.batch_size,
                                                    shuffle=True, class_mode='categorical')

validation_generator = datagenerator.flow_from_directory(args.val_dir, target_size=(img_height, img_width), batch_size=args.batch_size,
                                                         shuffle=False, class_mode='categorical')


callback_list = [EarlyStopping(monitor='val_accuracy', patience=es_patience), ReduceLROnPlateau(monitor='val_loss', factor=reduce_factor, patience=args.reduce_patience)]

history = model.fit_generator(train_generator, steps_per_epoch=args.step, epochs=args.epochs, validation_data=validation_generator, validation_steps=len(validation_generator),
                              callbacks=callback_list)

# save the model weight
model.save(weight_save_dir)
