import tensorflow as tf
from keras.models import Model, load_model
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from tqdm import tqdm, trange
from sklearn import metrics
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, classification_report
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import argparse
from utils import *

parser = argparse.ArgumentParser(description='Pretrain the models')
parser.add_argument('-model', required=True, type=str, help='pretrained backbone model')
parser.add_argument('-test_dir', required=True, type=str, help='fine tune image directory')
args = parser.parse_args()

#HardSwish.className = 'HardSwish'
#tf.serialization.SerializationMap.register(HardSwish)

test_dir = args.test_dir
model = args.model

test_datagen = ImageDataGenerator(rescale=1./255)

test50_generator = test_datagen.flow_from_directory(test_dir,
                                                  target_size=(64, 64),
                                                  batch_size=128,
                                                  shuffle=False,
                                                  class_mode='categorical')
custom_object = {"HardSwish":HardSwish, "HardSigmoid":HardSigmoid}
model_top = tf.keras.models.load_model(model, custom_objects=custom_object)

output = model_top.predict(test50_generator, steps=len(test50_generator), verbose=1)
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
#print(test50_generator.class_indices)
print(output)

output_score50 = []
output_class50 = []
answer_class50 = []
answer_class50_1 =[]

for i in trange(len(test50_generator)):
    output50 = model_top.predict_on_batch(test50_generator[i][0])
    output_score50.append(output50)
    answer_class50.append(test50_generator[i][1])
    
output_score50 = np.concatenate(output_score50)
answer_class50 = np.concatenate(answer_class50)

output_class50 = np.argmax(output_score50, axis=1)
answer_class50_1 = np.argmax(answer_class50, axis=1)

print(output_class50)
print(answer_class50_1)

cm50 = confusion_matrix(answer_class50_1, output_class50)
report50 = classification_report(answer_class50_1, output_class50)

recall50 = cm50[0][0] / (cm50[0][0] + cm50[0][1])
fallout50 = cm50[1][0] / (cm50[1][0] + cm50[1][1])

fpr50, tpr50, thresholds50 = roc_curve(answer_class50_1, output_score50[:, 1], pos_label=1.)
eer50 = brentq(lambda x : 1. - x - interp1d(fpr50, tpr50)(x), 0., 1.)
thresh50 = interp1d(fpr50, thresholds50)(eer50)

print(report50)
print(cm50)
print("AUROC: %f" %(roc_auc_score(answer_class50_1, output_score50[:, 1])))
print(thresh50)
print('test_acc: ', len(output_class50[np.equal(output_class50, answer_class50_1)]) / len(output_class50))