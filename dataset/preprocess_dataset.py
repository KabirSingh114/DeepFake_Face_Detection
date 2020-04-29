import os
import glob
import cv2
from mtcnn import MTCNN
import tensorflow as tf

os.environ["CUDA_DEVICE_ORDER"] = "0, 1"

def crop_face(img_dir):
    detector = MTCNN()

    os.chdir(img_dir)

    image_list = glob.glob("*")

    for i in image_list:
        try:
            image = cv2.imread(i)
            image2 = image.copy()
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            result = detector.detect_faces(image)
            bounding_box = result[0]['box']

            cropped_image = image2[bounding_box[1]:(bounding_box[1] + bounding_box[3]),
                                    bounding_box[0]:(bounding_box[0] + bounding_box[2])]

            
            cv2.imwrite(i, cropped_image)
        except:
            print("couldn't update image ")
            print(i)

    #os.chdir(root_dir)

def load_img(path_to_img):
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_jpeg(img, channels=3)
    return tf.image.resize_with_pad(img, 64, 64, method=tf.image.ResizeMethod.BILINEAR, antialias=False)


def resize(img_dir):
    list_ds = tf.data.Dataset.list_files(str(img_dir + '/*.jpg'))
    for f in list_ds.take(5):
        print(f.numpy())

    for f in list_ds:
        print(f.numpy())
        image = load_img(f.numpy())
        image = tf.dtypes.cast(image, tf.uint8)
        op = tf.image.encode_jpeg(image, format='rgb', quality=100)
        writeOp = tf.io.write_file(f.numpy(), op)
       
def decode_img(img):
  img = tf.image.decode_jpeg(img, channels=3)
  img = tf.image.convert_image_dtype(img, tf.float32)
  return tf.image.resize(img, [64, 64])