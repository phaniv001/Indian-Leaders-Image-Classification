#from crypt import methods
from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
from werkzeug.utils import secure_filename
import numpy as np
import PIL
import os, glob
#from IPython.display import Image

app = Flask(__name__)

label = ["APJ Abdul Kalam","Atal Bihari Vajpayee","Chandra Babu Naidu","Dr. B. R. Ambedkar",
          "Indira Gandhi","Lata Mangeshkar","Ratan Tata",
          "Narendra Modi","Vallabhbhai Patel"]


load_model = tf.keras.models.load_model("Image_classifier.h5")

def model_predict(image_path, load_model):
    #img_path = r"D:/Data Science/Deep Learning/image_classification/sample_dataset/"
    #img = image.load_img(image_path, target_size = (224, 224))
    img = tf.keras.preprocessing.image.load_img(image_path, target_size = (224, 224))
    img_arr = image.img_to_array(img)
    img_expand_arr = np.expand_dims(img_arr, axis = 0)
    predict = np.argmax(load_model.predict(preprocess_input(img_expand_arr)), axis = 1)
    #result = label[predict.item()]
    return predict


@app.route("/", methods = ['GET'])
def index():
    return render_template("index.html")


@app.route("/predict", methods = ["GET", "POST"])
def upload():
    if request.method == "POST":
        f = request.files["file"]
        base_path = os.path.dirname(__file__)
        file_path = os.path.join(base_path, "uploads", secure_filename(f.filename))
        f.save(file_path)

        preds = model_predict(file_path, load_model) 
        result = label[preds.item()]
        return result
    return None


if __name__ == '__main__':
    app.run(host = '0.0.0.0')

