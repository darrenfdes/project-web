from flask import Flask, request, render_template, send_from_directory, redirect
import os
from PIL import Image
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from numpy.random import randint
from matplotlib import pyplot
from numpy import vstack
from numpy import load
from numpy import asarray
from keras.models import load_model
import io
import numpy
import tensorflow as tf
from os import listdir
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from PIL import Image


app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))


def load_image(path, size=(256, 256)):
    data_list = []
    # load and resize the image
    pixels = load_img(path, color_mode="rgb",
                      target_size=size, interpolation='nearest')
    pixels = img_to_array(pixels)
    # store
    data_list.append(pixels)
    return asarray(data_list)


def select_sample(dataset, n_samples):
    ix = randint(0, dataset.shape[0], n_samples)
    print("ix:", ix)
    X = (dataset[ix] - 127.5)/127.5
    return X


def show_plot2(imagesX, imagesY1):
    images = vstack((imagesX, imagesY1))
    titles = ['CT', 'Generated MRI']
    # scale from [-1,1] to [0,1]
    images = (images + 1) / 2.0
    # plot images row by row
    for i in range(len(images)):
        # define subplot
        pyplot.subplot(1, len(images), 1 + i)
        # turn off axis
        pyplot.axis('off')
        # plot raw pixel data
        pyplot.imshow(images[i])
        # title
        pyplot.title(titles[i])
    pyplot.savefig('static/images/plot2.png')


def show_plot(imagesX, imagesY1, imagesY2):
    images = vstack((imagesX, imagesY1, imagesY2))
    titles = ['Real', 'Generated', 'Reconstructed']
    # scale from [-1,1] to [0,1]
    images = (images + 1) / 2.0
    # plot images row by row
    for i in range(len(images)):
        # define subplot
        pyplot.subplot(1, len(images), 1 + i)
        # turn off axis
        pyplot.axis('off')
        # plot raw pixel data
        pyplot.imshow(images[i])
        # title
        pyplot.title(titles[i])
    pyplot.savefig('static/images/plot3.png')


# default access page
@app.route("/")
def main():
    return render_template('index.html')


# upload selected image and forward to processing page
@app.route("/upload", methods=["POST"])
def upload():
    target = os.path.join(APP_ROOT, 'static/images/')

    # create image directory if not found
    if not os.path.isdir(target):
        os.mkdir(target)

    # retrieve file from html file-picker
    upload = request.files.getlist("file")[0]
    print("File name: {}".format(upload.filename))
    filename = upload.filename

    # file support verification
    ext = os.path.splitext(filename)[1]
    if (ext == ".jpg") or (ext == ".png") or (ext == ".bmp"):
        print("File accepted")
    else:
        return render_template("error.html", message="The selected file is not supported"), 400

    # save file
    destination = "/".join([target, filename])
    print("File saved to to:", destination)
    upload.save(destination)

    # forward to processing page
    return render_template("processing.html", image_name=filename)


# flip filename 'vertical' or 'horizontal'
@app.route("/flip", methods=["POST"])
def flip():

    # retrieve parameters from html form
    if 'horizontal' in request.form['mode']:
        mode = 'horizontal'
    elif 'vertical' in request.form['mode']:
        mode = 'vertical'
    elif 'both' in request.form['mode']:
        mode = 'both'
    else:
        return render_template("error.html", message="Mode not supported"), 400
    filename = request.form['image']

    # open and process image
    target = os.path.join(APP_ROOT, 'static/images')
    destination = "/".join([target, filename])

    img = Image.open(destination)
    data = load_image(destination)
    if mode == 'vertical':
        cust = {'InstanceNormalization': InstanceNormalization}
        model_AtoB = load_model('static/models/g_model_AtoB_018060.h5', cust)
        A_real = select_sample(data, 1)
        B_generated = model_AtoB.predict(A_real)
        b_gen = numpy.squeeze(B_generated)
        img = tf.keras.preprocessing.image.array_to_img(
            b_gen, data_format=None, scale=True, dtype=None
        )
    elif mode == 'horizontal':
        cust = {'InstanceNormalization': InstanceNormalization}
        model_AtoB = load_model('static/models/g_model_AtoB_018060.h5', cust)
        A_real = select_sample(data, 1)
        B_generated = model_AtoB.predict(A_real)
        show_plot2(A_real, B_generated)
        img = Image.open('static/images/plot2.png')
    else:
        cust = {'InstanceNormalization': InstanceNormalization}
        model_AtoB = load_model('static/models/g_model_AtoB_018060.h5', cust)
        model_BtoA = load_model('static/models/g_model_BtoA_018060.h5', cust)
        A_real = select_sample(data, 1)
        B_generated = model_AtoB.predict(A_real)
        A_generated = model_BtoA.predict(B_generated)
        show_plot(A_real, B_generated, A_generated)
        img = Image.open('static/images/plot3.png')

    # save and return image
    destination = "/".join([target, 'temp.png'])
    if os.path.isfile(destination):
        os.remove(destination)
    img.save(destination)

    return send_image('temp.png')


# retrieve file from 'static/images' directory
@app.route('/static/images/<filename>')
def send_image(filename):
    return send_from_directory("static/images", filename)


if __name__ == "__main__":
    app.run()
