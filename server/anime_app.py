from flask import request
from flask import jsonify
from flask import Flask,render_template
from flask import url_for
# from keras.preprocessing.image import img_to_array
# import cStringIO
import base64
import io
from PIL import Image
import numpy as np
# from torchvision import transforms    

import base64
from io import BytesIO

import torch
from flask import render_template

# from net import *
from Pyramid_nets import *

app = Flask(__name__)

# model = get_model("AnimeGenerator_state_dict.pth")
model = get_model("Pyramid_gen_state_dict.pth")
main_noise = model.sample_noise(10, noise_size)
#m_noise = micro_noise(batch_size, noise_size, 1000)
num = 500

@app.route("/")
@app.route("/index")
def index():
    my_list = []
    for j in range(num):
        my_list.append(j)
    return render_template('index.html', max_num = batch_size, arr = my_list)

@app.route("/den")
def den():
    my_list = []
    for j in range(10):
        my_list.append(j)
    return render_template('slider.html', max_num = batch_size, arr = my_list)

@app.route('/gnoise', methods=['POST'])
def gnoise():
    message = request.get_json(force=True)
    num = int(message['number'])
    noise = main_noise 

    for i in range(num):
        noise += micro_noise(10, noise_size, 5)
    # main_noise = model.sample_noise(batch_size, noise_size)
    fake_images = model(noise)
    # trans = transforms.ToPILImage(mode='RGB')
    # print(trans(fake_images[0]))
    response = {}

    for i in range(10):
        res = Image.fromarray(((fake_images[i].permute(1, 2, 0).detach().numpy() + 1) / 2 * 255).astype("uint8"), "RGB")
        buff = BytesIO() 
        res.save(buff, format="PNG") 
        result = base64.b64encode(buff.getvalue()).decode("utf-8")

        response["image" + str(i)] = result

    print(result[0:5])
    return jsonify(response)

@app.route('/vector', methods=['GET'])
def vec():
    g_fake_seed = model.sample_noise(batch_size, noise_size)
    fake_images = model(g_fake_seed)
    # print(fake_images.shape)
    # trans = transforms.ToPILImage(mode='RGB')
    # print(trans(fake_images[0]))
    response = {}

    for i in range(batch_size):
        res = Image.fromarray(((fake_images[i].permute(1, 2, 0).detach().numpy() + 1) / 2 * 255).astype("uint8"), "RGB")
    # res = fake_images[10].detach().numpy()

    # assume data contains your decoded image
    # file_like = cStringIO.StringIO(res)
    # result = base64.b64encode(res)
        buff = BytesIO() 
        res.save(buff, format="PNG") 
        result = base64.b64encode(buff.getvalue()).decode("utf-8")

        response["image" + str(i)] = result
    
    # response = {
    #     'image': result
    # }
    return jsonify(response)

@app.route('/generate', methods=['POST'])
def generate():
    message = request.get_json(force=True)
    num = int(message['number'])
    g_fake_seed = model.sample_noise(num, noise_size)
    fake_images = model(g_fake_seed)
    print(fake_images.shape)
    # trans = transforms.ToPILImage(mode='RGB')
    # print(trans(fake_images[0]))
    response = {}

    for i in range(num):
        res = Image.fromarray(((fake_images[i].permute(1, 2, 0).detach().numpy() + 1) / 2 * 255).astype("uint8"), "RGB")
    # res = fake_images[10].detach().numpy()

    # assume data contains your decoded image
    # file_like = cStringIO.StringIO(res)
    # result = base64.b64encode(res)
        buff = BytesIO() 
        res.save(buff, format="PNG") 
        result = base64.b64encode(buff.getvalue()).decode("utf-8")

        response["image" + str(i)] = result
    
    # response = {
    #     'image': result
    # }
    return jsonify(response)


# @app.route("/predict", methods =["POST"])
# def predict():
#     # message = request.get_json(force = True)
#     # encoded = message['image']
#     # decoded = base64.b64decode(encoded)
#     # image = Image.open(io.BytesIO(decoded))
#     # processed_image = preprocess_image(image, target_size=(224,224))
#     image = model.predict()
#     result = base64.b64encode(image)
#     response = {
#         'image': result
#     }    
#     return jsonify(response)
#     # return request.get_json(force = True)