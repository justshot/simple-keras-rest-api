# USAGE
# Start the server:
# 	python run_keras_server.py
# Submit a request via cURL:
# 	curl -X POST -F image=@dog.jpg 'http://localhost:5000/predict'
# Submita a request via Python:
#	python simple_request.py

# import the necessary packages
from keras.applications import ResNet50
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
from PIL import Image
import SiameseNet_Signature
import numpy as np
import flask
import io
import json
import cv2 as cv
import tensorflow as tf
from flask_cors import CORS

# initialize our Flask application and the Keras model
app = flask.Flask(__name__)
CORS(app)
siamese_net = None
image_size = 105
graph = tf.get_default_graph()

def load_model():
	# load the pre-trained Keras model (here we are using a model
	# pre-trained on ImageNet and provided by Keras, but you can
	# substitute in your own networks just as easily)
	global siamese_net 
	siamese_net = SiameseNet_Signature.create_model()
	siamese_net.load_weights('weights')

@app.route("/verify", methods=["POST"])
def verify():
	data = "0"
	if flask.request.method == "POST":
		image1 = flask.request.files.get("image1")
		image2 = flask.request.files.get("image2")
		if  image1 and image2:
			image1 = image1.read()
			image1 = np.fromstring(image1, np.uint8)
			image1 = cv.imdecode(image1, 0)
			image2 = image2.read()
			image2 = np.fromstring(image2, np.uint8)
			image2 = cv.imdecode(image2, 0)
			new_image1 = cv.resize(image1,(image_size,image_size))
			new_image2 = cv.resize(image2,(image_size,image_size))
			print("shape of new image 1: {}".format(new_image1.shape))

			#prepare input
			h = new_image1.shape[0]
			w = new_image1.shape[1]
			pairs=[np.zeros((1, h, w,1)) for i in range(2)]
			pairs[0][0,:,:,:] = new_image1.reshape(w, h, 1)
			pairs[1][0,:,:,:] = new_image2.reshape(w, h, 1)

			# inputs, targets = SiameseNet_Signature.loader.get_batch(10,'sig_data_train')
			global graph
			with graph.as_default():
				probs = siamese_net.predict(pairs)
			probs = probs.tolist()
			data = json.dumps(probs)
			print("data: {}".format(data))
	return data

# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
	print(("* Loading Keras model and Flask starting server..."
		"please wait until server has fully started"))
	load_model()
	app.run()