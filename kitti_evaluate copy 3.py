import cv2
import os
import numpy as np
from six.moves import cPickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from keras import backend as K
from keras.models import Model, model_from_json
from keras.layers import Input, Dense, Flatten

from prednet import PredNet
from data_utils import SequenceGenerator
from kitti_settings import *

n_plot = 40
batch_size = 10
nt = 10

weights_file = os.path.join(WEIGHTS_DIR, 'prednet_kitti_weights.hdf5')
json_file = os.path.join(WEIGHTS_DIR, 'prednet_kitti_model.json')
test_file = os.path.join(DATA_DIR, 'X_test.hkl')  # Correct path for X_test.hkl
test_sources = os.path.join(DATA_DIR, 'sources_test.hkl')  # Correct path for sources_test.hkl

print("weights_file", weights_file)
print("json_file", json_file)
print("test_file", test_file)
print("test_sources", test_sources)

# Load trained model
print("Loading model...")
f = open(json_file, 'r')
json_string = f.read()
f.close()
train_model = model_from_json(json_string, custom_objects={'PredNet': PredNet})
train_model.load_weights(weights_file)
print("Model loaded successfully.")

# Create testing model (to output predictions)
print("Creating testing model...")
layer_config = train_model.layers[1].get_config()
layer_config['output_mode'] = 'prediction'
data_format = layer_config['data_format'] if 'data_format' in layer_config else layer_config['dim_ordering']
test_prednet = PredNet(weights=train_model.layers[1].get_weights(), **layer_config)
input_shape = list(train_model.layers[0].batch_input_shape[1:])
input_shape[0] = nt
inputs = Input(shape=tuple(input_shape))
predictions = test_prednet(inputs)
test_model = Model(inputs=inputs, outputs=predictions)

# Initialize test data generator
print("Loading test data...")
test_generator = SequenceGenerator(test_file, test_sources, nt, sequence_start_mode='unique', data_format=data_format)
X_test = test_generator.create_all()
print(f"Loaded {X_test.shape[0]} test samples.")

# Predict using the model
print("Predicting test samples...")
X_hat = test_model.predict(X_test, batch_size)
print(f"Predictions done. Predicted {X_hat.shape[0]} samples.")

# Handle data format (if 'channels_first')
if data_format == 'channels_first':
    X_test = np.transpose(X_test, (0, 1, 3, 4, 2))
    X_hat = np.transpose(X_hat, (0, 1, 3, 4, 2))

# Create a video writer for saving the video
output_video_path = os.path.join(RESULTS_SAVE_DIR, 'predicted_vs_actual_video.mp4')
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Using mp4 format
fps = 30  # You can adjust FPS as needed
frame_width = X_test.shape[3] * 2  # Assuming combined frame width (actual + predicted)
frame_height = X_test.shape[2]  # Assuming frame height remains the same
video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

print("Saving video...")
for i in range(X_test.shape[0]):  # Loop through test samples
    for t in range(nt):  # Loop through the frames of each sample
        # Get the actual and predicted frames
        actual_frame = X_test[i, t]
        predicted_frame = X_hat[i, t]

        # Resize predicted frame to match the actual frame dimensions (if necessary)
        if actual_frame.shape != predicted_frame.shape:
            predicted_frame = cv2.resize(predicted_frame, (actual_frame.shape[1], actual_frame.shape[0]))

        # Stack the frames horizontally (actual + predicted)
        combined_frame = np.hstack((actual_frame, predicted_frame))

        # Write the combined frame to the video file
        video_writer.write(combined_frame)

# Release the video writer
video_writer.release()

print(f"Video saved at {output_video_path}")
print("Evaluation complete.")
