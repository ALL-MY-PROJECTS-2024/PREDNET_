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
nt = 60

weights_file = os.path.join(WEIGHTS_DIR, 'prednet_kitti_weights.hdf5')
json_file = os.path.join(WEIGHTS_DIR, 'prednet_kitti_model.json')
test_file = os.path.join(DATA_DIR, 'X_train.hkl')  # Correct path for X_test.hkl
test_sources = os.path.join(DATA_DIR, 'sources_train.hkl')  # Correct path for sources_test.hkl

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

# Check the shape of X_test before slicing
print(f"Shape of X_test: {X_test.shape}")

# Cutting the test data for prediction (start prediction after 60th frame)
if X_test.shape[1] > 60:
    # Cutting the test data for prediction (start prediction after 60th frame)
    X_test_cut = X_test[:, 60:]  # After the 60th frame, the data is cut for prediction
    print(f"Using {X_test_cut.shape[1]} frames for prediction.")
    
    # Predict using the model
    print("Predicting test samples...")
    X_hat = test_model.predict(X_test_cut, batch_size)  # Predicting on the cut test data
    print(f"Predictions done. Predicted {X_hat.shape[0]} samples.")
else:
    print("X_test does not contain enough frames for slicing. Adjusting prediction range.")
    # If data is less than 60 frames, adjust nt and predict on the entire X_test
    nt = X_test.shape[1]  # Adjust nt to the available frames
    print(f"Adjusting nt to {nt} based on available data.")
    X_hat = test_model.predict(X_test, batch_size)  # Predict on the entire `X_test`

# Handle data format (if 'channels_first')
if data_format == 'channels_first':
    X_test = np.transpose(X_test, (0, 1, 3, 4, 2))
    X_hat = np.transpose(X_hat, (0, 1, 3, 4, 2))

# Create a video writer for saving the video
output_video_path = os.path.join(RESULTS_SAVE_DIR, 'predicted_vs_actual_video.mp4')
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Using mp4 format
fps = 30  # You can adjust FPS as needed

# Get frame dimensions from X_test (assuming all frames have the same dimensions)
frame_height, frame_width = X_test.shape[2], X_test.shape[3]
frame_width_combined = frame_width * 2  # Combined width (actual + predicted)

# Create the video writer with the correct dimensions
video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width_combined, frame_height))

# Folder to save images
image_save_dir = os.path.join(RESULTS_SAVE_DIR, 'prediction_images')
if not os.path.exists(image_save_dir):
    os.makedirs(image_save_dir)

print("Saving video and images...")
for i in range(X_test.shape[0]):  # Loop through test samples
    for t in range(nt):  # Loop through the frames of each sample
        # Get the actual and predicted frames
        actual_frame = X_test[i, t]
        predicted_frame = X_hat[i, t]

        # Ensure the predicted frame matches the actual frame dimensions (resize if necessary)
        if actual_frame.shape != predicted_frame.shape:
            predicted_frame = cv2.resize(predicted_frame, (actual_frame.shape[1], actual_frame.shape[0]))

        # Convert the frame back to 0-255 range (if normalized)
        actual_frame = np.uint8(actual_frame * 255)
        predicted_frame = np.uint8(predicted_frame * 255)

        # Convert RGB to BGR if needed
        actual_frame_bgr = cv2.cvtColor(actual_frame, cv2.COLOR_RGB2BGR)
        predicted_frame_bgr = cv2.cvtColor(predicted_frame, cv2.COLOR_RGB2BGR)

        # Stack the frames horizontally (actual + predicted)
        combined_frame = np.hstack((actual_frame_bgr, predicted_frame_bgr))

        # Save the actual, predicted, and combined frames as images
        actual_image_path = os.path.join(image_save_dir, f"actual_frame_{i}_t{t}.png")
        predicted_image_path = os.path.join(image_save_dir, f"predicted_frame_{i}_t{t}.png")
        combined_image_path = os.path.join(image_save_dir, f"combined_frame_{i}_t{t}.png")

        # Debugging: Check if the frames are being written
        print(f"Saving frames for sample {i}, frame {t}")

        cv2.imwrite(actual_image_path, actual_frame_bgr)  # Save actual frame
        cv2.imwrite(predicted_image_path, predicted_frame_bgr)  # Save predicted frame
        cv2.imwrite(combined_image_path, combined_frame)  # Save combined frame

        # Write the combined frame to the video file
        video_writer.write(combined_frame)

# Release the video writer
video_writer.release()

print(f"Video saved at {output_video_path}")
print(f"Images saved in {image_save_dir}")
print("Evaluation complete.")
