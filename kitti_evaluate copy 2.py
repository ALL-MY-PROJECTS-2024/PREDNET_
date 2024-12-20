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

# Calculate MSE of PredNet predictions vs. using last frame
print("Calculating MSE...")
mse_model = np.mean((X_test[:, 1:] - X_hat[:, 1:])**2)  # Compare predicted frames to actual frames
mse_prev = np.mean((X_test[:, :-1] - X_test[:, 1:])**2)  # Compare previous frames to actual frames

# Print MSE results
print(f"Model MSE: {mse_model}")
print(f"Previous Frame MSE: {mse_prev}")

# Create results directory if it doesn't exist
if not os.path.exists(RESULTS_SAVE_DIR):
    os.makedirs(RESULTS_SAVE_DIR)

# Write MSE results to file
print("Saving results to files...")
with open(os.path.join(RESULTS_SAVE_DIR, 'prediction_scores.txt'), 'w') as f:
    f.write("Model MSE: %f\n" % mse_model)
    f.write("Previous Frame MSE: %f" % mse_prev)

# Plot predictions
print("Plotting predictions...")
aspect_ratio = float(X_hat.shape[2]) / X_hat.shape[3]
plt.figure(figsize=(nt, 2 * aspect_ratio))
gs = gridspec.GridSpec(2, nt)
gs.update(wspace=0., hspace=0.)

# Set up save directory for plots
plot_save_dir = os.path.join(RESULTS_SAVE_DIR, 'prediction_plots/')
if not os.path.exists(plot_save_dir):
    os.makedirs(plot_save_dir)

plot_idx = np.random.permutation(X_test.shape[0])[:n_plot]
for i in plot_idx:
    for t in range(nt):
        plt.subplot(gs[t])
        plt.imshow(X_test[i, t], interpolation='none')
        plt.tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off', labelbottom='off', labelleft='off')
        if t == 0: 
            plt.ylabel('Actual', fontsize=10)

        plt.subplot(gs[t + nt])
        plt.imshow(X_hat[i, t], interpolation='none')
        plt.tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off', labelbottom='off', labelleft='off')
        if t == 0:
            plt.ylabel('Predicted', fontsize=10)

    plt.savefig(os.path.join(plot_save_dir, 'plot_' + str(i) + '.png'))
    plt.clf()

print("Evaluation complete.")
