"""Train the model"""

import argparse
import logging
import os

import tensorflow as tf
import numpy as np

from model.statoil_input_fn import statoil_input_fn
from model.statoil_model_fn import statoil_model_fn
from model.utils import Params
from model.utils import set_logger

# **** Source command line arguments ****
parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='statoil_experiments\\base_model',
                    help="Experiment directory containing params.json")
parser.add_argument('--data_dir', default='data',
                    help="Directory containing the dataset")
# parser.add_argument('--restore_from', default=None,
#                     help="Optional, directory or file containing weights to reload before training")


if __name__ == '__main__':
    # Set the random seed for the whole graph for reproductible experiments
    np_rand_seed = 97
    tf_rand_seed = 82
    np.random.seed(np_rand_seed)
    tf.set_random_seed(tf_rand_seed)

    # Load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    params = Params(json_path)

    # Check that we are not overwriting some previous experiment
    # Comment these lines if you are developing your model and don't care about over writing
    # model_dir_has_best_weights = os.path.isdir(os.path.join(strRootPath, "best_weights"))
    # over writing = model_dir_has_best_weights  # and args.restore_from is None
    # assert not over writing, "Weights found in model_dir, aborting to avoid overwrite"

    # Set the logger
    set_logger(os.path.join(args.model_dir, 'train.log'))

    # Create the input data sets
    logging.info("Creating the data set ...")
    data_dir = args.data_dir
    train_data_dir = os.path.join(data_dir, "train")
    dev_data_dir = os.path.join(data_dir, "dev")
    test_data_dir = os.path.join(data_dir, "test")

    # **** Augment and prepare the training sets, Concatenate the band1 and band2 data into a 3D image ****
    model_inputs = statoil_input_fn(train_data_dir, dev_data_dir, test_data_dir)

    # **** Define and Execute the model ****
    logging.info("Creating and Executing the model...")
    model_outputs = {'best_train_loss': 1000000, 'best_validation_loss': 1000000, 'best_valid_accuracy': 0,
                     'best_model_num': 1, 'best_iteration': 0}

    num_of_models = params.num_of_models
    valid_loss = {1: ''}

    for model_num in range(0, num_of_models):
        print('model number: ' + str(model_num + 1))

        if model_num < (num_of_models - 1):
            model_outputs = statoil_model_fn(model_inputs, model_outputs, params, model_num, False)
            valid_loss[model_num + 1] = model_outputs['valid_loss']
        else:
            model_outputs = statoil_model_fn(model_inputs, model_outputs, params, model_num, True)
