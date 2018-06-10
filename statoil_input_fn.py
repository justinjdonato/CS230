"""Augment and prepare the training sets, Concatenate the band1 and band2 data into a 3D image"""

import pandas as pd
import os
import numpy as np

import model.statoil_utils as statoil_utils


def augment_data(band1, band2, angles, labels):
    """a function to augment band1 and band2 image"""

    # list to store the generated data
    band1_generated = []
    band2_generated = []
    angles_generated = []
    labels_generated = []

    # iterate through each point in train set
    for i in range(labels.shape[0]):
        # rotate by positive degree
        angle = np.random.randint(5, 20)
        band1_generated.append(statoil_utils.rotate_image(band1[i], angle))
        band2_generated.append(statoil_utils.rotate_image(band2[i], angle))
        angles_generated.append(angles[i])
        labels_generated.append(labels[i])

        # rotate by negative degree
        angle = np.random.randint(5, 20)
        band1_generated.append(statoil_utils.rotate_image(band1[i], -angle))
        band2_generated.append(statoil_utils.rotate_image(band2[i], -angle))
        angles_generated.append(angles[i])
        labels_generated.append(labels[i])

        # positive horizontal shift
        shift = np.random.randint(3, 7)
        band1_generated.append(statoil_utils.translate_horizontal(band1[i], +shift))
        band2_generated.append(statoil_utils.translate_horizontal(band2[i], +shift))
        angles_generated.append(angles[i])
        labels_generated.append(labels[i])

        # negative horizontal shift
        shift = np.random.randint(3, 7)
        band1_generated.append(statoil_utils.translate_horizontal(band1[i], -shift))
        band2_generated.append(statoil_utils.translate_horizontal(band2[i], -shift))
        angles_generated.append(angles[i])
        labels_generated.append(labels[i])

        # positive vertical shift
        shift = np.random.randint(0, 7)
        band1_generated.append(statoil_utils.translate_vertical(band1[i], +shift))
        band2_generated.append(statoil_utils.translate_vertical(band2[i], +shift))
        angles_generated.append(angles[i])
        labels_generated.append(labels[i])

        # negative vertical shift
        shift = np.random.randint(3, 7)
        band1_generated.append(statoil_utils.translate_vertical(band1[i], -shift))
        band2_generated.append(statoil_utils.translate_vertical(band2[i], -shift))
        angles_generated.append(angles[i])
        labels_generated.append(labels[i])

        # translate along positive diagonal in positive direction
        shift = np.random.randint(3, 7)
        band1_generated.append(statoil_utils.translate_positive_diagonal(band1[i], +shift))
        band2_generated.append(statoil_utils.translate_positive_diagonal(band2[i], +shift))
        angles_generated.append(angles[i])
        labels_generated.append(labels[i])

        # translate along positive diagonal in negative direction
        shift = np.random.randint(3, 7)
        band1_generated.append(statoil_utils.translate_positive_diagonal(band1[i], -shift))
        band2_generated.append(statoil_utils.translate_positive_diagonal(band2[i], -shift))
        angles_generated.append(angles[i])
        labels_generated.append(labels[i])

        # translate along negative diagonal in positive direction
        shift = np.random.randint(3, 7)
        band1_generated.append(statoil_utils.translate_negative_diagonal(band1[i], +shift))
        band2_generated.append(statoil_utils.translate_negative_diagonal(band2[i], +shift))
        angles_generated.append(angles[i])
        labels_generated.append(labels[i])

        # translate along negative diagonal in negative direction
        shift = np.random.randint(3, 7)
        band1_generated.append(statoil_utils.translate_negative_diagonal(band1[i], -shift))
        band2_generated.append(statoil_utils.translate_negative_diagonal(band2[i], -shift))
        angles_generated.append(angles[i])
        labels_generated.append(labels[i])

        # vertical flip
        band1_generated.append(statoil_utils.flip(band1[i], 0))
        band2_generated.append(statoil_utils.flip(band2[i], 0))
        angles_generated.append(angles[i])
        labels_generated.append(labels[i])

        # horizontal flip
        band1_generated.append(statoil_utils.flip(band1[i], 1))
        band2_generated.append(statoil_utils.flip(band2[i], 1))
        angles_generated.append(angles[i])
        labels_generated.append(labels[i])

        # zoom in image
        zoom_shift = np.random.randint(2, 5)
        band1_generated.append(statoil_utils.zoom(band1[i], zoom_shift))
        band2_generated.append(statoil_utils.zoom(band2[i], zoom_shift))
        angles_generated.append(angles[i])
        labels_generated.append(labels[i])

        # zoom out image
        zoom_shift = np.random.randint(2, 5)
        band1_generated.append(statoil_utils.zoom(band1[i], -zoom_shift))
        band2_generated.append(statoil_utils.zoom(band2[i], -zoom_shift))
        angles_generated.append(angles[i])
        labels_generated.append(labels[i])

    # convert the generated data into numpy array
    band1_generated = np.array(band1_generated)
    band2_generated = np.array(band2_generated)
    angles_generated = np.array(angles_generated)
    labels_generated = np.array(labels_generated)

    # concatenate the generated data to original train set
    band1_augmented = np.concatenate((band1, band1_generated), axis=0)
    band2_augmented = np.concatenate((band2, band2_generated), axis=0)
    angles_augmented = np.concatenate((angles, angles_generated), axis=0)
    labels_augmented = np.concatenate((labels, labels_generated), axis=0)

    return band1_augmented, band2_augmented, angles_augmented, labels_augmented


def statoil_input_fn(train_data_dir, dev_data_dir, test_data_dir):
    """Input function for the dataset."""

    np_rand_seed = 97
    # tf_rand_seed = 82
    np.random.seed(np_rand_seed)
    # tf.set_random_seed(tf_rand_seed)

    inputs = {'images': ''}

    try:
        # Read in from csv store
        df_read_b1_train_csv = pd.read_csv(os.path.join(train_data_dir, 'band_1_train.csv'), header=None, index_col=0, encoding="ISO-8859-1")
        df_read_b2_train_csv = pd.read_csv(os.path.join(train_data_dir, 'band_2_train.csv'), header=None, index_col=0, encoding="ISO-8859-1")
        df_read_angles_train_csv = pd.read_csv(os.path.join(train_data_dir, 'angles_train.csv'), header=None, index_col=0, encoding="ISO-8859-1")
        df_read_labels_train_csv = pd.read_csv(os.path.join(train_data_dir, 'labels_train.csv'), header=None, index_col=0, encoding="ISO-8859-1")

        df_read_b1_valid_csv = pd.read_csv(os.path.join(dev_data_dir, 'band_1_valid.csv'), header=None, index_col=0, encoding="ISO-8859-1")
        df_read_b2_valid_csv = pd.read_csv(os.path.join(dev_data_dir, 'band_2_valid.csv'), header=None, index_col=0, encoding="ISO-8859-1")
        df_read_angles_valid_csv = pd.read_csv(os.path.join(dev_data_dir, 'angles_valid.csv'), header=None, index_col=0, encoding="ISO-8859-1")
        df_read_labels_valid_csv = pd.read_csv(os.path.join(dev_data_dir, 'labels_valid.csv'), header=None, index_col=0, encoding="ISO-8859-1")

        df_read_b1_test_csv = pd.read_csv(os.path.join(test_data_dir, 'band_1_test.csv'), header=None, index_col=0, encoding="ISO-8859-1")
        df_read_b2_test_csv = pd.read_csv(os.path.join(test_data_dir, 'band_2_test.csv'), header=None, index_col=0, encoding="ISO-8859-1")
        df_read_angles_test_csv = pd.read_csv(os.path.join(test_data_dir, 'angles_test.csv'), header=None, index_col=0, encoding="ISO-8859-1")
        df_read_ID_csv = pd.read_csv(os.path.join(test_data_dir, 'ID.csv'), header=None, index_col=0, encoding="ISO-8859-1")

        # Push out to an array
        band_1_train_a = np.asarray(df_read_b1_train_csv).tolist()
        band_2_train_a = np.asarray(df_read_b2_train_csv).tolist()
        angles_train_a = np.asarray(df_read_angles_train_csv)
        labels_train_a = np.asarray(df_read_labels_train_csv)

        band_1_valid_a = np.asarray(df_read_b1_valid_csv).tolist()
        band_2_valid_a = np.asarray(df_read_b2_valid_csv).tolist()
        angles_valid_a = np.asarray(df_read_angles_valid_csv)
        labels_valid_a = np.asarray(df_read_labels_valid_csv)

        band_1_test_a = np.asarray(df_read_b1_test_csv).tolist()
        band_2_test_a = np.asarray(df_read_b2_test_csv).tolist()
        angles_test_a = np.asarray(df_read_angles_test_csv)
        ID_a = np.asarray(df_read_ID_csv)

        # reshape back to the 75x75 image
        band_1_train_reshape = np.array([np.array(b).astype(np.float32).reshape(75, 75) for b in band_1_train_a])
        band_2_train_reshape = np.array([np.array(b).astype(np.float32).reshape(75, 75) for b in band_2_train_a])
        band_1_valid_reshape = np.array([np.array(b).astype(np.float32).reshape(75, 75) for b in band_1_valid_a])
        band_2_valid_reshape = np.array([np.array(b).astype(np.float32).reshape(75, 75) for b in band_2_valid_a])
        band_1_test_reshape = np.array([np.array(b).astype(np.float32).reshape(75, 75) for b in band_1_test_a])
        band_2_test_reshape = np.array([np.array(b).astype(np.float32).reshape(75, 75) for b in band_2_test_a])

        print("Shape of restored band_1_train_reshape:", band_1_train_reshape.shape)
        print("Shape of restored band_2_train_reshape:", band_2_train_reshape.shape)

        #  **** CAUTION shapes have been hard coded ****
        band_1_train = band_1_train_reshape
        band_2_train = band_2_train_reshape
        angles_train = angles_train_a.reshape(1103,)
        labels_train = labels_train_a.reshape(1103,)

        band_1_validation = band_1_valid_reshape
        band_2_validation = band_2_valid_reshape
        angles_validation = angles_valid_a.reshape(368,)
        labels_validation = labels_valid_a.reshape(368,)

        band_1_test = band_1_test_reshape
        band_2_test = band_2_test_reshape
        angles_test = angles_test_a.reshape(8424,)  # angles_test.shape)
        iD = ID_a.reshape(8424,)

        # **** Augment the training  set ****
        band_1_train, band_2_train, angles_train, labels_train = augment_data(band_1_train, band_2_train, angles_train, labels_train)

        print('**** Examine the shape of augmented data ****')
        print("Shape of band_1_train:", band_1_train.shape)
        print("Shape of band_2_train:", band_2_train.shape)
        print("Shape of angles_train:", angles_train.shape)
        print("Shape of labels_train:", labels_train.shape)

        # **** Concatenate the band1 and band2 data into 3D image ****
        # **** Here we stack band_1, band_2, and average of the two to create a 3D image ****
        image_train = np.concatenate([band_1_train[:, :, :, np.newaxis],
                                     band_2_train[:, :, :, np.newaxis],
                                     ((band_1_train+band_2_train)/2)[:, :, :, np.newaxis]],
                                     axis=-1)

        image_validation = np.concatenate([band_1_validation[:, :, :, np.newaxis],
                                     band_2_validation[:, :, :, np.newaxis],
                                     ((band_1_validation+band_2_validation)/2)[:, :, :, np.newaxis]],
                                     axis=-1)

        image_test = np.concatenate([band_1_test[:, :, :, np.newaxis],
                                     band_2_test[:, :, :, np.newaxis],
                                     ((band_1_test+band_2_test)/2)[:, :, :, np.newaxis]],
                                     axis=-1)

        # Memory clean up
        del(band_1_train, band_1_validation, band_1_test, band_2_train, band_2_validation, band_2_test)

        print('**** Examine the shape of 3D images ****')
        print("Shape of image_train:", image_train.shape)
        print("Shape of image_validation:", image_validation.shape)
        print("Shape of image_test:", image_test.shape)

        inputs["image_train"] = image_train
        inputs["image_validation"] = image_validation
        inputs["image_test"] = image_test
        inputs["labels_train"] = labels_train
        inputs["labels_validation"] = labels_validation
        inputs["iD"] = iD

    except Exception as e:
        print(e)

    return inputs
