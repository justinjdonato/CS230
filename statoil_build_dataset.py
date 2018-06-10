
# import time
import time
t1 = time.time()

import os
import math
import random
import pandas as pd
import numpy as np


# **Function which return the count and the index of mismatched data**
def find_missing_data(series, shape):
    """function which return the count and the index of mismatched data"""
    count = 0
    missing_list = []
    for i, x in enumerate(series):
        if np.shape(series.iloc[i]) != shape:
            missing_list.append(i)
            count += 1

    return missing_list, count


# **Function to drop data by index**
def drop_data(df, index):
    """function to drop data by index"""
    return df.drop(df.index[index])


# **3 standardization to technique we can try on**
def standardise_vector(vector):
    """standardise vector"""
    standardised_vector = (np.array(vector) - np.mean(vector)) / np.std(vector)
    return standardised_vector.tolist()


if __name__ == '__main__':

    strRootPath = 'data\\'  # get from config
    train_data_dir = os.path.join(strRootPath, 'train')
    dev_data_dir = os.path.join(strRootPath, 'dev')
    test_data_dir = os.path.join(strRootPath, 'test')

    if os.path.exists(strRootPath):
        if not os.path.exists(train_data_dir):
            os.makedirs(train_data_dir)
        if not os.path.exists(dev_data_dir):
            os.makedirs(dev_data_dir)
        if not os.path.exists(test_data_dir):
            os.makedirs(test_data_dir)

    # Making the results reproducible (knowing the random seed of the two libraries)
    np_rand_seed = 97
    tf_rand_seed = 82
    np.random.seed(np_rand_seed)

    # # **1. Load and Inspect the data**
    try:
        data = pd.read_json(os.path.join(strRootPath, 'train.json'))
        test_data = pd.read_json(os.path.join(strRootPath, 'test.json'))
    except Exception as e:
        print(e)

    print("Shape of train set:", data.shape)
    print("Shape of test set:", test_data.shape)

    print("Shape of band 1:",  np.shape(data.band_1.iloc[0]))
    print("Shape of band 2:",  np.shape(data.band_2.iloc[0]))

    print("Type of band 1:",  type(data.band_1.iloc[0]))
    print("Type of band 2:",  type(data.band_2.iloc[0]))

    # # **2. Feature Engineering**

    # ## **2.1 Feature engineering on train set**

    # ### **2.1.1 Replacing the na in inc_angle with mean - confirm this is happening**********

    data[data['inc_angle'] == 'na'] = data[data['inc_angle'] != 'na']['inc_angle'].mean()

    # ### **2.1.2 Converting the angle from degrees to radian******

    #data['inc_angle'] = data['inc_angle'].apply(lambda x: math.radians(x))

    data.inc_angle.head()

    # ### ** 2.1.3 Finding and dropping points with mismatch band1 and band2 data**

    # **Count and list of mismatched points in band1**
    missing_list1, count1 = find_missing_data(data.band_1, (5625,))
    print("count: ", count1)
    print("missing data: ", missing_list1)

    # **Count and list of mismatched points in band2**
    missing_list2, count2 = find_missing_data(data.band_2, (5625,))
    print("count: ", count1)
    print("missing data: ", missing_list2)

    # **Check if the missing points are same**
    missing_list1 == missing_list2

    # **Drop the points with mismatched images**
    data = drop_data(data, missing_list1)

    print(data.shape)
    print(data.head)

    print("Number of positive classes: ", len(data[data['is_iceberg'] == 1.0]))
    print("Number of negative classes: ", len(data[data['is_iceberg'] == 0.0]))

    # ### 2.1.4 Scale the image data
    # **We will use standardisation as the  normalization technique since this works well with images**
    data['band_1'] = data['band_1'].apply(standardise_vector)
    data['band_2'] = data['band_2'].apply(standardise_vector)

    print(data.head(5))

    # ### **2.1.5 Reshaping the band1 and band2 data into 2D image**
    band_1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in data["band_1"]])
    band_2 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in data["band_2"]])

    print("Shape of band 1 image:", band_1.shape)
    print("Shape of band 2 image:", band_2.shape)

    # ## **2.2 Feature engineering on test Set**
    # **We carry out the same feature engineering as carried out on train set**
    #test_data['inc_angle'] = test_data['inc_angle'].apply(lambda x: math.radians(x))

    test_data.inc_angle.head()

    missing_list3, count3 = find_missing_data(test_data.band_1, (5625,))
    print("count: ", count3)
    print("missing data: ", missing_list3)

    missing_list4, count4 = find_missing_data(test_data.band_2, (5625,))
    print("count: ", count4)
    print("missing data: ", missing_list4)

    test_data['band_1'] = test_data['band_1'].apply(standardise_vector)
    test_data['band_2'] = test_data['band_2'].apply(standardise_vector)

    band_1_test = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in test_data["band_1"]])
    band_2_test = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in test_data["band_2"]])

    print("Shape of test set band 1 image:", band_1_test.shape)
    print("Shape of test set band 2 image:", band_2_test.shape)

    # # **3. Train/test/validation split**
    # **Extract the labels and angles of train set**
    labels = data.is_iceberg.as_matrix()
    angles = data.inc_angle.as_matrix()

    # **Carry out splits**
    # randomly choosing the train and validation indices
    train_indices = np.random.choice(len(labels), round(len(labels)*0.75), replace=False)
    validation_indices = np.array(list(set(range(len(labels))) - set(train_indices)))

    # extract train set
    band_1_train = band_1[train_indices]
    band_2_train = band_2[train_indices]
    angles_train = angles[train_indices]
    labels_train = labels[train_indices]

    # extract validation set
    band_1_validation = band_1[validation_indices]
    band_2_validation = band_2[validation_indices]
    angles_validation = angles[validation_indices]
    labels_validation = labels[validation_indices]

    # extract test set
    band_1_test = band_1_test
    band_2_test = band_2_test
    angles_test = test_data.inc_angle.as_matrix()
    iD = test_data.id.as_matrix()

    # **Covert the types of all data to float**
    band_1_train = band_1_train.astype(np.float32)
    band_1_validation = band_1_validation.astype(np.float32)
    band_1_test = band_1_test.astype(np.float32)
    band_2_train = band_2_train.astype(np.float32)
    band_2_validation = band_2_validation.astype(np.float32)
    band_2_test = band_2_test.astype(np.float32)
    angles_train = angles_train.astype(np.float32)
    angles_validation = angles_validation.astype(np.float32)
    angles_test = angles_test.astype(np.float32)
    labels_train = labels_train.astype(np.float32)
    labels_validation = labels_validation.astype(np.float32)
    iD = iD.astype(np.str)

    # delete the unnecessary variables out of memory
    del(data, test_data, band_1, band_2)

    # **Examine the shape of the data**
    print("Shape of band_1_train:", band_1_train.shape)
    print("Shape of band_2_train:", band_2_train.shape)
    print("Shape of angles_train:", angles_train.shape)
    print("Shape of labels_train:", labels_train.shape)
    print("Shape of band_1_validation:", band_1_validation.shape)
    print("Shape of band_2_validation:", band_2_validation.shape)
    print("Shape of angles_validation:", angles_validation.shape)
    print("Shape of labels_validation:", labels_validation.shape)
    print("Shape of band_1_test:", band_1_test.shape)
    print("Shape of band_2_test:", band_2_test.shape)
    print("Shape of angles_test:", angles_test.shape)
    print("Shape of iD:", iD.shape)

    import matplotlib.pyplot as mplt

    # displaying original samples image

    print(labels_validation[1])
    print(angles_validation[1])
    print(labels_validation[6])
    print(angles_validation[6])

    mplt.figure(figsize=(16, 8), dpi=80, facecolor='w', edgecolor='k')
    image = band_1_validation[1].copy()
    mplt.subplot(3, 5, 1)
    mplt.title("Iceberg")
    mplt.imshow(image)

    mplt.figure(figsize=(16, 8), dpi=80, facecolor='w', edgecolor='k')
    image = band_1_validation[6].copy()
    mplt.subplot(3, 5, 1)
    mplt.title("Ship")
    mplt.imshow(image)

    #fig, axs = mplt.subplots(2, 1)
    #mplt.title("Iceberg")
    #axs[0, 0].imshow(band_1_validation[1].copy())
    #mplt.title("Ship")
    #axs[0, 1].imshow(band_1_validation[6].copy())

    try:
        # Flatten the 75x75 image back into its 5625
        band_1_train_flatten = np.array([np.array(b).astype(np.float32).flatten() for b in band_1_train])
        band_2_train_flatten = np.array([np.array(b).astype(np.float32).flatten() for b in band_2_train])
        band_1_valid_flatten = np.array([np.array(b).astype(np.float32).flatten() for b in band_1_validation])
        band_2_valid_flatten = np.array([np.array(b).astype(np.float32).flatten() for b in band_2_validation])
        band_1_test_flatten = np.array([np.array(b).astype(np.float32).flatten() for b in band_1_test])
        band_2_test_flatten = np.array([np.array(b).astype(np.float32).flatten() for b in band_2_test])

        # Add to a dataframe
        df_write_b1_train_csv = pd.DataFrame(band_1_train_flatten)
        df_write_b2_train_csv = pd.DataFrame(band_2_train_flatten)
        df_write_angles_train_csv = pd.DataFrame(angles_train)
        df_write_labels_train_csv = pd.DataFrame(labels_train)

        df_write_b1_valid_csv = pd.DataFrame(band_1_valid_flatten)
        df_write_b2_valid_csv = pd.DataFrame(band_2_valid_flatten)
        df_write_angles_valid_csv = pd.DataFrame(angles_validation)
        df_write_labels_valid_csv = pd.DataFrame(labels_validation)

        df_write_b1_test_csv = pd.DataFrame(band_1_test_flatten)
        df_write_b2_test_csv = pd.DataFrame(band_2_test_flatten)
        df_write_angles_test_csv = pd.DataFrame(angles_test)
        df_write_ID_csv = pd.DataFrame(iD)

        # write out to .csv store
        df_write_b1_train_csv.to_csv(os.path.join(train_data_dir, 'band_1_train.csv'), header=None)
        df_write_b2_train_csv.to_csv(os.path.join(train_data_dir, 'band_2_train.csv'), header=None)
        df_write_angles_train_csv.to_csv(os.path.join(train_data_dir, 'angles_train.csv'), header=None)
        df_write_labels_train_csv.to_csv(os.path.join(train_data_dir, 'labels_train.csv'), header=None)

        df_write_b1_valid_csv.to_csv(os.path.join(dev_data_dir, 'band_1_valid.csv'), header=None)
        df_write_b2_valid_csv.to_csv(os.path.join(dev_data_dir, 'band_2_valid.csv'), header=None)
        df_write_angles_valid_csv.to_csv(os.path.join(dev_data_dir, 'angles_valid.csv'), header=None)
        df_write_labels_valid_csv.to_csv(os.path.join(dev_data_dir, 'labels_valid.csv'), header=None)

        df_write_b1_test_csv.to_csv(os.path.join(test_data_dir, 'band_1_test.csv'), header=None)
        df_write_b2_test_csv.to_csv(os.path.join(test_data_dir, 'band_2_test.csv'), header=None)
        df_write_angles_test_csv.to_csv(os.path.join(test_data_dir, 'angles_test.csv'), header=None)
        df_write_ID_csv.to_csv(os.path.join(test_data_dir, 'ID.csv'), header=None)

    except Exception as e:
        print(e)
