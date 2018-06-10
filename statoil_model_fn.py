"""Define the model."""
import time
t1 = time.time()

import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as mplt
import math

from tensorflow.python.framework import ops


def create_weights(shape):
    """a function to create weight tensor"""
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))


def create_biases(size):
    """a function to create bias tensor"""
    return tf.Variable(tf.constant(0.05, shape=[size]))


def create_convolutional_layer(input,
                               num_input_channels,
                               conv_filter_size,
                               max_pool_filter_size,
                               num_filters,
                               keep_prob):
    """a function to create convoutional layer"""

    # create filter for the convolutional layer
    weights = create_weights(shape=[conv_filter_size, conv_filter_size, num_input_channels, num_filters])

    # create biases
    biases = create_biases(num_filters)

    # create covolutional layer
    layer = tf.nn.conv2d(input=input, filter=weights, strides=[1, 1, 1, 1], padding='SAME')

    # add the bias to the convolutional layer
    layer += biases

    # relu activation layer fed into layer ** Changed from relu **
    layer = tf.nn.relu(layer)

    if keep_prob > 0.0:
        layer = tf.nn.dropout(layer, keep_prob)

    # max pooling to half the size of the image
    layer = tf.nn.max_pool(value=layer,
                           ksize=[1, max_pool_filter_size, max_pool_filter_size, 1],
                           strides=[1, 2, 2, 1],
                           padding='SAME')

    # return the output layer of the convolution
    return layer


def create_flatten_layer(layer):
    """a function for creating flattened layer from convolutional output"""

    # extract the shape of the layer
    layer_shape = layer.get_shape()
    # calculate the number features of the flattened layer
    num_features = layer_shape[1:4].num_elements()
    # create the flattened layer
    layer = tf.reshape(layer, [-1, num_features])
    # return the layer
    return layer


def create_fc_layer(input,
                    num_inputs,
                    num_outputs,
                    use_relu=True,
                    dropout=False,
                    keep_prob=0.2):
    """a function for creating fully connected layer"""

    # Let's define trainable weights and biases.
    weights = create_weights(shape=[num_inputs, num_outputs])
    biases = create_biases(num_outputs)

    # matrix multiplication between input and weight matrix
    layer = tf.matmul(input, weights) + biases

    # add relu activation if wanted
    if use_relu:
        layer = tf.nn.relu(layer)

    # if dropout is wanted add dropout
    if dropout:
        layer = tf.nn.dropout(layer, keep_prob)

    # return layer
    return layer


def random_mini_batches(x, y, mini_batch_size=64, seed=0):

    m = y.shape[0]  # number of training examples
    mini_batches = []
    np.random.seed(seed)

    # Step 1: Shuffle (X, Y)
    permutation = np.random.permutation(m)

    shuffled_x = x[permutation, :]
    shuffled_y = y[permutation, :]

    # Step 2: Partition (shuffled_x, shuffled_y). Minus the end case.
    num_complete_minibatches = math.floor(m / mini_batch_size)  # number of mini batches of size mini_batch

    for k in range(0, num_complete_minibatches):

        mini_batch_x = shuffled_x[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        mini_batch_y = shuffled_y[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        mini_batch = (mini_batch_x, mini_batch_y)

        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_x = shuffled_x[num_complete_minibatches * mini_batch_size: m, :]
        mini_batch_y = shuffled_y[num_complete_minibatches * mini_batch_size: m, :]
        mini_batch = (mini_batch_x, mini_batch_y)

        mini_batches.append(mini_batch)

    return mini_batches


def show_images(img_original, saliency, max_class, title):
    # get out the first map and class from the mini-batch
    saliency = saliency[0]
    max_class = max_class[0]
    # convert saliency from BGR to RGB, and from c01 to 01c
    saliency = saliency[::-1].transpose(1, 2, 0)
    # plot the original image and the three saliency map variants
    mplt.figure(figsize=(10, 10), facecolor='w')
    mplt.suptitle("Class: " + ". Saliency: " + title)
    mplt.subplot(2, 2, 1)
    mplt.title('input')
    mplt.imshow(img_original)
    mplt.subplot(2, 2, 2)
    mplt.title('abs. saliency')
    mplt.imshow(np.abs(saliency).max(axis=-1), cmap='gray')
    mplt.subplot(2, 2, 3)
    mplt.title('pos. saliency')
    mplt.imshow((np.maximum(0, saliency) / saliency.max()))
    mplt.subplot(2, 2, 4)
    mplt.title('neg. saliency')
    mplt.imshow((np.maximum(0, -saliency) / -saliency.min()))
    mplt.show()


def compute_and_diplay_saliency_map(layer_unit_1):

    try:

        fig, axs = mplt.subplots(2, 2)
        axs[0, 0].imshow(layer_unit_1[0, :, :, 0], cmap="hot")
        axs[0, 1].imshow(layer_unit_1[1, :, :, 0], cmap="hot")
        axs[1, 0].imshow(layer_unit_1[2, :, :, 0], cmap="hot")
        axs[1, 1].imshow(layer_unit_1[3, :, :, 0], cmap="hot")

        mplt.subplot_tool()
        mplt.show()

    except Exception as e:
        print(e)


def statoil_model_fn(model_inputs, model_outputs, params, model_num, make_prediction=False):
    """Model function defining the graph operations."""

    image_train = model_inputs['image_train']
    image_validation = model_inputs['image_validation']
    image_test = model_inputs['image_test']
    labels_train = model_inputs["labels_train"]
    labels_validation = model_inputs["labels_validation"]
    iD = model_inputs["iD"]

    mplt.figure(figsize=(16, 8), dpi=80, facecolor='w', edgecolor='k')
    image = image_validation[1].copy()
    mplt.subplot(3, 5, 1)
    mplt.title("Iceberg")
    mplt.imshow(image)

    mplt.figure(figsize=(16, 8), dpi=80, facecolor='w', edgecolor='k')
    image = image_validation[6].copy()
    mplt.subplot(3, 5, 1)
    mplt.title("Ship")
    mplt.imshow(image)

    ops.reset_default_graph()

    np_rand_seed = 97
    tf_rand_seed = 82
    np.random.seed(np_rand_seed)
    tf.set_random_seed(tf_rand_seed)

    # **** Create the Convolutional Neural Network ****

    # ## **6.1 One hot encoding labels**
    labels_train = pd.get_dummies(labels_train).as_matrix()
    labels_validation = pd.get_dummies(labels_validation).as_matrix()

    print('**** Shape of labels ****')
    print("Shape of labels_train:", labels_train.shape)
    print("Shape of labels_validation:", labels_validation.shape)

    # ## **6.2 Create placeholders**
    # image dimensions
    width = 75
    height = 75
    num_channels = 3
    flat = width * height
    num_classes = 2

    # **Create placeholder for image, labels,  dropout keep probability, and optionally angle**
    image = tf.placeholder(tf.float32, shape=[None, height, width, num_channels])
    # angle = tf.placeholder(tf.float32, shape= [None, 1])
    y_true = tf.placeholder(tf.int32, shape=[None, num_classes])
    keep_prob = tf.placeholder(tf.float32)

    # ## **6.4 Create Layers of Covnet**

    m_num = str(model_num + 1)

    num_of_conv_layers = params.dict['m' + m_num + '_num_of_conv_layers']
    num_of_fc_layers = params.dict['m' + m_num + '_num_of_fc_layers']
    layer_keep_prob = params.dict['m' + m_num + '_layer_keep_prob']

    if num_of_conv_layers >= 1:
        conv1_features = params.dict['m' + m_num + '_conv1_features']
        conv1_filter_size = params.dict['m' + m_num + '_conv1_filter_size']
        max_pool_size1 = params.dict['m' + m_num + '_max_pool_size1']

    if num_of_conv_layers >= 2:
        conv2_features = params.dict['m' + m_num + '_conv2_features']
        conv2_filter_size = params.dict['m' + m_num + '_conv2_filter_size']
        max_pool_size2 = params.dict['m' + m_num + '_max_pool_size2']

    if num_of_conv_layers >= 3:
        conv3_features = params.dict['m' + m_num + '_conv3_features']
        conv3_filter_size = params.dict['m' + m_num + '_conv3_filter_size']
        max_pool_size3 = params.dict['m' + m_num + '_max_pool_size3']

    if num_of_conv_layers >= 4:
        conv4_features = params.dict['m' + m_num + '_conv4_features']
        conv4_filter_size = params.dict['m' + m_num + '_conv4_filter_size']
        max_pool_size4 = params.dict['m' + m_num + '_max_pool_size4']

    if num_of_conv_layers >= 5:
        conv5_features = params.dict['m' + m_num + '_conv5_features']
        conv5_filter_size = params.dict['m' + m_num + '_conv5_filter_size']
        max_pool_size5 = params.dict['m' + m_num + '_max_pool_size5']

    if num_of_conv_layers >= 6:
        conv6_features = params.dict['m' + m_num + '_conv6_features']
        conv6_filter_size = params.dict['m' + m_num + '_conv6_filter_size']
        max_pool_size6 = params.dict['m' + m_num + '_max_pool_size6']

    if num_of_fc_layers >= 1:
        fc_layer_size1 = params.dict['m' + m_num + '_fc_layer_size1']

    if num_of_fc_layers >= 2:
        fc_layer_size2 = params.dict['m' + m_num + '_fc_layer_size2']

    if num_of_conv_layers >= 1:
        layer_conv1 = create_convolutional_layer(input=image,
                                                 num_input_channels= num_channels,
                                                 conv_filter_size = conv1_filter_size,
                                                 max_pool_filter_size = max_pool_size1,
                                                 num_filters = conv1_features,
                                                 keep_prob = layer_keep_prob)

        conv_to_layer_flat = layer_conv1

    if num_of_conv_layers >= 2:
        layer_conv2 = create_convolutional_layer(input=layer_conv1,
                                                 num_input_channels= conv1_features,
                                                 conv_filter_size = conv2_filter_size,
                                                 max_pool_filter_size = max_pool_size2,
                                                 num_filters = conv2_features,
                                                 keep_prob = layer_keep_prob)

        conv_to_layer_flat = layer_conv2

    if num_of_conv_layers >= 3:
        layer_conv3 = create_convolutional_layer(input=layer_conv2,
                                                 num_input_channels= conv2_features,
                                                 conv_filter_size = conv3_filter_size,
                                                 max_pool_filter_size = max_pool_size3,
                                                 num_filters = conv3_features,
                                                 keep_prob=layer_keep_prob)

        conv_to_layer_flat = layer_conv3

    if num_of_conv_layers >= 4:
        layer_conv4 = create_convolutional_layer(input=layer_conv3,
                                                 num_input_channels= conv3_features,
                                                 conv_filter_size = conv4_filter_size,
                                                 max_pool_filter_size = max_pool_size4,
                                                 num_filters = conv4_features,
                                                 keep_prob=layer_keep_prob)

        conv_to_layer_flat = layer_conv4

    if num_of_conv_layers >= 5:
        layer_conv5 = create_convolutional_layer(input=layer_conv4,
                                                 num_input_channels= conv4_features,
                                                 conv_filter_size = conv5_filter_size,
                                                 max_pool_filter_size = max_pool_size5,
                                                 num_filters = conv5_features,
                                                 keep_prob=layer_keep_prob)

        conv_to_layer_flat = layer_conv5

    if num_of_conv_layers >= 6:
        layer_conv6 = create_convolutional_layer(input=layer_conv5,
                                                 num_input_channels= conv5_features,
                                                 conv_filter_size = conv6_filter_size,
                                                 max_pool_filter_size = max_pool_size6,
                                                 num_filters = conv6_features,
                                                 keep_prob = layer_keep_prob)

        conv_to_layer_flat = layer_conv6

    layer_flat = create_flatten_layer(conv_to_layer_flat)

    if num_of_fc_layers >= 1:
        layer_fc1 = create_fc_layer(input=layer_flat,
                                    num_inputs=layer_flat.get_shape()[1:4].num_elements(),
                                    num_outputs=fc_layer_size1,
                                    use_relu=True,
                                    dropout =True,
                                    keep_prob = keep_prob)

        fc_layer_size = fc_layer_size1
        fc_to_output_layer = layer_fc1

    if num_of_fc_layers >= 2:
        layer_fc2 = create_fc_layer(input=layer_fc1,
                                    num_inputs=fc_layer_size1,
                                    num_outputs=fc_layer_size2,
                                    use_relu=True,
                                    dropout =True,
                                    keep_prob = keep_prob)

        fc_layer_size = fc_layer_size2
        fc_to_output_layer = layer_fc2

    # **Create the output layer**
    output_layer = create_fc_layer(input=fc_to_output_layer,
                         num_inputs = fc_layer_size,
                         num_outputs = num_classes,
                         use_relu=False)

    # ## **6.5 Create prediction & accuracy metric**
    # softmax operation on the output layer
    y_pred = tf.nn.softmax(output_layer)
    # extract the vector of predicted class
    y_pred_cls = tf.argmax(y_pred, axis=1, output_type=tf.int32)
    # extract the vector of labels
    y_true_cls = tf.argmax(y_true, axis=1, output_type=tf.int32)

    # extract the vector of correct prediction
    correct_prediction = tf.equal(y_pred_cls, y_true_cls)
    # operation to calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # ## **6.6 Create Optimizer**
    # operation to calculate cross entropy
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=output_layer, labels=y_true)

    print(output_layer)

    # mean of cross entropy to act as the loss
    loss = tf.reduce_mean(cross_entropy)

    # learning rate of optimizer
    learning_rate = params.dict['m' + str(model_num + 1) + '_learning_rate']
    keep_prob_param = params.dict['m' + str(model_num + 1) + '_keep_prob']

    # train step
    train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

    # # **7. Train Model**
    # lists to store the train loss, validation loss, validation accuracy at each iteration
    train_loss = []
    valid_loss = []
    valid_acc = []

    # batch size
    batch_size = 256

    # create a saver object
    saver = tf.train.Saver(max_to_keep=1)

    # variables to store the accuracy, loss, iteration of our best model
    best_accuracy = model_outputs['best_valid_accuracy']
    best_loss = model_outputs['best_validation_loss']
    best_train_loss = model_outputs['best_train_loss']
    best_iteration = model_outputs['best_iteration']
    best_model_num = model_outputs['best_model_num']

    num_epochs = 50
    seed = 3
    iteration = 0
    validation_prediction = None
    t_train_start = time.time()

    # create a graph session and optimize under it
    with tf.Session() as sess:
        # initialize variables
        sess.run(tf.global_variables_initializer())

        for epoch in range(num_epochs):

            epoch_cost = 0.  # Defines a cost related to an epoch
            num_minibatches = int(labels_train.shape[0] / batch_size)

            seed = seed + 1
            minibatches = random_mini_batches(image_train, labels_train, batch_size, seed)

            i = 0

            for minibatch in minibatches:
                # Select a minibatch
                (image_train_batch, labels_train_batch) = minibatch

                #  feed dictionary for batch
                feed_dict_batch = {image: image_train_batch, y_true: labels_train_batch, keep_prob: keep_prob_param}

                # execute optimization step
                sess.run(train_step, feed_dict=feed_dict_batch)

                # calculate temporary train loss and append it to the designated list
                temp_train_loss = loss.eval(session=sess, feed_dict=feed_dict_batch)

                epoch_cost += temp_train_loss / num_minibatches

                i += 1
                # print("Cost after mini batch %i: %f" % (i, epoch_cost))

                print(' **** Saliency Mapping****')
                units_1 = sess.run(layer_conv1, feed_dict={image: image_validation, keep_prob: 1.0})
                #units_2 = sess.run(layer_conv2, feed_dict={image: image_validation, keep_prob: 1.0})
                #units_3 = sess.run(layer_conv3, feed_dict={image: image_validation, keep_prob: 1.0})
                #units_4 = sess.run(layer_conv4, feed_dict={image: image_validation, keep_prob: 1.0})

                compute_and_diplay_saliency_map(units_1)

            # Print the cost after every epoch
            print("Cost after epoch %i: %f" % (epoch, epoch_cost))
            train_loss.append(epoch_cost)

            # feed dictionary for validation
            feed_dict_validation = {image: image_validation, y_true: labels_validation, keep_prob: 1.0}

            # calculate temporary validation loss and append it to the designated list
            temp_validation_loss = loss.eval(session=sess, feed_dict=feed_dict_validation)
            valid_loss.append(temp_validation_loss)

            # calculate temporary validation accuracy and append it to the designated list
            temp_validation_accuracy = accuracy.eval(session=sess, feed_dict=feed_dict_validation)
            valid_acc.append(temp_validation_accuracy)

            # if valid loss is better than best recorded update and save the model
            if temp_validation_loss < best_loss:
                print('**** current best epoch ****')
                best_loss = temp_validation_loss
                best_train_loss = temp_train_loss
                best_accuracy = temp_validation_accuracy
                best_iteration = epoch
                best_model_num = model_num + 1
                saver.save(sess, './my-model', global_step=best_iteration)

                # **** For Debugging - store the correct prediction set for the best case ****
                validation_prediction = correct_prediction.eval(session=sess, feed_dict=feed_dict_validation)

            # print metric info
            print("Epoch:", epoch, "| train_loss:", temp_train_loss, "| validation_loss:", temp_validation_loss,
                  "| valid_accuracy:", temp_validation_accuracy)

            units_1 = sess.run(layer_conv1, feed_dict=feed_dict_batch)
            units_2 = sess.run(layer_conv2, feed_dict=feed_dict_batch)
            units_3 = sess.run(layer_conv3, feed_dict=feed_dict_batch)
            units_4 = sess.run(layer_conv4, feed_dict=feed_dict_batch)

            compute_and_diplay_saliency_map(units_1, units_2, units_3, units_4)


    # delete unnecessary variables out of memory
    # del (image_train, image_validation, angles_train, angles_validation, labels_train, labels_validation)
    del (image_train, image_validation, labels_train, labels_validation)

    t_train_end = time.time()
    print("time take for training model number: " + str(model_num), t_train_end - t_train_start)

    if validation_prediction is not None:
        try:
            validation_prediction = validation_prediction.tolist()
            validation_prediction_csv = pd.DataFrame(validation_prediction)
            validation_prediction_csv.to_csv('data\\dev\\validation_prediction.csv', header=None)
        except Exception as e:
            print(e)

    model_outputs['best_iteration'] = best_iteration
    model_outputs['best_train_loss'] = best_train_loss
    model_outputs['best_validation_loss'] = best_loss
    model_outputs['best_valid_accuracy'] = best_accuracy
    model_outputs['model_num'] = best_model_num
    model_outputs['valid_loss'] = valid_loss

    if make_prediction:
        print('make prediction')
        t_pred_start = time.time()

        with tf.Session() as sess:
            # restore the best model
            model_path = "./" + "my-model-" + str(best_iteration)
            saver.restore(sess, model_path)

            # break the test set into k folds other wise kernel will be out of memory
            n = len(iD)
            k = 12
            step = n // k

            # array to store the prediction
            preds = np.array([])

            # iterate through each fold
            for i in range(k):
                # start and end indices of the fold
                start = (step * i)
                end = (step * (i + 1))

                # feed dictionary for the fold
                feed_dict_test = {image: image_test[start:end],
                                  #                            angle: np.transpose([angles_test[start:end]]),
                                  keep_prob: 1.0}

                # evaluate predictions of the fold
                fold_preds = y_pred.eval(session=sess, feed_dict=feed_dict_test)[:, 1]
                # append the predictions of the fold to the designated array
                preds = np.append(preds, fold_preds)

            # save the submission csv file
            submission_path = "./submission.csv"
            submission = pd.DataFrame({"id": iD, "is_iceberg": preds})
            submission.to_csv(submission_path, header=True, index=False)

            # save the csv file containing performance metrics of the best model
            results = pd.DataFrame([int(best_iteration), best_train_loss, best_loss, best_accuracy, best_model_num],
                                   index=["epoch", "train loss", "valid loss", "accuracy", "model number"],
                                   columns=["results"])
            results_path = "./results.csv"
            results.to_csv(results_path, header=True, index=True)

        t_pred_end = time.time()
        print("time taken for prediction: ", t_pred_end - t_pred_start)

        # # **9. Visualization of the performance**
        # ## **9.1 Plot of loss over iteration**

        mplt.figure(figsize=(16, 8), dpi=80, facecolor='w', edgecolor='k')
        mplt.plot(num_epochs, valid_loss, label="valid loss " + str(model_num + 1))
        # mplt.plot(iterations, valid_loss_values[2], label="valid loss 2")
        # mplt.plot(iterations, valid_loss_values[1], label="valid loss 1")
        mplt.title("Loss")
        mplt.xlabel("iter")
        mplt.ylabel("loss")
        mplt.legend()
        mplt.grid()
        mplt.show()

        mplt.figure(figsize=(16, 8), dpi=80, facecolor='w', edgecolor='k')
        mplt.plot(num_epochs, train_loss, label="train loss")
        mplt.plot(num_epochs, valid_loss, label="valid loss")
        mplt.title("Loss")
        mplt.xlabel("iter")
        mplt.ylabel("loss")
        mplt.legend()
        mplt.grid()
        mplt.show()

        # ## **9.2 Plot of training accuracy over iteration**

        mplt.figure(figsize=(16, 8), dpi=80, facecolor='w', edgecolor='k')
        mplt.plot(num_epochs, valid_acc, label="train loss")
        mplt.title("Accuracy")
        mplt.xlabel("iter")
        mplt.ylabel("accuracy")
        mplt.grid()
        mplt.show()
    
    print("pause")

    return model_outputs
