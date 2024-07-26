import numpy as np
import os

import tensorflow as tf
from tensorflow import keras

from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras import optimizers
from tensorflow.keras import models
import tensorflow.keras.backend as K


def combine_groups(groupN, groupM, similarityLabel):
    """
    The function `combine_groups` takes two groups and a similarity label as input, and returns all
    possible combinations of elements from the two groups along with the similarity label assigned to
    each combination.

    :param groupN: groupN is a list representing the first group of items
    :param groupM: groupM is a list representing one group of items
    :param similarityLabel: The similarity label is a value that represents the similarity between the
    elements in the combined groups. It could be a numerical value, a string, or any other type of data
    that represents the similarity
    :return: two lists: `combinations` and `similarityLabels`.
    """
    combinations = []
    for i in range(len(groupN)):
        for j in range(len(groupM)):
            combinations.append([groupN[i], groupM[j]])
    similarityLabels = np.full(len(combinations), similarityLabel)

    return list(combinations), list(similarityLabels)


def extend_input_new_comb(groupM, groupN, similarityLabel, input, output):
    """
    The function extends the input and output lists by combining groups M and N based on a similarity
    label.

    :param groupM: A list representing one group of inputs
    :param groupN: The `groupN` parameter represents one of the groups that you want to combine with
    `groupM`. It is a list or array containing elements that you want to combine with the elements in
    `groupM`
    :param similarityLabel: The similarityLabel parameter is a label that indicates the similarity
    between the groups groupM and groupN
    :param input: A list of input data
    :param output: The `output` parameter is a list that contains the outputs corresponding to the
    inputs in the `input` list
    :return: the updated input and output lists.
    """
    newInputs, newOutputs = combine_groups(groupM, groupN, similarityLabel)
    input.extend(newInputs)
    output.extend(newOutputs)

    return input, output


def euclidean_distance(vectors):
    # unpack the vectors into separate lists
    (featsA, featsB) = vectors
    # compute the sum of squared distances between the vectors
    sumSquared = K.sum(K.square(featsA - featsB), axis=1, keepdims=True)
    # return the euclidean distance between the vectors
    return K.sqrt(K.maximum(sumSquared, K.epsilon()))


def randomize_and_split(inputs):
    """
    The function `randomize_and_split` takes a list of inputs, shuffles them randomly, and splits them
    into training and testing sets based on a specified ratio.

    Args:
      inputs: Waveform data for training and testing the model.

    Returns:
      The function `randomize_and_split` returns two outputs: `trainFunctions` and `testFunctions`,
    which are the randomly shuffled and split input `inputs`. The `trainFunctions` contain 85% of the
    input functions for training, while the `testFunctions` contain the remaining 15% for testing.
    """
    p = np.random.permutation(len(inputs))

    functions = inputs[p]

    trainSplitSample = int(0.85 * len(functions))
    trainFunctions = functions[:trainSplitSample]
    testFunctions = functions[trainSplitSample:]

    return trainFunctions, testFunctions


def loss(margin=1):
    """Provides 'contrastive_loss' an enclosing scope with variable 'margin'.

    Arguments:
        margin: Integer, defines the baseline for distance for which pairs
                should be classified as dissimilar. - (default is 1).

    Returns:
        'contrastive_loss' function with data ('margin') attached.
    """

    # Contrastive loss = mean( (1-true_value) * square(prediction) +
    #                         true_value * square( max(margin-prediction, 0) ))
    def contrastive_loss(y_true, y_pred):
        """Calculates the contrastive loss.

        Arguments:
            y_true: List of labels, each label is of type float32.
            y_pred: List of predictions of same length as of y_true,
                    each label is of type float32.

        Returns:
            A tensor containing contrastive loss as floating point value.
        """

        square_pred = tf.math.square(y_pred)
        margin_square = tf.math.square(tf.math.maximum(margin - (y_pred), 0))
        return tf.math.reduce_mean(
            (1 - y_true) * square_pred + (y_true) * margin_square
        )

    return contrastive_loss


def loadArrays(path):
    """
    The function `loadArrays` loads arrays from files in a given path, performs some preprocessing on
    the arrays, and returns a numpy array containing all the loaded arrays.

    Args:
      path: The `path` parameter is the directory path where the files are located.

    Returns:
      a numpy array called totalArr.
    """
    files = os.listdir(path)
    totalArr = []
    for file in files:
        loadedArr = np.load(path + file)
        if len(loadedArr) == 14600:
            totalArr.append(loadedArr)
        else:
            loadedArr = loadedArr[200:14800] / np.max(np.abs(loadedArr[200:14800]))
            if len(loadedArr) == 14600:
                totalArr.append(loadedArr)

    return np.array(totalArr)


def scheduler(epoch, lr):
    """
    The scheduler function returns a learning rate of 0.001 if the epoch is greater than 0 and the
    learning rate is greater than 0.05, otherwise it returns the original learning rate.
    The purpose is to avoid the minimum overshooting with a large learning rate (which is necessary to converge at the beginning).

    :param epoch: The epoch is an integer value that represents the number of epochs that have passed
    :param lr: The learning rate (lr) is a hyperparameter that determines the step size at each
    iteration while moving towards a minimum of the loss function. It controls how much the model
    parameters are adjusted during training
    :return: either 0.01 or the value of lr, depending on the conditions specified.
    """
    if epoch > 0 and lr > 0.05:
        return 0.01
    else:
        return lr


def combine_two_groups(group1, group2):
    """
    The function combines two groups of inputs and their similarities, extends the inputs with
    additional combinations, shuffles the inputs and similarities randomly, and returns the shuffled
    inputs and similarities.

    Args:
      group1: The `group1` parameter represents the first group of inputs. It could be a list or array
    containing a set of input data.
      group2: The `group2` parameter represents the second group of inputs that you want to combine with
    `group1`. It is assumed that both `group1` and `group2` are lists or arrays containing input data.

    Returns:
      two arrays: `inputs[p]` and `similarities[p]`.
    """
    inputs, similarities = combine_groups(group1, group2, 1)
    inputs, similarities = extend_input_new_comb(
        group1, group1, 0, inputs, similarities
    )
    inputs, similarities = extend_input_new_comb(
        group2, group2, 0, inputs, similarities
    )

    inputs = np.array(inputs)
    inputs = np.transpose(inputs, (0, 2, 1))
    similarities = np.array(similarities, dtype="float32")

    p = np.random.permutation(len(inputs))

    return inputs[p], similarities[p]


def CNN_model():
    """
    The `CNN_model` function defines a convolutional neural network model with specific configurations
    and pretrained weights loaded from a saved model file.

    Returns:
      The function `CNN_model` returns a convolutional neural network (CNN) model with specific
    configurations for layers, activation functions, regularization constants, and weights initialized
    from a pre-trained model. The model architecture includes convolutional layers, dropout layers,
    global average pooling, and a dense layer with a specified embedding dimension. The model is then
    built and returned to the calling function.
    """
    activation_func = "elu"
    reg_const = 0.0001
    embeddingDim = 48

    preTrainedModel = models.load_model(
        "FirstWorkingModel.keras",
        custom_objects={"contrastive_loss": loss(margin=1)},
    ).layers[2]
    conv1Weights = preTrainedModel.layers[1].get_weights()
    conv2Weights = preTrainedModel.layers[3].get_weights()
    denseWeights = preTrainedModel.layers[5].get_weights()

    input = tf.keras.layers.Input(shape=(None, 1))

    conv1 = layers.Conv1D(
        32,
        kernel_size=4,
        strides=2,
        activation=activation_func,
        trainable=True,
        weights=conv1Weights,
        kernel_regularizer=regularizers.l2(reg_const),
    )(input)
    conv1 = layers.Dropout(0.3)(conv1)
    conv1 = layers.Conv1D(
        64,
        kernel_size=4,
        strides=2,
        trainable=True,
        weights=conv2Weights,
        activation=activation_func,
        kernel_regularizer=regularizers.l2(reg_const),
    )(conv1)

    conv1 = layers.GlobalAveragePooling1D()(conv1)

    outputs = layers.Dense(embeddingDim, trainable=True, weights=denseWeights)(conv1)
    # build the model
    model = models.Model(input, outputs)
    # return the model to the calling function
    return model


def similarity_model():
    """
    The `similarity_model` function defines a neural network model for calculating similarity between
    two inputs using a pre-trained model and custom loss function.

    Returns:
      The `similarity_model` function is returning a Siamese neural network model for similarity
    comparison. The model takes two inputs, `wf1` and `wf2`, which are passed through a shared feature
    extractor (CNN_model) to extract features. These features are then compared using a lambda layer to
    calculate the Euclidean distance between them. The distance is passed through a dense layer with
    weights initialized
    """
    preTrainedModel = models.load_model(
        "FirstWorkingModel.keras",
        custom_objects={"contrastive_loss": loss(margin=1)},
    )

    denseWeights = preTrainedModel.layers[4].get_weights()

    wf1 = layers.Input(shape=(None, 1))
    wf2 = layers.Input(shape=(None, 1))
    featureExtractor = CNN_model()
    featsA = featureExtractor(wf1)
    featsB = featureExtractor(wf2)

    distance = layers.Lambda(euclidean_distance)([featsA, featsB])
    distance = layers.Dense(
        1, activation="sigmoid", trainable=True, weights=denseWeights
    )(distance)

    model = models.Model(inputs=[wf1, wf2], outputs=distance)
    rmsprop = optimizers.Adam(learning_rate=0.1)
    model.compile(loss=loss(margin=1), optimizer=rmsprop, metrics=["accuracy"])

    return model


# The code is first defining two file paths for loading data - one for "lahar" files and one
# for "noise" files. It then calls a function `loadArrays()` to load the data from these files into
# the variables `laharData` and `noiseData` respectively.
laharFilesPath = "waveformData/lahars/"
laharData = loadArrays(laharFilesPath)

noiseFilesPath = "waveformData/noise/"
noiseData = loadArrays(noiseFilesPath)

# Splitting the data into training and testing sets, and randomizing their order
trainLahars, testLahars = randomize_and_split(laharData)
trainNoise, testNoise = randomize_and_split(noiseData)

# Combining two groups of data, `trainLahars` and `trainNoise`, into `trainInputs`
# and `trainSimilarities`. Similarly, it is combining two other groups of data, `testLahars` and
# `testNoise`, into `testInputs` and `testSimilarities`.
trainInputs, trainSimilarities = combine_two_groups(trainLahars, trainNoise)
testInputs, testSimilarities = combine_two_groups(testLahars, testNoise)

# Creating an instance of a similarity model.
model = similarity_model()

# Setting up an early stopping callback in Keras. This callback will monitor the
# validation accuracy during training and stop the training process if the validation accuracy does
# not improve for 10 consecutive epochs. The `restore_best_weights=True` argument ensures that the
# model's weights are restored to the best achieved during training before stopping.
earlystop = keras.callbacks.EarlyStopping(
    patience=10, monitor="val_accuracy", restore_best_weights=True
)

# Creating a callback function `reduce_lr` used during training to reduce the learning
# rate when a monitored metric, in this case, "val_accuracy", has stopped improving.
reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor="val_accuracy", factor=0.2, patience=4, min_lr=0.0000001
)
# Ceating a learning rate scheduler callback in TensorFlow/Keras. This callback
# will adjust the learning rate during training according to the specified scheduler function. The
# `scheduler` function should take the current epoch and current learning rate as inputs and return a
# new learning rate. This allows for dynamic learning rate adjustments during training.
lrScheduler = tf.keras.callbacks.LearningRateScheduler(scheduler)

# Training the model
history = model.fit(
    [trainInputs[:, :, 0], trainInputs[:, :, 1]],
    trainSimilarities[:],
    validation_data=([testInputs[:, :, 0], testInputs[:, :, 1]], testSimilarities[:]),
    batch_size=32,
    callbacks=[earlystop, reduce_lr, lrScheduler],
    epochs=100,
)

print("Training Accuracy")
print(history.history["accuracy"][-1])
print("Validation Accuracy")
print(history.history["val_accuracy"][-1])
