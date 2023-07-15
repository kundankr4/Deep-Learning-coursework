# import tensorflow
import os, tensorflow as tf, tensorflow_datasets as TFDatasets
from keras import callbacks, layers, losses, metrics, optimizers, Input, Model
from tensorflow._api.v2.data import Dataset

# load mnist dataset
dataset_dir = os.path.normpath("~/Downloads/Datasets/")
dataset = TFDatasets.load("mnist", data_dir=dataset_dir, as_supervised=True)
training_dataset, testing_dataset = dataset['train'], dataset['test'] # type: ignore
training_dataset: Dataset
testing_dataset: Dataset
training_dataset = training_dataset.map(lambda image, l: (image / 255, l)).batch(128) # must have preprocess mapping
testing_dataset = testing_dataset.map(lambda image, l: (image / 255, l)).batch(128) # must have preprocess mapping

# layers
conv1 = layers.Conv2D(32, (3,3), strides=(1,1), padding='same', activation=tf.nn.relu, name='conv1')
maxpool2 = layers.MaxPool2D(name='maxpool2')
conv3 = layers.Conv2D(64, (3,3), strides=(1,1), padding='same', activation=tf.nn.relu, name='conv3')
maxpool4 = layers.MaxPool2D(name='maxpool4')
conv5 = layers.Conv2D(128, (3,3), strides=(1,1), padding='same', activation=tf.nn.relu, name='conv5')
maxpool6 = layers.MaxPool2D(name='maxpool6')
flatten = layers.Flatten(name='flatten')
dense1 = layers.Dense(1024, activation=tf.nn.relu, name='dense1')
dense2 = layers.Dense(256, activation=tf.nn.relu, name='dense2')
classification = layers.Dense(10, activation=tf.nn.softmax, name='classification')

# logic
input_img = Input((28,28,1))
x = conv1(input_img)
x = maxpool2(x)
x = conv3(x)
x = maxpool4(x)
x = conv5(x)
x = maxpool6(x)
x = flatten(x)
x = dense1(x)
x = dense2(x)
y = classification(x)

# create model instance
model = Model(inputs=input_img, outputs=y, name="CNN")

# compile model
optimizer = optimizers.Adam(0.0001)
loss_fn = losses.SparseCategoricalCrossentropy(name='loss')
acc_fn = metrics.SparseCategoricalAccuracy(name='accuracy')
model.compile(optimizer, loss_fn, [acc_fn])

# train model
tensorboard_callback = callbacks.TensorBoard()
model.fit(training_dataset, epochs=10, callbacks=[tensorboard_callback], validation_data=testing_dataset)
model.evaluate(testing_dataset)

# save model
model.save('model')
