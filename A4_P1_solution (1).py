# import tensorflow
import tensorflow as tf
from tensorflow.keras import *

# import tensorflow datasets
import tensorflow_datasets as TFDatasets

"""
-----------------------------------------------------------------------------------------------
Step 1: Load dataset spoken_digit [30 points]
Load dataset from Tensorflow Datasets;
Set batch size to 64;
Preprocess the dataset to get spectogram by applying the short-time Fourier transform; 
Split the dataset with 70% of training, 10% of validation, and 20% of testing;
-----------------------------------------------------------------------------------------------
"""

# load mnist dataset
dataset, info = TFDatasets.load("spoken_digit", data_dir="~/Downloads/Datasets/", as_supervised=True, with_info=True)
num_classes: int = info.features['label'].num_classes

# calculate dataset size
dataset_size: int = info.splits['train'].num_examples
training_dataset_size = int(dataset_size * 0.7)
validation_dataset_size = int(dataset_size * 0.1)
testing_dataset_size = dataset_size - training_dataset_size - validation_dataset_size
steps_per_epoch = int(training_dataset_size / 64)

def stft(waveform: tf.Tensor, label: int) -> tuple[tf.Tensor, tf.Tensor]:
    '''
    Fourier transform with zero padding

    - Parameters:
        - waveform: A `Tensor` of waveform
        - label: An `int` of label
    - Returns: A `tuple` of spectrogram in `Tensor` and onehot label in `Tensor`
    '''
    # padding zeros
    zero_padding: tf.Tensor = tf.zeros(tf.maximum([16000] - tf.shape(waveform), 0), dtype=tf.float32)
    waveform = waveform[:16000]
    waveform = tf.cast(waveform, tf.float32)
    waveform = tf.concat([waveform, zero_padding], axis=0)

    # fourier transform
    spectrogram: tf.Tensor = tf.signal.stft(waveform, frame_length=256, frame_step=128)
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.expand_dims(spectrogram, -1)

    # to one hot label
    onehot_label: tf.Tensor = tf.one_hot(label, num_classes)
    return spectrogram, onehot_label

# mapping to dataset
dataset: tf.data.Dataset = dataset["train"]
dataset = dataset.map(stft)

# split dataset
testing_dataset: tf.data.Dataset = dataset.take(testing_dataset_size).batch(64)
remaining_dataset = dataset.skip(testing_dataset_size)
training_dataset: tf.data.Dataset = remaining_dataset.take(training_dataset_size).batch(64)
validation_dataset: tf.data.Dataset = remaining_dataset.skip(training_dataset_size).take(validation_dataset_size).batch(64)

"""
-----------------------------------------------------------------------------------------------
Step 2: Define CRNN model [30 points]
1. Resize layer to resize the spectrogram to 32x32
2. Normalization layer to normalize the input data based on its mean and standard deviation
3. Conv2D layer named 'conv1' with 64 filters, 3x3 kernel size, same padding, and relu activation
4. BatchNormalization layer to normalize axis 3
5. MaxPooling2D layer to pool the features
6. Conv2D layer named 'conv2' with 64 filters, 3x3 kernel size, same padding, and relu activation
7. BatchNormalization layer to normalize axis 3
8. MaxPooling2D layer to pool the features
9. Permute layer to permute the frequency axis and time axis
10. Reshape the permuted output to (-1, shape[1], shape[2] * shape[3])
11. A GRU layer named 'gru1' with 512 units
12. A GRU layer named 'gru2' with 512 units
13. A Dropout layer with dropout ratio 0.5
14. A Dense layer to do the classification
-----------------------------------------------------------------------------------------------
"""


input_data: tf.Tensor = Input(shape=(None, None, 1))
x: tf.Tensor = tf.cast(input_data, tf.float32)
x = layers.experimental.preprocessing.Resizing(32, 32)(x)
norm_layer = layers.experimental.preprocessing.Normalization()
norm_layer.adapt(training_dataset.map(lambda x, _: x))
x = norm_layer(x)
x = layers.Conv2D(64, 3, padding='same', activation='relu', name='conv1')(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D()(x)
x = layers.Conv2D(64, 3, padding='same', activation='relu', name='conv2')(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D()(x)
x = layers.Permute((2, 1, 3))(x)
resize_shape = x.shape[2] * x.shape[3]
x = layers.Reshape((x.shape[-2], -1))(x)
x = layers.GRU(512, return_sequences=True, name='gru1')(x)
x = layers.GRU(512, return_sequences=True, name='gru2')(x)
x = layers.Flatten()(x)
x = layers.Dropout(0.5)(x)
y: tf.Tensor = layers.Dense(num_classes, activation='softmax')(x)


model = Model(inputs=input_data, outputs=y, name='spoken_digit_classification')
model.summary()

"""
-----------------------------------------------------------------------------------------------
Step 3: Compile the model [10 points]
Adam optimizer with 0.001 learning rate, multiplies 0.9 for each epoch;
Categorical crossentropy as loss function;
Accuracy as the metrics;
-----------------------------------------------------------------------------------------------
"""

# define optimizera
lr_schedule = optimizers.schedules.ExponentialDecay(0.001, steps_per_epoch, 0.9)
optimizer = optimizers.Adam(lr_schedule)

# compile model
model.compile(optimizer, loss='categorical_crossentropy', metrics=['categorical_accuracy'])

"""
-----------------------------------------------------------------------------------------------
Step 4: Train the model with training dataset [10 points]
TensorBoard Callback to record the metrics for each epoch;
Checkpoint Callback to save checkpoints;
Set epoch size to 30;
-----------------------------------------------------------------------------------------------
"""

# initialize callbacks
tensorboard_callback = callbacks.TensorBoard()
ckpt_callback = callbacks.ModelCheckpoint('checkpoints', save_weights_only=True)

# train model for 100 epochs
model.fit(training_dataset, epochs=30, validation_data=validation_dataset, callbacks=[tensorboard_callback, ckpt_callback])

"""
-----------------------------------------------------------------------------------------------
Step 5: Evaluate the model with testing dataset [10 points]
Final accuracy =
-----------------------------------------------------------------------------------------------
"""

# evaluate dataset
test_loss, test_acc = model.evaluate(testing_dataset, verbose=0)
print('Test loss:', test_loss)
print('Test accuracy:', test_acc*100)
"""
-----------------------------------------------------------------------------------------------
Step 6: Remember to submit results to Canvas. [10 points]
A screenshot of TensorBoard;
A screenshot of the training and testing procedure;
-----------------------------------------------------------------------------------------------
"""
