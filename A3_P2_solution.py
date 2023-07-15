# import modules
import gzip, os, pickle, tensorflow as tf, tensorflow_datasets as TFDatasets
from keras import Model, models
from tensorflow._api.v2.data import Dataset

"""
-----------------------------------------------------------------------------------------------
Step 1. Restore the model you saved in Problem 1 [5 points]
-----------------------------------------------------------------------------------------------
"""
model = models.load_model("model")
assert isinstance(model, Model), "Given saved model is not a valid keras model."

"""
-----------------------------------------------------------------------------------------------
Step 2. Test the restored model with both MNIST and USPS testing samples [10 points]
Set batch size to 128
MNIST accuracy = > 98%
USPS accuracy before fine-tune = about 92%
-----------------------------------------------------------------------------------------------
"""
with gzip.open('usps.pkl') as f:
    u = pickle._Unpickler(f)
    u.encoding = 'latin1' # type: ignore
    usps_train_set, usps_test_set = u.load()

    x_train, y_train = usps_train_set
    x_train = tf.reshape(x_train, (-1, 28, 28, 1))
    x_test, y_test = usps_test_set
    x_test = tf.reshape(x_test, (-1, 28, 28, 1))
    
# load mnist dataset
dataset_dir = os.path.normpath("~/Downloads/Datasets/")
mnist: Dataset = TFDatasets.load("mnist", data_dir=dataset_dir, split='test', as_supervised=True) # type: ignore
mnist = mnist.map(lambda image, l: (image / 255, l)).batch(128) # must normalize

# model.evaluate(mnist)
model.evaluate(mnist)
model.evaluate(x_test, y_test, batch_size=128)

"""
-----------------------------------------------------------------------------------------------
Step 3. Train you CNN with USPS training samples [10 points]
Set epochs to 5 and batch size to 128
-----------------------------------------------------------------------------------------------
"""
# fine-tune
model.fit(x_train, y_train, epochs=5, batch_size=128, validation_data=(x_test, y_test))

"""
-----------------------------------------------------------------------------------------------
Step 4. Test your fine tuned CNN on USPS testing data and report testing accuracy [5 points]
USPS accuracy after fine-tune = > 98%
-----------------------------------------------------------------------------------------------
"""
model.evaluate(x_test, y_test)
