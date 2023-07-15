import tensorflow as tf
from keras import losses, metrics, optimizers
from keras.initializers.initializers_v2 import GlorotUniform
from keras.datasets import mnist
from tensorflow._api.v2.summary import create_file_writer

"""
-----------------------------------------------------------------------------------------------
Step 1: Load dataset MNIST [10 points]
Load dataset from Keras;
Wrap to tensorflow dataset;
Set epoch size to 10;
Set batch size to 256 (or fit your GPU GRAM size);
-----------------------------------------------------------------------------------------------
"""
# load dataset from keras
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# normalize with onehot
x_train: tf.Tensor = tf.convert_to_tensor(x_train / 255.) # type: ignore # must normalize
y_train: tf.Tensor = tf.one_hot(y_train, depth=10) # must convert to onehot
x_test: tf.Tensor = tf.convert_to_tensor(x_test / 255.) # type: ignore # must normalize
y_test: tf.Tensor = tf.one_hot(y_test, depth=10) # must convert to onehot

# wrap to tensorflow dataset
training_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(256).repeat(10)
testing_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(256)

"""
-----------------------------------------------------------------------------------------------
Step 2: Define logistic regression model [15 points]
Initialize the model instance;
Use CategoricalCrossentropy as your loss function;
Use CategoricalAccuracy as your metrics;
Use Adam with its default settings as your optimizer;
-----------------------------------------------------------------------------------------------
"""

# logistic regression model method
class LogisticRegression(tf.Module):
    W: tf.Variable
    b: tf.Variable

    def __init__(self, name=None):
        super().__init__(name=name)
        initializer = GlorotUniform()
        self.W = tf.Variable(initializer((784,10)), shape=(784,10), name='weight')
        self.b = tf.Variable(tf.zeros(10), shape=10, name='bias')
            
    # forward pass
    def __call__(self, input_data: tf.Tensor) -> tf.Tensor:
        x: tf.Tensor = tf.cast(input_data, tf.float32) # type: ignore
        x = tf.reshape(x, (-1, 784)) # (-1, 28, 28, 1) -> (-1, 784)
        x = tf.matmul(x, self.W) + self.b # y = wx + b, do not use x * self.W
        y: tf.Tensor = tf.nn.softmax(x) # type: ignore
        return y
        
# initialize model
model = LogisticRegression(name='logistic_regression')
loss_fn = losses.CategoricalCrossentropy()
acc_fn = metrics.CategoricalAccuracy()
optimizer = optimizers.Adam()

"""
-----------------------------------------------------------------------------------------------
Step 3: train the model [20 points]
Use summrary writer record TensorBoard data;
Use Checkpoint to save checkpoints;
Use training_dataset to train the model;
-----------------------------------------------------------------------------------------------
"""

# initialize summary writer
train_summary_writer = create_file_writer("logs/train") # type: ignore

# initialize checkpoint path
ckpt_path = "ckpt/checkpoint"

# loop for training dataset
for i, (x_batch, y_batch) in enumerate(training_dataset):
    # gradient tape (forward pass)
    with tf.GradientTape() as tape:
        y = model(x_batch)
        loss: tf.Tensor = loss_fn(y, y_batch)
        acc: tf.Tensor = acc_fn(y, y_batch)
        
    # backward pass
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    
    # record loss and accuracy
    if i % 50 == 0:
        # print out result
        print(f"Iteration {i}: loss={loss:.4f}, accuracy={acc:.4f}")
        
        # write into tensorboard
        with train_summary_writer.as_default():
            tf.summary.scalar('loss', loss, step=i)
            tf.summary.scalar('accuracy', acc, step=i)
    
    # save checkpoint
    if i % 2000 == 0:
        checkpoint = tf.train.Checkpoint(model=model)
        checkpoint.write(ckpt_path)

"""
-----------------------------------------------------------------------------------------------
Step 4: Test the model [15 points]
Record each loss and accuracy in the list;
Calculate the mean loss and accuracy;
Hint: If the accuracy is less then 85%, increase your epoch size and use a different random function for the weights.
Final accuracy =
-----------------------------------------------------------------------------------------------
"""

# initialize evaluating
loss_list: list[tf.Tensor] = list()
accuracy_list: list[tf.Tensor] = list()

# evaluate dataset
for x_batch, y_batch in testing_dataset:
    # forward pass
    y = model(x_batch)
    loss = loss_fn(y_batch, y)
    acc = acc_fn(y_batch, y)
    loss_list.append(loss)
    accuracy_list.append(acc)
    
# calculate mean loss and mean accuracy
loss_eval: tf.Tensor = tf.reduce_mean(loss_list)
acc_eval: tf.Tensor = tf.reduce_mean(accuracy_list)
tf.print("eval_loss=%f, eval_acc=%f" % (loss_eval, acc_eval))

"""
-----------------------------------------------------------------------------------------------
Remember to submit results to Canvas. [10 points]
-----------------------------------------------------------------------------------------------
"""
