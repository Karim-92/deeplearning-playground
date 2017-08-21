import input_data
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import summaries
from tensorflow.python.client import timeline

""""MNIST Example for CNN with dropout and relU """
# Run options for tensorflow to enable the logging/tensorboard
run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
run_metadata = tf.RunMetadata()

# Load data, enable one hot encoding
mnist=input_data.read_data_sets("MNIST_data/", one_hot=True)

# Parameters
learning_rate = 0.001
training_iters = 200000
batch_size = 128
display_step = 10
classes= 10 # Classes in MNIST 0-9
inputShape= 784 # 28x28 pixels of MNIST image
dropoutValue = 0.75

# Create input, label and dropout values tensors
x = tf.placeholder(tf.float32, [None, inputShape])
y = tf.placeholder(tf.float32, [None, classes])
keepProbability = tf.placeholder(tf.float32)

# Create abstract network wrappers
# Convolutional layer with relU activion, Stride=1 by default, padding =0
def conv2d(x, W, b, strides=1):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

# Max pool layer where stride =2 as default, stride =2 by default, same padding
def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')

# CNN model
def createModel(x, weights, biases, dropoutValue):
    # Reshape input
    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    # Create first conv, max pool layers
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    conv1 = maxpool2d(conv1, k=2)

    # Create second conv, max pool layers
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    conv2 = maxpool2d(conv2, k=2)

    # Final layer, Fully connected layer
    # flatten output from last conv layer
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)

    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropoutValue)

    # Output, class prediction
    output = tf.add(tf.matmul(fc1, weights['output']), biases['output'])
    return output


# Store layers weight & bias Parameters
weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([7*7*64, 1024])),
    # 1024 inputs, 10 outputs (class prediction)
    'output': tf.Variable(tf.random_normal([1024, classes]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'output': tf.Variable(tf.random_normal([classes]))
}

# Construct model
prediction = createModel(x, weights, biases, keepProbability)

# Define cost function and optimizer(ADAM)
with tf.name_scope('cost'):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    summaries.summarize_tensor(cost, tag='cost')

# Training step
with tf.name_scope('optimizer'):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Model evaluation
with tf.name_scope('accuracy'):
    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    summaries.summarize_tensor(accuracy, tag='accuracy')

# Create summary writer, TF Session and initialize all variables
step = 0
sess = tf.Session()
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter('log/mnist', sess.graph)
init = tf.global_variables_initializer()
sess.run(init)

# Launch Session and train until max training is reached
while (step*batch_size <= training_iters):
    batch_x, batch_y = mnist.train.next_batch(batch_size)
    # Back prop error
    _, summaryString = sess.run([optimizer, merged], feed_dict={x: batch_x, y: batch_y, keepProbability: dropoutValue})
    writer.add_summary(summaryString, step)
    if step % display_step == 0:
        # Calculate batch loss and accuracy
        loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,
                                                          y: batch_y,
                                                          keepProbability: 1.}, options=run_options,
                                                          run_metadata=run_metadata)
        print("Iter " + str(step*batch_size) + ", Minibatch Loss= %.2f%% , Training Accuracy= %.2f%%" % ((loss/100) , (acc) * 100))
    step += 1
print("Training Finished!")

# Calculate accuracy for 256 mnist test images
print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: mnist.test.images[:256],
                                  y: mnist.test.labels[:256],
                                  keepProbability: 1.}, options=run_options,
                                  run_metadata=run_metadata))
