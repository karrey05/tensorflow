
import numpy as np
import pandas as pd

import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

milk = pd.read_csv('monthly-milk-production.csv',index_col='Month')
milk.index = pd.to_datetime(milk.index)

train_set = milk.head(156)
test_set = milk.tail(12)

scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train_set)
test_scaled = scaler.transform(test_set)

# Just one feature, the time series
num_inputs = 1
# Num of steps in each batch
num_time_steps = 12
# 100 neuron layer, play with this
num_neurons = 100
# Just one output, predicted time series
num_outputs = 1

learning_rate = 0.03 
# how many iterations to go through (training steps), you can play with this
num_train_iterations = 1000
# Size of the batch of data
batch_size = 1

model_dir = './time_series_model/'


def next_batch(training_data,batch_size,steps):
    # Grab a random starting point for each batch
    rand_start = np.random.randint(0,len(training_data)-steps) 
    # Create Y data for time series in the batches
    y_batch = np.array(training_data[rand_start:rand_start+steps+1]).reshape(1,steps+1)
    return y_batch[:, :-1].reshape(-1, steps, 1), y_batch[:, 1:].reshape(-1, steps, 1) 


def build_graph(x, y):
    # Also play around with GRUCell
    cell = tf.contrib.rnn.OutputProjectionWrapper(
        tf.contrib.rnn.BasicLSTMCell(num_units=num_neurons, activation=tf.nn.relu),
        output_size=num_outputs) 
    outputs, states = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)
    loss = tf.reduce_mean(tf.square(outputs - y)) # MSE
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train = optimizer.minimize(loss)
    saver = tf.train.Saver()
    with tf.Session(graph=prediction_graph) as sess:
        sess.run([tf.global_variables_initializer(), tf.tables_initializer()])
        for iteration in range(num_train_iterations):
            x_batch, y_batch = next_batch(train_scaled, batch_size, num_time_steps)
            sess.run(train, feed_dict={x: x_batch, y: y_batch})
            if iteration % 100 == 0:
                mse = loss.eval(feed_dict={x: x_batch, y: y_batch})
                print(iteration, "MSE:", mse)
        saver.save(sess, model_dir)
    return outputs, saver

# model training
with tf.Graph().as_default() as prediction_graph:
    x = tf.placeholder(tf.float32, [None, num_time_steps, num_inputs])
    y = tf.placeholder(tf.float32, [None, num_time_steps, num_outputs])
    outputs, saver = build_graph(x, y)

# save model
with tf.Session(graph=prediction_graph) as sess:
    sess.run([tf.global_variables_initializer(), tf.tables_initializer()])
    tf.saved_model.simple_save(
        sess,
        export_dir='SavedModel',
        inputs={"x": x},
        outputs={"y": outputs}
    )




