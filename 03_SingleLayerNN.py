from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./MNIST_data/", one_hot = True)

import tensorflow as tf

print(tf.convert_to_tensor(mnist.train.images).get_shape())
print(tf.convert_to_tensor(mnist.train.labels).get_shape())

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

x = tf.placeholder("float", [None, 784])
y_ = tf.placeholder("float", [None, 10])

y = tf.nn.softmax(tf.matmul(x, W) +b)

cross_entropy = -tf.reduce_sum(y_ * tf.log(y))

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(1000):
		batch_xs, batch_ys = mnist.train.next_batch(100)

		sess.run(train_step, feed_dict = {x:batch_xs, y_:batch_ys})

		correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
		accuracy_ = sess.run(accuracy, feed_dict = {x:mnist.test.images, y_:mnist.test.labels})

		print('step: {:01d} | accuracy: {:.4f}'.format(step, float(accuracy_)))


