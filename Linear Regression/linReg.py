import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

num_points=100
points=[]

for i in range (num_points):
    x1=np.random.normal(0.0, 0.55)
    y1=x1*0.1+0.3+np.random.normal(0.0, 0.03)
    points.append([x1, y1])

x_data=[v[0] for v in points]
y_data=[v[1] for v in points]

plt.plot(x_data, y_data, 'ro', label='Original data')
plt.legend()
plt.show()


W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.zeros([1]))
y = W * x_data + b

loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

for step in range(101):
    sess.run(train)
    if step % 10 == 0:
        print(step, sess.run(W), sess.run(b))

plt.plot(x_data, y_data, 'ro')
plt.plot(x_data, sess.run(W) * x_data + sess.run(b))
plt.legend()
plt.show()
