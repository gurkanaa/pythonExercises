import numpy as np
import tensorflow as tf
#model
x=tf.placeholder(tf.float32)
y=tf.placeholder(tf.float32)
W=tf.Variable([.3],tf.float32)
b=tf.Variable([-.3],tf.float32)
linear_model=W*x+b
#loss
loss=tf.reduce_sum(tf.square(linear_model-y))
#optimizer
optimizer=tf.train.GradientDescentOptimizer(0.02)
train=optimizer.minimize(loss)
#training data
x_train=[1,3,5]
y_train=[4.99,9.01,13]
#training
init=tf.global_variables_initializer()
sess=tf.Session()
sess.run(init)#init values in W to .3 and b to -.3
for i in range(1000):
    sess.run(train,{x:x_train,y:y_train})

#display
W_,b_,loss_=sess.run([W,b,loss],{x:x_train,y:y_train})
print("W: %s b: %s loss: %s",W_,b_,loss_)
