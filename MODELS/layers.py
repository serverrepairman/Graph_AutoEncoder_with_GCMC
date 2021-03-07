import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.layers import Layer
from utils import *

class GNN(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(GNN, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_variable(name='kernel', 
                                      shape=(input_shape[0][2],self.output_dim),
                                      initializer='random_normal',
                                      trainable=True)
        self.bias = self.add_variable(name='bias', 
                                      shape=(input_shape[0][1],self.output_dim),
                                      initializer='random_normal',
                                      trainable=True)
        super(GNN, self).build(input_shape)

    def call(self, input):
        x = tf.linalg.matmul(input[1], input[0])
        x = tf.linalg.matmul(x,self.kernel)
        return x + self.bias

    def get_config(self):
        config = super(GNN, self).get_config()
        config.update({"output_dim" : self.output_dim})
        return config

class GDense(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(GDense, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_variable(name='kernel', 
                                      shape=(input_shape[-1],self.output_dim),
                                      initializer='random_normal',
                                      trainable=True)
        self.bias = self.add_variable(name='bias', 
                                      shape=(input_shape[1],self.output_dim),
                                      initializer='random_normal',
                                      trainable=True)
        super(GDense, self).build(input_shape)

    def call(self, input):
        return tf.linalg.matmul(input,self.kernel) + self.bias

    def get_config(self):
        config = super(GDense, self).get_config()
        config.update({"output_dim" : self.output_dim})
        return config

class inner_dot(Layer):

    def __init__(self, **kwargs):
        super(inner_dot, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_variable(name='kernel', 
                                      shape=(input_shape[-1],input_shape[-1]),
                                      initializer='random_normal',
                                      trainable=True)
        super(inner_dot, self).build(input_shape)

    def call(self, input):
        x = tf.linalg.matmul(input,self.kernel)
        return tf.linalg.matmul(x,tf.transpose(input,[0,2,1]))

    def get_config(self):
        config = super(inner_dot, self).get_config()
        return config

class VAE(Layer): #[z_mean, z_log_var]

    def __init__(self, **kwargs):
        super(VAE, self).__init__(**kwargs)

    def build(self, input_shape):
        super(VAE, self).build(input_shape)

    def call(self, input):
        eps = tf.random.normal(shape = tf.shape(input[0]))
        return eps*tf.math.exp(0.5*input[1])+input[0]

    def get_config(self):
        config = super(VAE, self).get_config()
        return config
