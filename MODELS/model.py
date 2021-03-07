import tensorflow as tf
import tensorflow.linalg as tlin
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical
import numpy as np
from layers import *
from utils import *

def GNN_AE(N,F,latent_dim):

    vec = []

    A = Input(shape=(N, N))
    input = Input(shape=(N, F))
    
    x = GNN(32)([input,A])
    x = Activation(mish)(x)

    x = GNN(64)([x,A])
    x = Activation(mish)(x)

    x = GNN(128)([x,A])
    x = Activation(mish)(x)
    vec.append(x)


    x = GNN(64)([input,A])
    x = Activation(mish)(x)

    x = GNN(128)([x,A])
    x = Activation(mish)(x)
    vec.append(x)


    x = GNN(128)([input,A])
    x = Activation(mish)(x)
    vec.append(x)

    x = Add()(vec)

    x = GDense(64)(x)
    x = Activation(mish)(x)
    x = GDense(32)(x)
    x = Activation(mish)(x)
    latent = GDense(latent_dim)(x)
    latent = Activation(mish)(latent)

    x0 = inner_dot()(latent)
    x1 = inner_dot()(latent)
    x = tf.nn.softmax([x0,x1],0)[1]

    return Model(inputs = [A,input],outputs = x)

def GNN_VAE(N,F,latent_dim):

    vec = []

    A = Input(shape=(N, N))
    input = Input(shape=(N, F))
    
    x = GNN(32)([input,A])
    x = Activation(mish)(x)

    x = GNN(64)([x,A])
    x = Activation(mish)(x)

    x = GNN(128)([x,A])
    x = Activation(mish)(x)
    vec.append(x)


    x = GNN(64)([input,A])
    x = Activation(mish)(x)

    x = GNN(128)([x,A])
    x = Activation(mish)(x)
    vec.append(x)


    x = GNN(128)([input,A])
    x = Activation(mish)(x)
    vec.append(x)

    x = Add()(vec)
    x = GDense(64)(x)
    x = Activation(mish)(x)
    x = GDense(32)(x)
    x = Activation(mish)(x)

    z_mean = GDense(latent_dim)(x)
    z_log_var = GDense(latent_dim)(x)

    latent = VAE()([z_mean,z_log_var])

    x0 = inner_dot()(latent)
    x1 = inner_dot()(latent)
    x = tf.nn.softmax([x0,x1],0)[1]

    return Model(inputs = [A,input],outputs = x)