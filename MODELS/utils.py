import tensorflow as tf
from tensorflow.keras.layers import Layer
from keras import backend as K
import numpy as np
from tensorflow.keras.callbacks import * 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def mish(x):
    return x * K.tanh(K.softplus(x))

def DEG(x):
    N = (np.shape(x))[1]
    T = (np.shape(x))[0]

    x = np.sum(x,axis = 1)
    print(np.shape(x[0]))
    y = np.diag(1/x[0]).reshape(1,N,N)
    for i in range(1,T):
        if i % 1000 == 0 :
            print(i)
        y = np.append(y,np.diag(1/x[i]).reshape(1,N,N),axis = 0)
    y = np.sqrt(y)
    return y

def double_image(input_image):
    N = np.shape(input_image)[0]
    M = np.shape(input_image)[1]
    return_image = np.ones((2*N, 2*M))
    for i in range(N) :
        for j in range(M) :
            return_image[2*i][2*j] = return_image[2*i+1][2*j] = return_image[2*i][2*j+1] = return_image[2*i+1][2*j+1] = input_image[i][j] 
    return return_image

class MyCbk(Callback):

    def __init__(self, model, test_show,saving_path,N,F):
         self.model_to_save = model
         self.test_show = test_show
         self.saving_path = saving_path
         self.N = N
         self.F = F

    def on_epoch_end(self, epoch, logs=None):
        if epoch%5==0 :
            self.model_to_save.save(self.saving_path+'callbacks/model_at_epoch_%d.h5' % epoch)
        testpred=self.model_to_save.predict([self.test_show,np.ones((1, self.N, self.F))])
        testpred=testpred.reshape(30,30)
        if(epoch <= 100 or epoch % 10 == 0):
            plt.imsave(self.saving_path+'images/output1_at_epoch_%d.png' % epoch, double_image(testpred))
