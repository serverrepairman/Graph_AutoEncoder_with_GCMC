import tensorflow as tf
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical
import numpy as np
#import matplotlib.pyplot as plt
import sys
sys.setrecursionlimit(10000)
import os
from layers import *
from utils import *
from model import *
import pickle

N = 30
F = 1
batch = 500
showing_data_num = 1


[filename, dataset, is_VAE, is_GCN, latent_dim, epoch, GPU_NUM] = sys.argv
[is_VAE, is_GCN, latent_dim, epoch] = list(map(int,[is_VAE, is_GCN, latent_dim, epoch]))

os.environ["CUDA_VISIBLE_DEVICES"]=GPU_NUM

saving_path = '../OUTPUTS/'+dataset
data_path = '../DATAS/'+dataset

if is_VAE :
	saving_path += '/VAE/'
else :
	saving_path += '/AE/'

if is_GCN :
	saving_path += 'GCN/'
	data_path +='/GCN/'
else :
	saving_path += 'GNN/'
	data_path += '/GNN/'

saving_path += 'Dim_'+str(latent_dim)+'/'

# make dir
if not os.path.isdir(saving_path) :
	os.makedirs(saving_path)
if not os.path.isdir(saving_path+'images') :
	os.makedirs(saving_path+'images')
if not os.path.isdir(saving_path+'callbacks') :
	os.makedirs(saving_path+'callbacks')
sys.stdout = open(saving_path+'outlog.txt',"w")

A_train=np.loadtxt(data_path+'train.txt',dtype=np.int32)
A_test=np.loadtxt(data_path+'test.txt',dtype=np.int32)
train_data_size = np.shape(A_train)[0]
test_data_size = np.shape(A_test)[0]
A_train = A_train.reshape(train_data_size, N, N)
A_test = A_test.reshape(test_data_size, N, N)

if is_VAE :
	AE = GNN_VAE(N,F,latent_dim)
else :
	AE = GNN_AE(N,F,latent_dim)

AE.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
AE.summary()


test_show=A_test[showing_data_num]
plt.imsave(saving_path+'images/input1.png',double_image(test_show))
test_show = test_show.reshape(1,N,N)
cbk = MyCbk(AE,test_show = test_show, saving_path = saving_path, N = N, F = F)

# fitting
hist = AE.fit([A_train, np.ones((train_data_size, N, F))], A_train,
	epochs=epoch, batch_size=batch,callbacks=[cbk],validation_data=([A_test, np.ones((test_data_size, N, F))], A_test))

# save history
with open(saving_path+'history', 'wb') as file_pi:
	pickle.dump(hist.history, file_pi)

AE.save('GNN.h5')
AE.save_weights('GNN_weight.h5')

plt.figure()
plt.plot(hist.history['val_loss'],'y', label="test_loss", color='red')
plt.savefig('loss.png', dpi=300)

plt.figure()
plt.plot(hist.history['val_accuracy'],'y', label="test_acc", color='blue')
plt.savefig('acc.png',dpi=300)
