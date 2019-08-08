'''
@author:Lee
@file:RNN.py
@Time: 2019/4/9 15:30
@Description:使用随机生成的数据集训练单层RNN(LSTM、GRU)
'''
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.contrib import  rnn
class Data:
    def __init__(self,data_size,num_batch,batch_size,time_step):
        self.data_size = data_size
        self.num_batch = num_batch
        self.batch_size = batch_size
        self.time_step = time_step
        self.without_rel = []       #保存随机生成的没有关联的数据
        self.data_with_rel = []     #保存有时序关联的数据
    def generate_data(self):
        self.without_rel = np.array(np.random.choice(2,size=(self.data_size,)))
        for i in range(self.data_size):
            if self.without_rel[i-1] ==1 and self.without_rel[i-2]==1:
                self.data_with_rel.append(0)
                continue
            elif self.without_rel[i-1]==0 and self.without_rel[i-2]==0:
                self.data_with_rel.append(1)
                continue
            else:
                if np.random.rand()>=0.5:
                    self.data_with_rel.append(1)
                else: self.data_with_rel.append(0)
        return self.without_rel,self.data_with_rel

   #生成循环次数
    def generate_epochs(self):
        self.generate_data()
        data_x = np.zeros([self.num_batch,self.batch_size],dtype=np.int32)
        data_y = np.zeros([self.num_batch,self.batch_size],dtype=np.int32)

        #将数据划分成num_batch组
        for i in range(self.num_batch):
            data_x[i] = self.without_rel[i*self.batch_size:(i+1)*self.batch_size]
            data_y[i] = self.data_with_rel[i*self.batch_size:(i+1)*self.batch_size]
        epoch_size = self.batch_size//self.time_step

        #返回最终数据
        for i in range(epoch_size):
            x = data_x[:,self.time_step*i:self.time_step*(i+1)]
            y = data_y[:,self.time_step*i:self.time_step*(i+1)]
            yield (x,y)
class Model:
    def __init__(self,data_size,batch_size,time_step,state_size):
        self.data_size = data_size
        self.batch_size = batch_size
        self.time_step = time_step
        self.state_size= state_size
        self.num_batch = self.data_size//self.batch_size

        #输入数据的占位符
        self.x = tf.placeholder(tf.int32,shape=[self.num_batch,self.time_step],name='placeholder_x')
        self.y = tf.placeholder(tf.int32,shape=[self.num_batch,self.time_step],name='placeholder_y')

        #记忆单元的占位符
        self.init_state = tf.zeros([self.num_batch,self.state_size])
        #进行one-hot编码
        self.rnn_inputs = tf.one_hot(self.x,2)

        #隐藏层权重矩阵和偏置项
        self.W = tf.get_variable('W',[self.state_size,2])
        self.b = tf.get_variable('b',[2],initializer=tf.constant_initializer(0.0))

        #RNN隐藏层的输出
        self.rnn_output,self.final_state = self.model()

        #计算输出层的输出
        logits = tf.reshape(tf.matmul(tf.reshape(self.rnn_output,[-1,self.state_size]),
                                      self.W)+self.b,[self.num_batch,self.time_step,2])
        self.losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y,logits=logits)
        self.total_loss = tf.reduce_mean(self.losses)
        self.train_step = tf.train.AdagradOptimizer(0.1).minimize(self.total_loss)
        

    #定义RNN模型
    def model(self):
        cell = rnn.BasicRNNCell(self.state_size)
        rnn_outputs,final_state = tf.nn.dynamic_rnn(cell,self.rnn_inputs,initial_state=self.init_state)
        return rnn_outputs,final_state

    def train(self):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            training_losses = []
            d = Data(self.data_size,self.num_batch,self.batch_size,self.time_step)
            training_loss = 0
            training_state = np.zeros((self.num_batch,self.state_size))
            for step, (x,y) in enumerate(d.generate_epochs()):
                tr_losses,training_loss_,training_state,_ = sess.run([self.losses,self.total_loss,self.final_state,self.train_step],
                                                feed_dict={self.x:x,self.y:y,self.init_state:training_state})
                training_loss+=training_loss_
                if step%20==0 and step>0:
                    training_losses.append(training_loss/20)
                    training_loss=0
        return training_losses

if __name__ == '__main__':
    data_size = 5000000
    batch_size = 2000
    time_step = 5
    state_size = 6

    m = Model(data_size, batch_size, time_step, state_size)
    training_losses = m.train()
    plt.plot(training_losses)
    plt.show()
