# coding: utf-8
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime
import time
from sklearn.metrics import mean_squared_error
import tensorflow as tf
import sys

class NN_KID_model:

    def __init__(self, KID_NAME):

        self.KID_NAME=KID_NAME


    def predict(self, X):

        tf.reset_default_graph()
        path="/home/nakajo/GT_Kids/Results/ML_data/vailidation_ver1/ml_data/"
        nInput=5
        nOutput=2
        epoch=10000
        current_state_server=X
        self.DefinePlaceHolder(nInput, nOutput)
        self.DetermineNeuron(nInput, nOutput)
        saver = tf.train.Saver()

        with tf.Session() as sess:  # your session object
            start=time.time()
            self.InitiateModel()
            sess.run(tf.global_variables_initializer()) # Initialize variables
            saver = tf.train.import_meta_graph(path+'/'+self.KID_NAME+'_NN_model/'+self.KID_NAME+'_NN_model.ckpt.meta')
            saver.restore(sess, tf.train.latest_checkpoint(path+'/'+self.KID_NAME+'_NN_model'))
            print("Model restored.")
            # Check the values of the variables
            #print("W_fc1 : %s" % W_fc1.eval())
            print("Current states: ", current_state_server)
            NN_predict=sess.run(y, feed_dict={x:current_state_server})
            print("Neural network Future Prediction values:", NN_predict[:, 0])
            print("Prediction time: ", time.time()-start)
            return NN_predict


    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape,stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.1,shape=shape)
        return tf.Variable(initial)

    def MakeInputData(self, df_KID, KID_NAME):
        input_current_state=df_KID[['CPU_utilization','FAN (%)', 'CPU_temperature(degC)', 'PS']].reset_index()
        input_job=df_KID[KID_NAME+'_jobs'].reset_index()
        Inp_np=np.empty((0, 5), float)
        for i in range(len(df_KID)-1):
            Inp_series=pd.concat([input_current_state.ix[i], input_job.ix[i+1]]).drop('Elapsed time (s)')
            tmp_np=np.array(Inp_series)
            Inp_np=np.append(Inp_np, [tmp_np], axis=0)
        input_data_np=Inp_np
        return input_data_np

    def MakeOutputData(self, df_KID, KID_NAME):
        output_candidates=df_KID[['CPU_temperature(degC)', KID_NAME+'_serve_time']].reset_index()
        Oup_np=np.empty((0,2), float)
        oup_series=output_candidates.drop('Elapsed time (s)', axis=1)
        tmp_np=np.array(oup_series)
        Oup_np=np.append(Oup_np,tmp_np, axis=0)
        output_data_np=Oup_np[1:] # skip the initital state 
        return output_data_np

    def DetermineNeuron(self, nInput, nOutput):
        global W_fc1, b_fc1, h_fc1, W_fc2, b_fc2, h_fc2, keep_prob, W_fc3, b_fc3, y
        #第１層
        W_fc1 = self.weight_variable([nInput,30])
        b_fc1 = self.bias_variable([30])
        h_fc1 = tf.nn.relu(tf.matmul(x,W_fc1) + b_fc1) #発火の定義，発火関数はrelu

        #第２層
        W_fc2 = self.weight_variable([30,10])
        b_fc2 = self.bias_variable([10])
        h_fc2 = tf.nn.relu(tf.matmul(h_fc1,W_fc2) + b_fc2)

        keep_prob = tf.placeholder(tf.float32)  

        #最終層（OutPut）
        W_fc3 = self.weight_variable([10,nOutput])
        b_fc3 = self.bias_variable([nOutput])
        y = tf.nn.relu(tf.matmul(h_fc2,W_fc3) + b_fc3)

    def DefinePlaceHolder(self, nInput, nOutput):
        global x, y_, w, b
        x = tf.placeholder(tf.float32, shape=[None, nInput], name = "input")
        #x = tf.placeholder(tf.float32, shape=[nInput], name = "input")
        y_ = tf.placeholder(tf.float32, shape=[None, nOutput], name = "output") 
        #y_ = tf.placeholder(tf.float32, shape=[nOutput], name = "output") 

        w = tf.Variable(tf.zeros([nInput, nOutput]))  #weight
        b = tf.Variable(tf.zeros([nOutput])) #bias

    def InitiateModel(self):
        global cross_entropy, train_step
        #y_ = training data , y = predection data
        cross_entropy = tf.reduce_mean(tf.reduce_sum(tf.square(y - y_)))
        #train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
        #sess = tf.InteractiveSession()

