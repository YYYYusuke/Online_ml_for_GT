# coding: utf-8
import tensorflow as tf
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import time
import csv
from sklearn.metrics import mean_squared_error

def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)
def bias_variable(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

def MakeInputData(df_KID, KID_NAME):
    input_current_state=df_KID[['CPU_utilization','FAN (%)', 'CPU_temperature(degC)', 'PS']].reset_index()
    input_job=df_KID[KID_NAME+'_jobs'].reset_index()
    Inp_np=np.empty((0, 5), float)
    for i in range(len(df_KID)-1):
        Inp_series=pd.concat([input_current_state.ix[i], input_job.ix[i+1]]).drop('Elapsed time (s)')
        tmp_np=np.array(Inp_series)
        Inp_np=np.append(Inp_np, [tmp_np], axis=0)
    input_data_np=Inp_np
    return input_data_np

def MakeOutputData(df_KID, KID_NAME):
    output_candidates=df_KID[['CPU_temperature(degC)', KID_NAME+'_serve_time']].reset_index()
    Oup_np=np.empty((0,2), float)
    oup_series=output_candidates.drop('Elapsed time (s)', axis=1)
    tmp_np=np.array(oup_series)
    Oup_np=np.append(Oup_np,tmp_np, axis=0)
    output_data_np=Oup_np[1:] # skip the initital state
    return output_data_np

def DetermineNeuron(nInput, nOutput):
    global W_fc1, b_fc1, h_fc1, W_fc2, b_fc2, h_fc2, keep_prob, W_fc3, b_fc3, y
    W_fc1 = weight_variable([nInput,30])
    b_fc1 = bias_variable([30])
    h_fc1 = tf.nn.relu(tf.matmul(x,W_fc1) + b_fc1) #発火の定義，発火関数はrelu

    W_fc2 = weight_variable([30,10])
    b_fc2 = bias_variable([10])
    h_fc2 = tf.nn.relu(tf.matmul(h_fc1,W_fc2) + b_fc2)

    keep_prob = tf.placeholder(tf.float32)

    W_fc3 = weight_variable([10,nOutput])
    b_fc3 = bias_variable([nOutput])
    y = tf.nn.relu(tf.matmul(h_fc2,W_fc3) + b_fc3)

def DefinePlaceHolder(nInput, nOutput):
    global x, y_, w, b
    x = tf.placeholder(tf.float32, shape=[None, nInput], name = "input")
    y_ = tf.placeholder(tf.float32, shape=[None, nOutput], name = "output")

    w = tf.Variable(tf.zeros([nInput, nOutput]))  #weight
    b = tf.Variable(tf.zeros([nOutput])) #bias

def InitiateModel():
    global sess, cross_entropy, train_step
    #y_ = training data , y = predection data 
    cross_entropy = tf.reduce_mean(tf.reduce_sum(tf.square(y - y_)))
    #train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    #sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer()) # Initialize variables

def Batch_NN_model(sess, input_data_np, output_data_np, train_step, cross_entropy, epoch):
    global losses
    batch_size = 8 # バッチサイズ  
    num_data = 256
    losses=[]
    print(num_data)
    for i in range(epoch):
        for idx in range(0, num_data, batch_size):
            Input_batch = input_data_np[idx:idx+batch_size]
            Output_batch = output_data_np[idx:idx+batch_size]
           # train_step.run(feed_dict={x:Input_batch, y_:Output_batch})
            train_step.run(feed_dict={x:Input_batch, y_:Output_batch, keep_prob:1.0})
        if i%200 == 0:
          #  loss = sess.run(cross_entropy,feed_dict={x:Input, y_:Output})
            loss = sess.run(cross_entropy,feed_dict={x:input_data_np, y_:output_data_np, keep_prob:1.0})
            print("set %d"%(i))
            print("loss ={} ".format(loss))
            losses.append(loss)
    print('end')

def OnlineLearning(df_KID, KID_NAME):

    cross_entropy = tf.reduce_mean(tf.reduce_sum(tf.square(y - y_)))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer()) # Initialize variables

    input_data_np_KID=MakeInputData(df_KID,KID_NAME)
    output_data_np_KID=MakeOutputData(df_KID, KID_NAME)
    start=time.time()
    Batch_NN_model(sess, input_data_np_KID, output_data_np_KID, train_step, cross_entropy, epoch)
    learning_time=time.time()-start
    print("Learning time:", learning_time)
    print("W_fc1 : %s" % W_fc1.eval())
    save_path_local=saver.save(sess, './'+KID_NAME+'_NN_model/'+KID_NAME+'_NN_model.ckpt')
    #save_path_vali=saver.save(sess, '/home/nakajo/GT_Kids/Results/ML_data/vailidation_ver1/ml_data/KID1_NN_model/KID1_NN_model.ckpt')
    print("Model saved in path: %s" % os.getcwd()+save_path_local)

    with open(os.getcwd()+"/"+KID_NAME+"_NN_model/KID_NN_model_wfc1.csv", mode='w') as f:
        writer=csv.writer(f, lineterminator='/n')
        for i in W_fc1.eval():
            writer.writerow([i])

    sess.close()
    tf.reset_default_graph()


if __name__=='__main__':

    # Training
    path="/home/nakajo/GT_Kids/Results/ML_data/data_5_ver1/ml_data/"
    df_KID1_ml=pd.read_pickle(path+"df_KID1_ml.pkl")
    df_KID3_ml=pd.read_pickle(path+"df_KID3_ml.pkl")
    df_KID7_ml=pd.read_pickle(path+"df_KID7_ml.pkl")
    df_KID9_ml=pd.read_pickle(path+"df_KID9_ml.pkl")
    df_KID11_ml=pd.read_pickle(path+"df_KID11_ml.pkl")

    nInput=5
    nOutput=2
    epoch=1000
    DefinePlaceHolder(nInput, nOutput)
    DetermineNeuron(nInput, nOutput)
    saver = tf.train.Saver()
    OnlineLearning(df_KID1_ml, "KID1")

    nInput=5
    nOutput=2
    epoch=1000
    DefinePlaceHolder(nInput, nOutput)
    DetermineNeuron(nInput, nOutput)
    saver = tf.train.Saver()
    OnlineLearning(df_KID3_ml, "KID3")

    nInput=5
    nOutput=2
    epoch=1000
    DefinePlaceHolder(nInput, nOutput)
    DetermineNeuron(nInput, nOutput)
    saver = tf.train.Saver()
    OnlineLearning(df_KID7_ml, "KID7")

    nInput=5
    nOutput=2
    epoch=1000
    DefinePlaceHolder(nInput, nOutput)
    DetermineNeuron(nInput, nOutput)
    saver = tf.train.Saver()
    OnlineLearning(df_KID9_ml, "KID9")

    nInput=5
    nOutput=2
    epoch=1000
    DefinePlaceHolder(nInput, nOutput)
    DetermineNeuron(nInput, nOutput)
    saver = tf.train.Saver()
    OnlineLearning(df_KID11_ml, "KID11")
