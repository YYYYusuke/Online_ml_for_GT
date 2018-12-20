# coding: utf-8
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime
import time
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from sklearn import linear_model, preprocessing, cross_validation, svm


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

def ShowGraph(input_data_np, output_data_np, KID_NAME):
    #予測値(y)の算出
    NN_predict=sess.run(y, feed_dict={x:input_data_np, y_: output_data_np})
    plt.plot(NN_predict[256:, 0], label="NN prediction")
    plt.plot(output_data_np[256:, 0], label="Measured data")
    plt.xlabel("Data point (time_scale)")
    plt.ylabel("Temperature (deg C)")
    plt.title("Model_for_"+ KID_NAME)
    plt.legend()
    print("MSE:", mean_squared_error(output_data_np, NN_predict))

def DetermineNeuron(nInput, nOutput):
    #非線形回帰モデル
    # 30分類器 full=connection
    global W_fc1, b_fc1, h_fc1, W_fc2, b_fc2, h_fc2, keep_prob, W_fc3, b_fc3, y
    #第１層
    W_fc1 = weight_variable([nInput,30])
    b_fc1 = bias_variable([30])
    h_fc1 = tf.nn.relu(tf.matmul(x,W_fc1) + b_fc1) #発火の定義，発火関数はrelu

    #第２層
    W_fc2 = weight_variable([30,10])
    b_fc2 = bias_variable([10])
    h_fc2 = tf.nn.relu(tf.matmul(h_fc1,W_fc2) + b_fc2)

    keep_prob = tf.placeholder(tf.float32)  

    #最終層（OutPut）
    W_fc3 = weight_variable([10,nOutput])
    b_fc3 = bias_variable([nOutput])
    y = tf.nn.relu(tf.matmul(h_fc2,W_fc3) + b_fc3)

def DefinePlaceHolder(nInput, nOutput):
    global x, y_, w, b
    x = tf.placeholder(tf.float32, shape=[None, nInput], name = "input")
    #x = tf.placeholder(tf.float32, shape=[nInput], name = "input")
    y_ = tf.placeholder(tf.float32, shape=[None, nOutput], name = "output") 
    #y_ = tf.placeholder(tf.float32, shape=[nOutput], name = "output") 

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
    
def ResetGraph():
    tf.reset_default_graph()  
    
def ShowGraph(input_data_np, output_data_np, KID_NAME, epoch):
    #予測値(y)の算出
    plt.figure(figsize=(15,5))
    plt.subplot(1, 2, 1)
    NN_predict=sess.run(y, feed_dict={x:input_data_np, y_: output_data_np})
    plt.plot(NN_predict[256:, 0], label="NN prediction")
    plt.plot(output_data_np[256:, 0], label="Measured data")
    plt.xlabel("Data point (time_scale)")
    plt.ylabel("Temperature (deg C)")
    plt.title("Model_for_"+ KID_NAME)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.scatter(NN_predict[256:, 0], output_data_np[256:, 0],  linestyle="solid", marker="o", label="NN_predict")
    plt.xlabel("True values (deg C)") 
    plt.ylabel("Predictions (deg C)")
    plt.title('NN_learning_process')
    plt.legend(loc ="upper right")
    plt.axis('equal')
    plt.axis('square')
    plt.xlim([55,plt.xlim()[1]])
    plt.ylim([55,plt.ylim()[1]])
    _ = plt.plot([55, 80], [55, 80])
    plt.show()
    print("MSE:", mean_squared_error(output_data_np, NN_predict))

def Main(df_KID_ml, KID_NAME):
    tf.reset_default_graph()
    path=os.getcwd()
    nInput=5
    nOutput=2
    epoch=10000
    DefinePlaceHolder(nInput, nOutput)
    DetermineNeuron(nInput, nOutput)
    saver = tf.train.Saver()

    with tf.Session() as sess:  # your session object
        InitiateModel()
        saver = tf.train.import_meta_graph(path+'/'+KID_NAME+'_NN_model.ckpt.meta')
        saver.restore(sess, tf.train.latest_checkpoint('./'))
        #sess.run(W_fc1)
        print("Model restored.")
        # Check the values of the variables
        print("W_fc1 : %s" % W_fc1.eval())
        print("W_fc2 : %s" % W_fc2.eval())
        input_data_np_KID=MakeInputData(df_KID_ml, KID_NAME)
        output_data_np_KID=MakeOutputData(df_KID_ml, KID_NAME)
        NN_predict=sess.run(y, feed_dict={x:input_data_np_KID, y_: output_data_np_KID})    
        NN_predict = y.eval(feed_dict={x:input_data_np_KID, y_: output_data_np_KID})
        ShowGraph(input_data_np_KID, output_data_np_KID, KID_NAME, epoch)


if __name__=="__main__":

    df_KID1_ml=pd.read_pickle("/home/nakajo/GT_Kids/Results/ML_data/vailidation_ver1/ml_data/df_KID1_ml.pkl")
    df_KID3_ml=pd.read_pickle("/home/nakajo/GT_Kids/Results/ML_data/vailidation_ver1/ml_data/df_KID3_ml.pkl")
    df_KID7_ml=pd.read_pickle("/home/nakajo/GT_Kids/Results/ML_data/vailidation_ver1/ml_data/df_KID7_ml.pkl")
    df_KID9_ml=pd.read_pickle("/home/nakajo/GT_Kids/Results/ML_data/vailidation_ver1/ml_data/df_KID9_ml.pkl")
    df_KID11_ml=pd.read_pickle("/home/nakajo/GT_Kids/Results/ML_data/vailidation_ver1/ml_data/df_KID11_ml.pkl")

    tf.reset_default_graph()
    path="/home/nakajo/GT_Kids/Results/ML_data/vailidation_ver1/ml_data/"
    nInput=5
    nOutput=2
    epoch=10000
    DefinePlaceHolder(nInput, nOutput)
    DetermineNeuron(nInput, nOutput)
    saver = tf.train.Saver()

    with tf.Session() as sess:  # your session object
        InitiateModel()
        saver = tf.train.import_meta_graph(path+'/KID11_NN_model/KID11_NN_model.ckpt.meta')
        saver.restore(sess, tf.train.latest_checkpoint(path+'/KID11_NN_model'))
        print("Model restored.")
        # Check the values of the variables
        print("W_fc1 : %s" % W_fc1.eval())
        #print("W_fc2 : %s" % W_fc2.eval())
        input_data_np_KID=MakeInputData(df_KID11_ml,"KID11")
        output_data_np_KID=MakeOutputData(df_KID11_ml, "KID11")
        NN_predict=sess.run(y, feed_dict={x:input_data_np_KID, y_: output_data_np_KID})    
        NN_predict = y.eval(feed_dict={x:input_data_np_KID, y_: output_data_np_KID})
        print("MSE:", mean_squared_error(output_data_np_KID, NN_predict))
