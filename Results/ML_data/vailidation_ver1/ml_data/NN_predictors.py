import pandas as pd
from KID_prediction_class import NN_KID_model
import numpy as np


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


if __name__=="__main__":

    # Need to build system that gets temperature by gRPC based communication below

    df_KID1_ml=pd.read_pickle("/home/nakajo/GT_Kids/Results/ML_data/vailidation_ver1/ml_data/df_KID1_ml.pkl")
    df_KID3_ml=pd.read_pickle("/home/nakajo/GT_Kids/Results/ML_data/vailidation_ver1/ml_data/df_KID3_ml.pkl")
    df_KID7_ml=pd.read_pickle("/home/nakajo/GT_Kids/Results/ML_data/vailidation_ver1/ml_data/df_KID7_ml.pkl")
    df_KID9_ml=pd.read_pickle("/home/nakajo/GT_Kids/Results/ML_data/vailidation_ver1/ml_data/df_KID9_ml.pkl")
    df_KID11_ml=pd.read_pickle("/home/nakajo/GT_Kids/Results/ML_data/vailidation_ver1/ml_data/df_KID11_ml.pkl")
    
    # Initiating NN prediction model object
    NN_KID1_model=NN_KID_model("KID1")
    NN_KID3_model=NN_KID_model("KID3")
    NN_KID7_model=NN_KID_model("KID7")
    NN_KID9_model=NN_KID_model("KID9")
    NN_KID11_model=NN_KID_model("KID11")

    input_data_KID1=MakeInputData(df_KID1_ml, "KID1")
    input_data_KID3=MakeInputData(df_KID3_ml, "KID3")
    input_data_KID7=MakeInputData(df_KID7_ml, "KID7")
    input_data_KID9=MakeInputData(df_KID9_ml, "KID9")
    input_data_KID11=MakeInputData(df_KID11_ml, "KID11")

    current_state_KID1=input_data_KID1[1:2, :]
    current_state_KID3=input_data_KID3[1:2, :]
    current_state_KID7=input_data_KID7[1:2, :]
    current_state_KID9=input_data_KID9[1:2, :]
    current_state_KID11=input_data_KID11[1:2, :]

    print("Current state (KID1):", current_state_KID1)
    print("Current state (KID3):", current_state_KID3)
    print("Current state (KID7):", current_state_KID7)
    print("Current state (KID9):", current_state_KID9)
    print("Current state (KID11):", current_state_KID11)

    future_KID1=NN_KID1_model.predict(current_state_KID1)
    future_KID3=NN_KID3_model.predict(current_state_KID3)
    future_KID7=NN_KID7_model.predict(current_state_KID7)
    future_KID9=NN_KID9_model.predict(current_state_KID9)
    future_KID11=NN_KID11_model.predict(current_state_KID11)

    print(future_KID1[:, 0], future_KID3[:, 0], future_KID7[:, 0], future_KID9[:, 0], future_KID11[:, 0])
