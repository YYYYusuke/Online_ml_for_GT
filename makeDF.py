# This script is written for making data-frame and storing data as pkl files.

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import shutil
import sys


def Get_table(KID_NAME):
    try:
        df_algo_time=pd.read_csv("/home/nakajo/GT_Kids/"+ KID_NAME+"/local_logs/algo_time.csv", header=None, names=["Latency (s)"])
    except:
        pass
    df_fan=pd.read_csv("/home/nakajo/GT_Kids/"+ KID_NAME+"/local_logs/FANtest.csv", header=None, names=['time (s)', 'FAN (%)'])
    df_PS=pd.read_csv("/home/nakajo/GT_Kids/"+ KID_NAME+"/local_logs/PStest.csv", header=None, names=['time (s)', 'PS'])
    df_CPU_util=pd.read_csv("/home/nakajo/GT_Kids/"+ KID_NAME+"/local_logs/CPU_util_test.csv", header=None, names=['time (s)', 'CPU_utilization'])
    df_sensors_temp=pd.read_csv("/home/nakajo/GT_Kids/"+ KID_NAME+"/local_logs/CPU_temp_sensorstest.csv", header=None, names=['time (s)', 'CPU_temperature(degC)'])
    A=pd.merge(df_fan, df_PS, on='time (s)', how='outer')
    B=pd.merge(df_CPU_util, df_sensors_temp, on='time (s)', how='outer')
    KID_table=pd.merge(A, B, on='time (s)', how='outer').set_index('time (s)')
    return KID_table

def GetServeTime(KID_NAME):
    df_serve_time=pd.read_csv("/home/nakajo/GT_Kids/"+ KID_NAME+"/local_logs/serve_time.csv", header=None, names=["Server Processing time (s)"])
    return df_serve_time

def GetServeTimePD():

    KID1_serve_time=GetServeTime("KID1")
    KID3_serve_time=GetServeTime("KID3")
    KID7_serve_time=GetServeTime("KID7")
    KID9_serve_time=GetServeTime("KID9")
    KID11_serve_time=GetServeTime("KID11")
    servetime_df=pd.concat([KID1_serve_time, KID3_serve_time, KID7_serve_time, KID9_serve_time, KID11_serve_time], axis=1)
    servetime_df.columns=['KID1_serve_time', 'KID3_serve_time', 'KID7_serve_time', 'KID9_serve_time', 'KID11_serve_time']

    KID1_job_intensity=GetJobIntensity("KID1")
    KID3_job_intensity=GetJobIntensity("KID3")
    KID7_job_intensity=GetJobIntensity("KID7")
    KID9_job_intensity=GetJobIntensity("KID9")
    KID11_job_intensity=GetJobIntensity("KID11")
    job_inte_df=pd.concat([KID1_job_intensity, KID3_job_intensity, KID7_job_intensity, KID9_job_intensity, KID11_job_intensity], axis=1)
    job_inte_df.columns=['KID1_jobs', 'KID3_jobs', 'KID7_jobs', 'KID9_jobs', 'KID11_jobs']

    KID1_timestamp=GetJobStamp("KID1")
    KID3_timestamp=GetJobStamp("KID3")
    KID7_timestamp=GetJobStamp("KID7")
    KID9_timestamp=GetJobStamp("KID9")
    KID11_timestamp=GetJobStamp("KID11")
    timestamp_df=pd.concat([KID1_timestamp, KID3_timestamp, KID7_timestamp, KID9_timestamp, KID11_timestamp], axis=1)
    timestamp_df.columns=['KID1_timestamp', 'KID3_timestamp', 'KID7_timestamp', 'KID9_timestamp', 'KID11_timestamp']
    return pd.concat([timestamp_df, servetime_df, job_inte_df], axis=1)

def GetJobIntensity(KID_NAME):
    df_job_intensity=pd.read_csv("/home/nakajo/GT_Kids/"+ KID_NAME+"/local_logs/job_intensity.csv", header=None, names=["Job_intensity"])
    return df_job_intensity

def GetJobStamp(KID_NAME):
    df_job_timestamp=pd.read_csv("/home/nakajo/GT_Kids/"+ KID_NAME+"/local_logs/process_timestamp.csv", header=None, names=["Timestamp"])
    return df_job_timestamp

def StoreTable(Table, Table_name):
    Table.to_pickle("/home/nakajo/GT_Kids/Results/ML_data/"+Table_name+".pkl")

def GetPKLfiles():
    files=[]
    for file in os.listdir():
        if os.path.isdir(path+file):
            pass
        else:
            files.append(file)
    return files

def MoveToFolder(Folder_name):
    os.chdir(path)
    os.makedirs(Folder_name)
    files=GetPKLfiles()

    for pklfile in files:
        shutil.move(path+pklfile, path+Folder_name)

    os.chdir(path+Folder_name)
    shutil.copy('Anlysis_Template.ipynb', path)
    shutil.copy('make_learning_data.py', path)


if __name__ == '__main__':

    args=sys.argv
    print("Export folder name is:", args[1])
    Folder_name=args[1]
    path="/home/nakajo/GT_Kids/Results/ML_data/"

    print("Learnig data is going to be produce.")
    
    print("Making data frame  ...")
    
    KID1_table_RR=Get_table("KID1").sort_index().interpolate()
    KID3_table_RR=Get_table("KID3").sort_index().interpolate()
    KID7_table_RR=Get_table("KID7").sort_index().interpolate()
    KID9_table_RR=Get_table("KID9").sort_index().interpolate()
    KID11_table_RR=Get_table("KID11").sort_index().interpolate()
    KID1_serve_time_RR=GetServeTime("KID1")
    KID3_serve_time_RR=GetServeTime("KID3")
    KID7_serve_time_RR=GetServeTime("KID7")
    KID9_serve_time_RR=GetServeTime("KID9")
    KID11_serve_time_RR=GetServeTime("KID11")
    df_jobs_RR=GetServeTimePD()
    RR_serve_time=pd.concat([KID1_serve_time_RR, KID3_serve_time_RR, KID7_serve_time_RR, KID9_serve_time_RR, KID11_serve_time_RR]).reset_index()
    print("Serving average time is ", RR_serve_time.mean())

    print("Storing data to results file ...")

    StoreTable(KID1_table_RR, "KID1_RR")
    StoreTable(KID3_table_RR, "KID3_RR")
    StoreTable(KID7_table_RR, "KID7_RR")
    StoreTable(KID9_table_RR, "KID9_RR")
    StoreTable(KID11_table_RR, "KID11_RR")
    StoreTable(KID1_serve_time_RR, "KID1_serve_time_RR")
    StoreTable(KID3_serve_time_RR, "KID3_serve_time_RR")
    StoreTable(KID7_serve_time_RR, "KID7_serve_time_RR")
    StoreTable(KID9_serve_time_RR, "KID9_serve_time_RR")
    StoreTable(KID11_serve_time_RR, "KID11_serve_time_RR")
    StoreTable(df_jobs_RR, "df_jobs_RR")

    MoveToFolder(Folder_name)

