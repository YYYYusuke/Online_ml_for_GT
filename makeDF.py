# This script is written for making data-frame and storing data as pkl files.
import pandas as pd
import numpy as np
import os
import shutil
import sys

def Get_table(KID_NAME, path):
    try:
        df_algo_time=pd.read_csv(path+KID_NAME+"/local_logs/algo_time.csv", header=None, names=["Latency (s)"])
    except:
        pass
    df_fan=pd.read_csv(path+KID_NAME+"/local_logs/FANtest.csv", header=None, names=['time (s)', 'FAN (%)'])
    df_PS=pd.read_csv(path+KID_NAME+"/local_logs/PStest.csv", header=None, names=['time (s)', 'PS'])
    df_CPU_util=pd.read_csv(path+KID_NAME+"/local_logs/CPU_util_test.csv", header=None, names=['time (s)', 'CPU_utilization'])
    df_sensors_temp=pd.read_csv(path+ KID_NAME+"/local_logs/CPU_temp_sensorstest.csv", header=None, names=['time (s)', 'CPU_temperature(degC)'])
    A=pd.merge(df_fan, df_PS, on='time (s)', how='outer')
    B=pd.merge(df_CPU_util, df_sensors_temp, on='time (s)', how='outer')
    KID_table=pd.merge(A, B, on='time (s)', how='outer').set_index('time (s)')
    return KID_table

def GetServeTime(KID_NAME, path):
    df_serve_time=pd.read_csv(path+KID_NAME+"/local_logs/serve_time.csv", header=None, names=["Server Processing time (s)"])
    return df_serve_time

def GetServeTimePD(path):
    KID1_serve_time=GetServeTime("KID1", path)
    KID3_serve_time=GetServeTime("KID3", path)
    KID7_serve_time=GetServeTime("KID7", path)
    KID9_serve_time=GetServeTime("KID9", path)
    KID11_serve_time=GetServeTime("KID11", path)
    servetime_df=pd.concat([KID1_serve_time, KID3_serve_time, KID7_serve_time, KID9_serve_time, KID11_serve_time], axis=1)
    servetime_df.columns=['KID1_serve_time', 'KID3_serve_time', 'KID7_serve_time', 'KID9_serve_time', 'KID11_serve_time']

    KID1_job_intensity=GetJobIntensity("KID1", path)
    KID3_job_intensity=GetJobIntensity("KID3", path)
    KID7_job_intensity=GetJobIntensity("KID7", path)
    KID9_job_intensity=GetJobIntensity("KID9", path)
    KID11_job_intensity=GetJobIntensity("KID11", path)
    job_inte_df=pd.concat([KID1_job_intensity, KID3_job_intensity, KID7_job_intensity, KID9_job_intensity, KID11_job_intensity], axis=1)
    job_inte_df.columns=['KID1_jobs', 'KID3_jobs', 'KID7_jobs', 'KID9_jobs', 'KID11_jobs']

    KID1_timestamp=GetJobStamp("KID1", path)
    KID3_timestamp=GetJobStamp("KID3", path)
    KID7_timestamp=GetJobStamp("KID7", path)
    KID9_timestamp=GetJobStamp("KID9", path)
    KID11_timestamp=GetJobStamp("KID11", path)
    timestamp_df=pd.concat([KID1_timestamp, KID3_timestamp, KID7_timestamp, KID9_timestamp, KID11_timestamp], axis=1)
    timestamp_df.columns=['KID1_timestamp', 'KID3_timestamp', 'KID7_timestamp', 'KID9_timestamp', 'KID11_timestamp']
    return pd.concat([timestamp_df, servetime_df, job_inte_df], axis=1)

def GetJobIntensity(KID_NAME, path):
    df_job_intensity=pd.read_csv(path+KID_NAME+"/local_logs/job_intensity.csv", header=None, names=["Job_intensity"])
    return df_job_intensity

def GetJobStamp(KID_NAME, path):
    df_job_timestamp=pd.read_csv(path+KID_NAME+"/local_logs/process_timestamp.csv", header=None, names=["Timestamp"])
    return df_job_timestamp

def StoreTable(Table, Table_name, path):
    Table.to_pickle(path+Table_name+".pkl")

def GetPKLfiles(path):
    files=[]
    for file in os.listdir(path):
        if os.path.isdir(path+file):
            pass
        else:
            files.append(file)
    return files

def MoveToFolder(Folder_name, path):
    os.chdir(path)
    os.makedirs(Folder_name)
    files=GetPKLfiles(path)

    for pklfile in files:
        shutil.move(path+pklfile, path+Folder_name)
    
    shutil.move(Folder_name, "/nethome/ynakajo6/GT_ML/ML_data/dB_pd/")


if __name__ == '__main__':

    args=sys.argv
    print("Export folder name is:", args[1])
    Folder_name=args[1]
    path_dB="/nethome/ynakajo6/GT_ML/dB_csv/"
    #path_ml

    print("Training data is going to be produced.")
    
    print("Making data frame  ...")
    KID1_table_RR=Get_table("KID1", path_dB).sort_index().interpolate()
    KID3_table_RR=Get_table("KID3", path_dB).sort_index().interpolate()
    KID7_table_RR=Get_table("KID7", path_dB).sort_index().interpolate()
    KID9_table_RR=Get_table("KID9", path_dB).sort_index().interpolate()
    KID11_table_RR=Get_table("KID11", path_dB).sort_index().interpolate()
    KID1_serve_time_RR=GetServeTime("KID1", path_dB)
    KID3_serve_time_RR=GetServeTime("KID3", path_dB)
    KID7_serve_time_RR=GetServeTime("KID7", path_dB)
    KID9_serve_time_RR=GetServeTime("KID9", path_dB)
    KID11_serve_time_RR=GetServeTime("KID11", path_dB)
    df_jobs_RR=GetServeTimePD(path_dB)
    RR_serve_time=pd.concat([KID1_serve_time_RR, KID3_serve_time_RR, KID7_serve_time_RR, KID9_serve_time_RR, KID11_serve_time_RR]).reset_index()
    print("Serving average time is ", RR_serve_time.mean())

    print("Storing data to results file ...")
    StoreTable(KID1_table_RR, "KID1_RR", path_dB)
    StoreTable(KID3_table_RR, "KID3_RR", path_dB)
    StoreTable(KID7_table_RR, "KID7_RR", path_dB)
    StoreTable(KID9_table_RR, "KID9_RR", path_dB)
    StoreTable(KID11_table_RR, "KID11_RR", path_dB)
    StoreTable(KID1_serve_time_RR, "KID1_serve_time_RR", path_dB)
    StoreTable(KID3_serve_time_RR, "KID3_serve_time_RR", path_dB)
    StoreTable(KID7_serve_time_RR, "KID7_serve_time_RR", path_dB)
    StoreTable(KID9_serve_time_RR, "KID9_serve_time_RR", path_dB)
    StoreTable(KID11_serve_time_RR, "KID11_serve_time_RR", path_dB)
    StoreTable(df_jobs_RR, "df_jobs_RR", path_dB)

    MoveToFolder(Folder_name, path_dB)

