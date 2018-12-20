import pandas as pd
import numpy as np
import os
import seaborn as sns
from datetime import datetime
import pickle
import sys


def __datetime(date_str):
    return datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S.%f')

def MakeElapsedTable(Table):
    ELT=pd.concat([Table.reset_index(), pd.DataFrame(EpTget(Table), columns=['Elapsed time (s)'])], axis=1)
    ELT=ELT.set_index('Elapsed time (s)')
    return ELT

def EpTget(KID_table_datetime):
    ElapsedTime=[]
    for i in range(len(KID_table_datetime.index)):
        tmp=__datetime(KID_table_datetime.index[i]) - __datetime(KID_table_datetime.index[0])
        ElapsedTime.append(tmp.total_seconds())
    return ElapsedTime

def MakeDataForML(Table, KID_NAME):
    RR_df=df_jobs_RR.loc[:,[KID_NAME+'_timestamp', KID_NAME+'_serve_time', KID_NAME+'_jobs']]
    RR_df=RR_df.rename(columns={KID_NAME+'_timestamp': 'time (s)'})
    Table_RR_rst=Table.reset_index()
    hoge=pd.concat([RR_df, Table_RR_rst])
    hoge=hoge.sort_values(by=["time (s)"], ascending=True)
    hoge=hoge.set_index("time (s)")
    Elapsed_hoge=MakeElapsedTable(hoge)
    fuga=Elapsed_hoge.reset_index()
    fuga['Elapsed time (s)']=fuga['Elapsed time (s)'].astype(np.int64)
    fuga_grouped=fuga.groupby('Elapsed time (s)').agg(np.mean)
    fuga_grouped_NA=fuga_grouped.dropna(how='any')
    return fuga_grouped_NA

if __name__=='__main__':

    path="/home/nakajo/GT_Kids/Results/ML_data/"
    args=sys.argv
    print("Inport folder name is:", args[1])
    Folder_name=args[1]


    print("Reading data .................")
    KID1_table_RR=pd.read_pickle(path+Folder_name+"/KID1_RR.pkl")
    KID3_table_RR=pd.read_pickle(path+Folder_name+"/KID3_RR.pkl")
    KID7_table_RR=pd.read_pickle(path+Folder_name+"/KID7_RR.pkl")
    KID9_table_RR=pd.read_pickle(path+Folder_name+"/KID9_RR.pkl")
    KID11_table_RR=pd.read_pickle(path+Folder_name+"/KID11_RR.pkl")
    KID1_serve_time_RR=pd.read_pickle(path+Folder_name+"/KID1_serve_time_RR.pkl")
    KID3_serve_time_RR=pd.read_pickle(path+Folder_name+"/KID3_serve_time_RR.pkl")
    KID7_serve_time_RR=pd.read_pickle(path+Folder_name+"/KID7_serve_time_RR.pkl")
    KID9_serve_time_RR=pd.read_pickle(path+Folder_name+"/KID9_serve_time_RR.pkl")
    KID11_serve_time_RR=pd.read_pickle(path+Folder_name+"/KID11_serve_time_RR.pkl")
    df_jobs_RR=pd.read_pickle(path+Folder_name+"/df_jobs_RR.pkl")
    
    print("Processing data..................")
    KID1_ml_data=MakeDataForML(KID1_table_RR, "KID1")
    KID3_ml_data=MakeDataForML(KID3_table_RR, "KID3")
    KID7_ml_data=MakeDataForML(KID7_table_RR, "KID7")
    KID9_ml_data=MakeDataForML(KID9_table_RR, "KID9")
    KID11_ml_data=MakeDataForML(KID11_table_RR, "KID11")

    print("Exporting dataframe .............")
    os.chdir(path+Folder_name)
    os.makedirs("ml_data")
    KID1_ml_data.to_pickle(path+Folder_name+"/ml_data/df_KID1_ml.pkl")
    KID3_ml_data.to_pickle(path+Folder_name+"/ml_data/df_KID3_ml.pkl")
    KID7_ml_data.to_pickle(path+Folder_name+"/ml_data/df_KID7_ml.pkl")
    KID9_ml_data.to_pickle(path+Folder_name+"/ml_data/df_KID9_ml.pkl")
    KID11_ml_data.to_pickle(path+Folder_name+"/ml_data/df_KID11_ml.pkl")

