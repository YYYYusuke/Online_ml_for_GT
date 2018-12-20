#!/bin/bash

echo "Reading data...."
sshpass -p "GALincoln8826" scp -r ynakajo6@130.207.110.11:/nethome/ynakajo6/local_logs /nethome/ynakajo6/GT_ML/dB_csv/KID1/
sshpass -p "GALincoln8826" scp -r ynakajo6@130.207.110.13:/nethome/ynakajo6/local_logs /nethome/ynakajo6/GT_ML/dB_csv/KID3/
sshpass -p "GALincoln8826" scp -r ynakajo6@130.207.110.17:/nethome/ynakajo6/local_logs /nethome/ynakajo6/GT_ML/dB_csv/KID7/
sshpass -p "GALincoln8826" scp -r ynakajo6@130.207.110.19:/nethome/ynakajo6/local_logs /nethome/ynakajo6/GT_ML/dB_csv/KID9/
sshpass -p "GALincoln8826" scp -r ynakajo6@130.207.110.21:/nethome/ynakajo6/local_logs /nethome/ynakajo6/GT_ML/dB_csv/KID11/

