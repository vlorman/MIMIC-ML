#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 20 19:25:34 2021

@author: vitalylorman
"""

import os
import pandas as pd
import numpy as np
import dill
from sklearn.model_selection import train_test_split
import datetime

print("Starting processing...")

os.chdir("/Volumes/Seagate/physionet")
dill.load_session("/Volumes/Seagate/physionet/6_17_2021")

#patients=pd.read_hdf("all_hourly_data.h5", 'patients')
#vitals_labs=pd.read_hdf("all_hourly_data.h5", 'vitals_labs')


ID_COLS           = ['subject_id', 'hadm_id', 'icustay_id']
def impute(df):
    idx = pd.IndexSlice
    df = df.copy()
    #if len(df.columns.names) > 2: df.columns = df.columns.droplevel(('label', 'LEVEL1', 'LEVEL2'))
    
    df_out = df.loc[:, idx[:, ['mean']]].copy()
    icustay_means = df_out.loc[:, idx[:, 'mean']].groupby(ID_COLS).mean()
    
    df_out.loc[:,idx[:,'mean']] = df_out.loc[:,idx[:,'mean']].groupby(ID_COLS).fillna(
        method='ffill'
    ).groupby(ID_COLS).fillna(icustay_means).fillna(0)    
    df_out.sort_index(axis=1, inplace=True)
    return df_out


def time_filter(df):
    df['CHARTTIME']=pd.to_datetime(df['CHARTTIME'])
    df['intime']=pd.to_datetime(df['intime'])
    df['time12']=df['intime']+datetime.timedelta(hours=12)
    df=df[df['CHARTTIME']<df['time12']]
    df=df.sort_values(by=["CHARTTIME", "hadm_id"])
    df=df[["TEXT"]]
    df=pd.DataFrame(df.groupby(['subject_id', 'hadm_id'])['TEXT'].apply('\n'.join))
    return df






windows=[12, 24, 48]

vl_index=vitals_labs.index.to_frame(index=False)
ids=[vlm_index[vlm_index['hours_in']>window+5]['icustay_id'].unique() for window in windows]
vl=[vitals_labs[np.isin(vitals_labs.index.get_level_values('icustay_id'), id_list)] for id_list in ids]

vl[0]=vl[0][vl[0].index.get_level_values('hours_in')<=12]
vl[1]=vl[1][vl[1].index.get_level_values('hours_in')<=24]
vl[2]=vl[2][vl[2].index.get_level_values('hours_in')<=48]

vl_imp=[impute(df) for df in vl]


for df in vl_imp:
    df.index=df.index.set_levels(df.index.levels[-1].astype(str), level=-1)

vl_flat=[df.pivot_table(index=ID_COLS, columns=['hours_in']) for df in vl_imp]
for df in vl_flat:
    df.columns=['_'.join(col).strip() for col in df.columns.values]
    
features_a=['gender', 'ethnicity', 'age', 'mort_icu']
patients_keep=patients.iloc[:, np.isin(patients.columns,features_a)].copy()
patients_keep["age"]=pd.qcut(patients_keep["age"], q=10)

vl_patients=[df.merge(patients_keep, how='left',
                     left_on=df.index.names[:3],
                     right_on=patients_keep.index.names) for df in vl_flat]    
    
noteevents=pd.read_csv('NOTEEVENTS.csv', engine='python', error_bad_lines=False)
noteevents=noteevents.set_index(['SUBJECT_ID', 'HADM_ID'])
noteevents=noteevents.drop(columns='ROW_ID')
noteevents.index.names=['subject_id', 'hadm_id']

patient_notes=noteevents.merge(patients, how='left', on=['subject_id', 'hadm_id'])

patient_notes=time_filter(patient_notes)


merged=[df.merge(patient_notes, how='left', on=['subject_id', 'hadm_id']) for df in vl_patients]


merged=[pd.get_dummies(df, columns=['age', 'gender', 'ethnicity']) for df in merged]


hadm_ids=merged[0].index.get_level_values('hadm_id')

train_ids, test_ids=train_test_split(hadm_ids, test_size=0.2, stratify=merged[0]['mort_icu'])

train=[df[np.isin(df.index.get_level_values('hadm_id'), train_ids)] for df in merged]
test=[df[np.isin(df.index.get_level_values('hadm_id'), test_ids)] for df in merged]


train_shapes=[df.shape for df in train]
test_shapes=[df. shape for df in test]
print(train_shapes, test_shapes)

X_train=[df.drop('mort_icu', axis=1) for df in train]
X_test=[df.drop('mort_icu', axis=1) for df in test]
y_train=[df[['mort_icu']] for df in train]
y_test=[df[['mort_icu']] for df in test]

for i in range(0,3):
    print(y_train[i]['mort_icu'].sum()/train_shapes[i][0])
    print(y_test[i]['mort_icu'].sum()/test_shapes[i][0])
    
    
print("Saving to csv...")
X_train[0].to_csv("X_train_12.csv")
X_train[1].to_csv("X_train_24.csv")
X_train[2].to_csv("X_train_48.csv")

X_test[0].to_csv("X_test_12.csv")
X_test[1].to_csv("X_test_24.csv")
X_test[2].to_csv("X_test_48.csv")

y_train[0].to_csv("y_train_12.csv")
y_train[1].to_csv("y_train_24.csv")
y_train[2].to_csv("y_train_48.csv")

y_test[0].to_csv("y_test_12.csv")
y_test[1].to_csv("y_test_24.csv")
y_test[2].to_csv("y_test_48.csv")
print("Saved to csv!")
print("Dumping session...")

dill.dump_session("/Volumes/Seagate/physionet/6_20_2021")

print("Done!")





    