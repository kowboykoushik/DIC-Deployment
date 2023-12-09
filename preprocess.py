#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def preprocess(df):
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import LabelEncoder
    import matplotlib.pyplot as plt
    from sklearn.cluster import KMeans
    import seaborn as sns
    from sklearn.preprocessing import MinMaxScaler
    import joblib
    df_copy = df.copy()
    if 'target' in df.columns:
        df = df.drop(columns = ['target'])
    #removing duplicate rows
    df = df.drop_duplicates()
    df_copy = df_copy.drop_duplicates()
    #Converting the wtkg column into numeric and handling non-numeric values with NaN Values
    df['wtkg'] = pd.to_numeric(df['wtkg'], errors='coerce')
    df['time'] = pd.to_numeric(df['time'], errors='coerce')
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    df['preanti'] = pd.to_numeric(df['preanti'], errors='coerce')
    df['strat'] = pd.to_numeric(df['strat'], errors='coerce')
    df['cd40'] = pd.to_numeric(df['cd40'], errors='coerce')
    df['cd420'] = pd.to_numeric(df['cd420'], errors='coerce')
    df['cd80'] = pd.to_numeric(df['cd80'], errors='coerce')
    df['cd820'] = pd.to_numeric(df['cd820'], errors='coerce')
    df['karnof'] = pd.to_numeric(df['karnof'], errors='coerce')
    #converting all the text columns into lowercase
    df['trt']=df['trt'].str.lower()
    df['hemo']=df['hemo'].str.lower()
    df['homo']=df['homo'].str.lower()
    df['drugs']=df['drugs'].str.lower()
    df['oprior']=df['oprior'].str.lower()
    df['z30']=df['z30'].str.lower()
    df['zprior']=df['zprior'].str.lower()
    df['race']=df['race'].str.lower()
    df['gender']=df['gender'].str.lower()
    df['str2']=df['str2'].str.lower()
    df['symptom']=df['symptom'].str.lower()
    df['treat']=df['treat'].str.lower()
    df['offtrt']=df['offtrt'].str.lower()
    #we replace typographical errors with correct values or we replace it with NaN
    df = df.replace({
                'trt':{'zdv and zal':'zdv + zal','zdv alone':'zdv', 'ddl only':'ddl', 'ddl alone':'ddl', 'zdv only':'zdv'},
                'hemo':{'nope':'no'},
                'homo':{'yup':'yes'},
                'z30':{'nope':'no'},
                'drugs' : {'nah': 'no', 'yup': 'yes', 'none': np.nan},
                'race' : {'non white': 'non-white', 'asian': 'non-white', 'african american': 'non-white', 'african': 'non-white', 'latinx': 'non-white', 'aapi': 'non-white', 'arab': 'non-white', 'brown': 'non-white', 'european': 'white'},
                'symptom' : {'sympomatic': 'symp', 'asymptomatic': 'asymp', 'asympomatic': 'asymp', 'symptomatic': 'symp'},
                'treat' : {'zdv alone': 'zdv', 'zdv only': 'zdv', 'other' : 'others'}
                })
    #dropping the NaN values from text columns
    columns_nanValues = ['trt', 'hemo', 'homo','drugs','oprior','z30','zprior','race','gender','str2','symptom','treat','offtrt']
    df = df.dropna(subset=columns_nanValues)
    remaining_indices = df.index
    df_copy = df_copy.iloc[remaining_indices]
    # Apply imputation values to a new dataset
    loaded_imputation_values = joblib.load('imputer.pkl')
    df = df.fillna(value=loaded_imputation_values)
    df.wtkg = df.wtkg.apply(lambda x: round(float(x), 2))
    # Apply the loaded label encoders to a new dataset
    loaded_label_encoders = joblib.load('label_encoders.pkl')
    for col, label_encoder in loaded_label_encoders.items():
        if col != 'target':
            df[col] = label_encoder.transform(df[col])
    df['strat'] = pd.cut(df['strat'], bins = np.array([-np.inf, 0, 52, np.inf]), labels = ['1','2','3'], right = True).astype(int)
    loaded_encoding_info = joblib.load('one_hot_encoding_info.pkl')
    df = pd.get_dummies(df, columns=loaded_encoding_info['columns'], dtype=int)
    # Ensure that the new dataset has the same dummy columns as the original dataset
    for dummy_col in loaded_encoding_info['dummies']:
        if dummy_col not in df.columns:
            df[dummy_col] = 0
    columnsList = ['time','wtkg','age','karnof','preanti','cd40','cd420','cd80','cd820']
    # Normalize a new dataset using the loaded scaler
    loaded_scaler = joblib.load('minmax_scaler.pkl')
    df[columnsList] = loaded_scaler.transform(df[columnsList])
    df = df.drop(columns = ['pid','hemo','zprior','cd820'])
    df = df[['time', 'age', 'wtkg', 'homo', 'drugs', 'karnof', 'oprior', 'z30', 'preanti', 'race', 'gender', 'symptom', 'treat', 'offtrt', 'cd40', 'cd420', 'cd80', 'str2', 'trt_ddl', 'trt_zdv', 'trt_zdv + ddl', 'trt_zdv + zal', 'strat_1', 'strat_2', 'strat_3']]
    return df, df_copy

