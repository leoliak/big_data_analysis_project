# -*- coding: utf-8 -*-
"""
@author: Leonidas Liakopoulos
"""



import sys, os
from tqdm import tqdm
from collections import Counter

from datetime import datetime
import pandas as pd
import numpy as np
from fuzzywuzzy import process

from sklearn import metrics
from sklearn import preprocessing
from sklearn import cluster
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize 
from sklearn.cluster import DBSCAN
from matplotlib import pyplot as plt
from sklearn.cluster import MeanShift 
import scipy.cluster.hierarchy as shc







part_2_1 = False
part_2_2 = True

if part_2_1:
    ## Read Beacons Dataset 
    df = pd.read_csv('data/beacons_dataset.csv', delimiter=';')
    
    ## Read all different rooms
    p1 = df['room']
    p11 = p1.tolist()
    p11 = Counter(p11)
    
    ## Homogenous transformation for rooms` name
    for col in df:
        if col=='part_id':
            if not os.path.isfile('data/beacons_dataset_filter.csv'):
                n_test = sum(1 for _ in df[col].iteritems())
                for row_value in tqdm(df[col].iteritems()):    
                    (i, row_value) = row_value
                    if (str.isdigit((row_value))==False):                  
                           df=df.drop(i)
                df.to_csv('data/beacons_dataset_filter.csv', index=False)
            else:
                df = pd.read_csv('data/beacons_dataset_filter.csv', delimiter=',')
        
        if col =='room':
            unique_values_list=df['room'].unique().tolist()
            homogenous_list=['Bedroom','Bathroom','Livingroom','Kitchen', 'Office']
            
            for val in unique_values_list:
                result,ratio = process.extractOne(str(val), homogenous_list)
                if ratio>80 and val!='T':
                    df.replace(val,result,inplace=True)
                if 'Seat' in str(val) or 'Sit' in str(val):
                    df.replace(val,'Livingroom',inplace=True)
                if 'Din' in str(val) or 'din' in str(val):
                    df.replace(val,'Livingroom',inplace=True)
                if val=='Chambre':
                    df.replace(val,'Bedroom',inplace=True)
                if val=='Washroom':
                    df.replace(val,'Bathroom',inplace=True)
            p2 = df['room']
            p21 = p2.tolist()
            p21 = Counter(p21)
            
            
    
    ## Create new dataset based on time remaiming into room
    unique_users_list=df['part_id'].unique().tolist()
    iter=0
    new_df=pd.DataFrame(index = list(range(0,len(unique_users_list))), columns = ['part_id','Bedroom','Bathroom','Livingroom','Kitchen'])
    
    for user, df_user in df.groupby('part_id'):
        rows,cols=df_user.shape
        if rows>1:
            df_user.sort_values(by='ts_date', inplace=True)
            df_user.sort_values(by=['ts_date', 'ts_time'], ascending = [True,True],inplace = True)
            df_user=df_user.reset_index(drop=True)
            df_user['time_in_room']=np.nan  
            
            FMT = '%H:%M:%S'
            all_rooms_time=0
            kitchen_time=0
            bedroom_time=0
            bathroom_time=0
            livingroom_time=0
            
            for i in range (0,rows-1):
                if i!=rows-1:
                    if df_user['ts_date'].loc[i]==df_user['ts_date'].loc[i+1]:#idia mera
                        s2=str(df_user['ts_time'].loc[i+1])
                        s1=str(df_user['ts_time'].loc[i])            
                        time_=datetime.strptime(s2, FMT) - datetime.strptime(s1, FMT)         
                        time_=time_.total_seconds()
                        if time_>10*3600:
                            calc=True
                        else:
                            calc=True        
                    if df_user['ts_date'].loc[i]!=df_user['ts_date'].loc[i+1]:#alli mera
                        if int(df_user['ts_date'].loc[i+1])-int(df_user['ts_date'].loc[i])==1:#mexri tin epomeni
                            s2=str(df_user['ts_time'].loc[i+1])
                            s1=str(df_user['ts_time'].loc[i])            
                            time_=abs(datetime.strptime(s2, FMT) - datetime.strptime(s1, FMT) )        
                            time_=24*3600-time_.total_seconds()
                            if time_>8*3600:
                                 if df_user['room'].loc[i]!='Bedroom':
                                     calc=False
                                 else:
                                     calc=True
                            else:
                                calc=True                
                        else: #mexri to telos tis imeras
                            s2='23:59:59'
                            s1=str(df_user['ts_time' ].loc[i])            
                            time_=abs(datetime.strptime(s2, FMT) - datetime.strptime(s1, FMT) ) 
                            time_=time_.total_seconds()
                            if time_>5*3600:
                                calc=False
                            else:
                                calc=True
                else: # gia tin teleutaia kataxwrisi
                    s2='23:59:59'
                    s1=str(df_user['ts_time'].loc[i])            
                    time_=abs(datetime.strptime(s2, FMT) - datetime.strptime(s1, FMT) ) 
                    time_=time_.total_seconds()
                    if time_>5*3600:
                        calc=False
                    else:
                        calc=True   
                        
                if calc==True:               
                    df_user['time_in_room'].loc[i]=time_
                    all_rooms_time=all_rooms_time+time_
                    if df_user['room'].loc[i]=='Bedroom':
                        bedroom_time=bedroom_time+time_
                    if df_user['room'].loc[i]=='Bathroom':
                        bathroom_time=bathroom_time+time_
                    if df_user['room'].loc[i]=='Livingroom':
                        livingroom_time=livingroom_time+time_
                    if df_user['room'].loc[i]=='Kitchen':
                        kitchen_time=kitchen_time+time_
                        
            if all_rooms_time!=0:
                new_df['part_id'].loc[iter]=user
                new_df['Kitchen'].loc[iter]=round((kitchen_time*100)/all_rooms_time,1)
                new_df['Livingroom'].loc[iter]=round((livingroom_time*100)/all_rooms_time,1)
                new_df['Bedroom'].loc[iter]=round((bedroom_time*100)/all_rooms_time,1)
                new_df['Bathroom'].loc[iter]=round((bathroom_time*100)/all_rooms_time,1)
                iter+=1
    
    ## Save new dataset
    new_df=new_df.dropna(axis=0)
    new_df.to_csv('data/beacons_per_user.csv',index=False)





#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------



def cluster_handler(df, dim_reduct=True, kclusters=3):    
    if dim_reduct==True:
        df=pca_dim_reduction(df)
    # Define KMeans clusters
    k=kclusters
    
    # Define clusters algos
    clusterers={'Kmeans_2':[KMeans(n_clusters=2, init='k-means++',n_jobs=-1, n_init=100, random_state=42)],
                'Kmeans_3':[KMeans(n_clusters=k, init='k-means++',n_jobs=-1, n_init=100, random_state=42)],
                'DBSCAN':[DBSCAN(eps=0.5, min_samples=1,n_jobs=-1)],
                'Mean_shift':[MeanShift(bandwidth=cluster.estimate_bandwidth(df, quantile=0.6, n_samples=None, random_state=42, n_jobs=-1))],
                }             
          
    for clusterer_name,fun in clusterers.items():  
        print('--------------------------------------')
        clusterer=fun[0]
        if clusterer_name!='DBSCAN':
            clusterer.fit(df)
            
        pred_labels = clusterer.fit_predict(df)
        print(clusterer_name,'silhouette score: ',round(metrics.silhouette_score(df, pred_labels ,metric='euclidean', random_state=42), 3))
        print(clusterer_name,'Davies Bouldin Score: ',round(metrics.davies_bouldin_score(df, pred_labels), 3))
       

def pca_dim_reduction(dff):
    pca = PCA(random_state = 42, n_components=0.95, svd_solver='full')
    return pca.fit_transform(dff)








#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------


if part_2_2:
    lis1 = ['A', 'B', 'C', 'D']
    for ins in lis1:
        # Read both datasets
        df1 = pd.read_csv('data/{}.csv'.format(ins))
        df2 = pd.read_csv('data/beacons_per_user.csv')
        
        # Merge datasets and remove specificfeatures
        new = pd.merge(df1, df2, on='part_id')
        labels_true = new['fried']
        new = new.drop(['fried','part_id'], axis=1)
            
        # Normalize data
        st_scaler = preprocessing.StandardScaler()
        st_scaler.fit(new)
        data1 = st_scaler.transform(new)
        data = normalize(new)
        
        # Run clustering experiments
        print('Processed dataset:  {}.csv'.format(ins))
        cluster_handler(data, dim_reduct=False, kclusters=3)
        print('\n\n')

#-----------------------------------




