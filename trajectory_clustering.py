import pandas as pd
import os 
import traja
import matplotlib,pylab as plt
import seaborn as sns
import matplotlib.ticker as ticker
from datetime import datetime as dt
import numpy as np


def cycle_label(hour):
   cycles=[7,19]
   if 12<hour<cycles[1] or cycles[0]<=hour<=12:
       return 'Day'
   else:
       return 'Night'


def add_labels_df(df,tag):
    df['datetime']=df['Timestamp'].apply(dt.fromtimestamp)
    df['hour']=df['datetime'].apply(lambda x: x.hour)
    df['cycle']=df['hour'].apply(cycle_label)
    df['Tag']=tag
    return df


def get_trajec_list(folder,fields,trajec_thres):
    tag=int(os.path.basename(folder))
    dfs=[pd.read_csv(folder+'/'+f,usecols=fields)for f in os.listdir(folder)]
    dfs=[df for df in dfs if df.iloc[-1]['Timestamp']-df.iloc[0]['Timestamp']>trajec_thres]
    dfs=[add_labels_df(df,tag) for df in dfs]
    dfs_label_check=[len(df['cycle'].unique()) for df in dfs]
    dfs_label_check=[i for i,v in enumerate(dfs_label_check) if v>1]
    if len(dfs_label_check)>0:
        for inde in dfs_label_check:
            temp_df=dfs[inde].copy()
            temp_night=temp_df.loc[temp_df['cycle']=='Night']
            temp_day=temp_df.loc[temp_df['cycle']=='Day']
            dfs.pop(inde)
            dfs.append(temp_night)
            dfs.append(temp_day)
    return dfs

#path_dic='/media/tony/data/data/ann_marie/mx1/'
#path_dic='/media/tony/data/data/ann_marie/273107/'
#path_dic='/media/tony/data/data/ann_marie/273115/'
#path_dic='/media/tony/data/data/ann_marie/hong1/'
path_dic='/media/tony/data/data/ann_marie/fx2/'
paths=[path_dic+folder for folder in os.listdir(path_dic) if folder[-4:]!='.txt' and folder[-4:] !='ions' and folder[-4:]!='.csv' and folder[-4:]!='.ini']


#tags=[210000304,210000312,210000590] #mx1
#tags= [2016050811,2016050855,2016080244] #273107
#tags=[201609261,210608376,2016050990] #2115
#tags=[2016050945,210608261,210608359] #hong1
tags=[210000364,210608377,210000183]
trajec_thres=3
fields = ['Timestamp','x', 'y']
df_list_mx1=[]
for tag in tags:
    trajec_paths=[path+'/trajectories/'+str(tag) for path in paths]
    df_list=[get_trajec_list(path,fields,trajec_thres) for path in trajec_paths]
    df_list=[df for list_df in df_list for df in list_df]
    df_list_mx1.append(df_list)


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

bin_tuple=(2,2)
grids=[]
count=0
dic_label={}
for df_list in df_list_mx1:
    for df in df_list:
        cycle=df.iloc[0]['cycle']
        tag=df.iloc[0]['Tag']
        dic_label[count]=[tag,cycle]
        grid=df.traja.trip_grid(bins=bin_tuple, hist_only=True)[0]
        grids.append(grid.flatten())
        count+=1


gridsarr=np.array(grids)
X = StandardScaler().fit_transform(gridsarr)
pca = PCA()
X_r = pca.fit(X).transform(X)

#pca.explained_variance_ratio_



#c_dict={'Day':'red','Night':'blue'}
#c_dict={210000304:'b',210000312:'r',210000590:'g'}#mx1
#r ko, g: rescue, , b wild type
#c_dict={2016050811:'r',2016050855:'b',2016080244:'g'}#
#c_dict={2016050990:'r',210608376:'b',201609261:'g'}#211115
#c_dict={2016050945:'r',210608359:'b',210608261:'g'}
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
#_, ax = plt.subplots()

for idx, animal in enumerate(X_r):
    #ax.scatter(X_r[idx, 0], X_r[idx, 1], color=c_dict[dic_label[idx][0]], alpha=.8, lw=8, label=dic_label[idx][0])
    ax.scatter(X_r[idx, 0], X_r[idx, 1], X_r[idx,2], color=c_dict[dic_label[idx][0]], alpha=.8, lw=8, label=dic_label[idx][0])
#plt.legend(title='Cycle', loc='best', shadow=False, scatterpoints=1)









