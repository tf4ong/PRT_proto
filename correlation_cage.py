import pandas as pd
import matplotlib.pylab as plt
import datetime as dt
import numpy as np
import scipy.stats as stats

# simple function to do itc analysis 

df_en2=pd.read_csv('/media/tony/data/data/si_itc/2021-09-23_15-59-01_EN2/210000359.csv')
dic_en2=itc_episodes(df_en2,2010230053)


df_en1=pd.read_csv('/media/tony/data/data/si_itc/2021-09-23_15-15-57_EN1/210000451.csv')
dic_en1=itc_episodes(df_en1,210000209)

df_en3=pd.read_csv('/media/tony/data/data/si_itc/2021-09-23_16-39-01_EN3/2010230046.csv')
dic_en3=itc_episodes(df_en3,210000478)





a=[v['Duration'] for v in [dic_en2,dic_en1,dic_en3]]
b=['EN2','EN1','EN3' ]
c={'Type': b,'Duration':a}
df_d=pd.DataFrame(c)

a=[v['n_itc'] for v in [dic_en2,dic_en1,dic_en3]]
b=['EN2','EN1','EN3' ]
c={'Type': b,'Episodes':a}
df_epi=pd.DataFrame(c)

a=dic_en2['episodes']+dic_en1['episodes']+dic_en3['episodes']
b=['EN2' for CM in dic_en2['episodes']]+['EN1' for CM in dic_en1['episodes']]+['SI' for CM in dic_en3['episodes']]
c={'Type': b,'Episodes':a}
df_itc_epi=pd.DataFrame(c)





def find_nonzero_runs(a):
    # Create an array that is 1 where a is nonzero, and pad each end with an extra 0.
    isnonzero = np.concatenate(([0], (np.asarray(a) != 0).view(np.int8), [0]))
    absdiff = np.abs(np.diff(isnonzero))
    # Runs start and end where absdiff is 1.
    ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
    return ranges

def itc_episodes(df,tag_int):
    total_dur=df[str(tag_int)].sum()
    itc_epi=find_nonzero_runs(df[str(tag_int)])
    n_epi=len(itc_epi)
    n_epi_dur=[df[it[0]:it[1]][str(tag_int)].sum() for it in itc_epi]
    n_epi_dur=[dur for dur in n_epi_dur if dur >0]
    dic={'Duration':total_dur,'n_itc':n_epi,'episodes':n_epi_dur}
    return dic


def get_itcs_details(tags,path):
    itc_dic_final={}
    df_list =[pd.read_csv(path+f"/{str(tag)}.csv") for tag in tags]
    count=0
    for tag in tags:
        df=df_list[count]
        df_itcs={t:itc_episodes(df,t) for t in tags if t !=tag}
        itc_dic_final[tag]=df_itcs
        count+=1
    return itc_dic_final


path='/media/tony/data/data/of_cage_no/cage11non'
with open(path+'/'+'logs.txt','r') as f:
    tags=f.readlines()
    tags=[int(i) for i in tags[1][6:].split(',')]

a=get_itcs_details(tags,path)

path='/media/tony/data/data/of_cage_no/one_one_cm'
with open(path+'/'+'logs.txt','r') as f:
    tags=f.readlines()
    tags=[int(i) for i in tags[1][6:].split(',')]

b=get_itcs_details(tags,path)

dic_c=itc_episodes(df_c,210000478)
dic_nc=itc_episodes(df_nc,210000210)
dic_si=itc_episodes(df_si,210000451)



a=[v['Duration'] for v in [dic_c,dic_nc,dic_si]]
b=['CM','Non CM','SI' ]
c={'Type': b,'Duration':a}
df_d=pd.DataFrame(c)

a=[v['n_itc'] for v in [dic_c,dic_nc,dic_si]]
b=['CM','Non CM','SI' ]
c={'Type': b,'Episodes':a}
df_epi=pd.DataFrame(c)

a=dic_c['episodes']+dic_nc['episodes']+dic_si['episodes']
b=['CM' for CM in dic_c['episodes']]+['Non CM' for CM in dic_nc['episodes']]+['SI' for CM in dic_si['episodes']]
c={'Type': b,'Episodes':a}
df_itc_epi=pd.DataFrame(c)



    
import matplotlib.ticker as ticker
from matplotlib import rc, rcParams
fig, axes = plt.subplots(1,3,figsize=(20,5))
sns.set_style("whitegrid")

sns.barplot(y='Duration', x= "Type", data=df_d, ax=axes[0])
axes[0].set_ylim(0,500)
axes[0].yaxis.set_major_locator(ticker.MultipleLocator(100))
axes[0].tick_params(axis="y", labelsize=15)
axes[0].tick_params(axis="x", labelsize=15)
axes[0].set_xlabel('')
axes[0].set_ylabel('Duration (s)',size=15,fontweight='bold',labelpad=10)
axes[0].set_title('Total Duration',fontweight='bold',fontsize=20,pad=15)



sns.barplot(y='Episodes', x= "Type", data=df_epi, ax=axes[1])
axes[1].set_ylim(0,400)
axes[1].yaxis.set_major_locator(ticker.MultipleLocator(100))
axes[1].tick_params(axis="y", labelsize=15)
axes[1].tick_params(axis="x", labelsize=15)
axes[1].set_ylabel('Count',size=15,fontweight='bold',labelpad=10)
axes[1].set_title('Number of Interaction',fontweight='bold',fontsize=20,pad=15)
axes[1].set_xlabel('')


sns.barplot(y='Episodes', x= "Type", data=df_itc_epi, ax=axes[2],capsize=0.2)
axes[2].set_ylim(0,1.8)
axes[2].yaxis.set_major_locator(ticker.MultipleLocator(0.6))
axes[2].tick_params(axis="y", labelsize=15)
axes[2].tick_params(axis="x", labelsize=15)
axes[2].set_ylabel('Duration(s)',size=15,fontweight='bold',labelpad=10)
axes[2].set_title('Duration Per Interaction',fontweight='bold',fontsize=20,pad=15)
axes[2].set_xlabel('')