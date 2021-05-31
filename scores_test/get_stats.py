import csv
from collections import defaultdict
import pandas as pd


def representsFloat(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def get_df(filename):
    csvfile = open(filename,'r',newline='')
    reader = csv.DictReader(csvfile,delimiter=';')

    d = defaultdict(lambda : [])
    for row in reader:
        for key in row.keys():
            if representsFloat(row[key]):
                d[key].append(float(row[key]))
            else:
                d[key].append(row[key])



    df =pd.DataFrame(d)
    return df



df = get_df('umx_vs_lowpass__smaller_grid.csv')
#df.drop(df['improvement'].argmax())

# remove possible nan
df = df.dropna()

# compute improvement
df['improvement']=df['sdr_song_lpf'] - df['sdr_song']

# remove outlier
#df.drop(df['improvement'].argmax())



df2 = get_df('scaled_mixture_vs_lowpass_order1.csv')
df = df.dropna()
df2['improvement']=df2['sdr_song_lpf'] - df2['sdr_song']

df3 = get_df('scaled_mixture_vs_lowpass_smaller_grid_2_big.csv')
df = df.dropna()
df3['improvement']=df3['sdr_song_lpf'] - df3['sdr_song']



for order in [1,2]:
    for fs in range(50,100,10):
        improvements = df.loc[((df['butter_order']==order) &(df['butter_cutoffFs'] == fs))]['improvement']
        print(order, fs,improvements.mean())




for order in [1]:
    for fs in range(100,551,50):
        improvements = df2.loc[((df2['butter_order'] == order) & (df2['butter_cutoffFs'] == fs))]['improvement']
        print(order, fs, improvements.mean())





for order in [2]:
    for fs in range(100,551,50):
        improvements = df3.loc[((df3['butter_order'] == order) & (df3['butter_cutoffFs'] == fs))]['improvement']
        print(order, fs, improvements.mean())