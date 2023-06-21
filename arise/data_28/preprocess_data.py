'''
_ Preprocesses the csv file created using sc_warts2csv scamper tool 
_ Create csv files containing RTT values

INPUT: csv file created as a result of converting warts file to csv file
OUTPUT: csv files containing RTT values, each file for one source-destination (SD) pair

USAGE
Specify the target directory to store resulting csv files
Specify fname, min_num_rtt, max_num_rtt
  fname: the csv file resulting from using sc_warts2csv scamper tool (users need to provide their own)
  min_num_rtt: the minimum number of data points for each dataset (current value: 2000)
  max_num_rtt: the maximum number of data points for each dataset (current value: 4000)

'''

import pandas as pd
import os, sys, glob
sys.path.append('../')
from utils import get_full_path


# Import csv file
# Specify path to csv file
print("... Import csv file")
fname = get_full_path('atl2-20190101.csv') # this file is not provided, users need to supply their own

df = pd.read_csv(fname, header=None, sep=';')

# Set header to version, userID, timestamp, ...
print("... Set header to version, userID, timestamp, ...")
new_header = df.iloc[0]
df = df[1:]
df.columns = new_header

# Remove unnecessary lines
print("... Remove unnecessary lines")
df_new = df[df['version'] != 'version']

# Extract only hopaddrs, timestamp, and rtts
print("... Extract only hopaddrs, timestamp, and rtts")
df_hopaddr_timestamp_rtt = df_new[['timestamp','hopaddr', 'rtt']]
df_hopaddr_timestamp_rtt['timestamp_rtt'] = df[['timestamp', 'rtt']].apply(tuple, axis=1)

df_final = df_hopaddr_timestamp_rtt.set_index('hopaddr')

# Group by hopaddrs with rtts in list
print("... Group by hopaddrs with rtts in list")
df_hrt = df_final['timestamp_rtt']
df_hrt_grouped = df_hrt.groupby(df_hrt.index).apply(list)
df_hrt_final = df_hrt_grouped.reset_index()

# Extract only the hopaddrs with the number of rtt more than desired value
print("... Extract only the hopaddrs with the number of rtt more than desired value")
min_num_rtt = 2000
max_num_rtt = 4000
df_hrt_final2 = df_hrt_final[df_hrt_final.timestamp_rtt.apply(lambda x: len(x) > min_num_rtt and len(x) < max_num_rtt)]

# Create csv files containing the RTTs, one csv file for each client, sort by datetime
print("... Create csv files containing the RTTs, one csv file for each client")

num_results = df_hrt_final2.shape[0] 
for i in range(num_results):
    hop = df_hrt_final2['hopaddr'].iloc[i]

    num_tuples = len(df_hrt_final2['timestamp_rtt'].iloc[i])    
    rtts = []
    timestamps = []
    for j in range(num_tuples):
        timestamp = df_hrt_final2['timestamp_rtt'].iloc[i][j][0]
        rtt = df_hrt_final2['timestamp_rtt'].iloc[i][j][1]
        rtts.append(rtt)
        timestamps.append(timestamp)
    timestamp_rtt = {'timestamp': timestamps, 'rtt': rtts}
    # create new dataframe from these lists
    df_file = pd.DataFrame.from_dict(timestamp_rtt)
    # convert epoch to datetime
    df_file['datetime'] = pd.to_datetime(df_file['timestamp'],unit='s')
    # sort by datetime
    df_file_sorted = df_file.sort_values(by='datetime')
    # only take columns datetime and rtt
    df_file_dt_rtt = df_file_sorted[['datetime', 'rtt']]
    # reset index
    df_file_final = df_file_dt_rtt.reset_index(drop=True)
    
    # save to csv    
    # print(str(hop) + '.csv')
    df_file_final.to_csv(str(hop) + '.csv')


# rename filenames 
files = glob.glob('*.csv')
for i in range(len(files)):
    os.rename(files[i], 'dataset_'+str(i)+'.csv')
