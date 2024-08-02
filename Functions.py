import shutil
from datetime import datetime
from tqdm import tqdm
import numpy as np
import pickle as pk
import pandas as pd
import os
import PARAMETERS
import matplotlib.pyplot as plt


Numerical_Features = ['IN_BYTES','OUT_BYTES','IN_PKTS','OUT_PKTS','SRC_TO_DST_IAT_AVG','DST_TO_SRC_IAT_AVG','FLOW_DURATION_MILLISECONDS','DURATION_IN','DURATION_OUT','MIN_TTL','MAX_TTL','LONGEST_FLOW_PKT','SHORTEST_FLOW_PKT',
                      'MIN_IP_PKT_LEN','MAX_IP_PKT_LEN','SRC_TO_DST_SECOND_BYTES','DST_TO_SRC_SECOND_BYTES','RETRANSMITTED_IN_BYTES', 'RETRANSMITTED_IN_PKTS', 'RETRANSMITTED_OUT_BYTES', 'RETRANSMITTED_OUT_PKTS', 'SRC_TO_DST_AVG_THROUGHPUT', 'DST_TO_SRC_AVG_THROUGHPUT',
                      'NUM_PKTS_UP_TO_128_BYTES', 'NUM_PKTS_128_TO_256_BYTES', 'NUM_PKTS_256_TO_512_BYTES', 'NUM_PKTS_512_TO_1024_BYTES', 'NUM_PKTS_1024_TO_1514_BYTES', 'TCP_WIN_MAX_IN', 'TCP_WIN_MAX_OUT', ]
Categoriacl_Features = ['IPV4_SRC_ADDR','IPV4_DST_ADDR','L4_SRC_PORT','L4_DST_PORT','PROTOCOL','L7_PROTO','TCP_FLAGS','CLIENT_TCP_FLAGS','SERVER_TCP_FLAGS','ICMP_TYPE', 'ICMP_IPV4_TYPE', 'DNS_QUERY_ID', 'DNS_QUERY_TYPE','DNS_TTL_ANSWER', 'FTP_COMMAND_RET_CODE']

def create_folder_if_not_exists(folder_name):
    try:
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
            print(f"Folder '{folder_name}' created.")
        else:
            print(f"Folder '{folder_name}' already exists.")
    except Exception as e:
        print(f"An error occurred while creating the folder: {e}")

def zero_Features(df, filter:str):
    df_ = df.copy()
    print(f'before SRC_TO_DST_IAT: {df_.SRC_TO_DST_IAT_STDDEV.mean()}')
    print(f'before IN_BYTES: {df_.IN_BYTES.mean()}')
    if filter == 'Benign':
        df_.loc[df_['Label'] == 1, 'SRC_TO_DST_IAT_STDDEV'] = 0
        df_.loc[df_['Label'] == 1, 'DST_TO_SRC_IAT_STDDEV'] = 0
        df_.loc[df_['Label'] == 1, 'IN_BYTES'] = 0
        df_.loc[df_['Label'] == 1, 'OUT_BYTES'] = 0
        df_.loc[df_['Label'] == 1, 'IN_PKTS'] = 0
        df_.loc[df_['Label'] == 1, 'IN_PKTS'] = 0
    elif filter == 'Attack':
        df_.loc[df_['Label'] == 0, 'SRC_TO_DST_IAT_STDDEV'] = 0
        df_.loc[df_['Label'] == 0, 'DST_TO_SRC_IAT_STDDEV'] = 0
        df_.loc[df_['Label'] == 0, 'IN_BYTES'] = 0
        df_.loc[df_['Label'] == 0, 'OUT_BYTES'] = 0
        df_.loc[df_['Label'] == 0, 'IN_PKTS'] = 0
        df_.loc[df_['Label'] == 0, 'IN_PKTS'] = 0
    print(f'After SRC_TO_DST_IAT: {df_.SRC_TO_DST_IAT_STDDEV.mean()}')
    print(f'After IN_BYTES: {df_.IN_BYTES.mean()}')
    return df_

def Single_ATT_Filtering_2hours(df,att):
    sub_df = df[df.ATTACK == att]
    first_time = sub_df.index.min()
    last_time = first_time + pd.Timedelta(hours=2)

    att_df = df[df.index >= first_time]
    att_df = att_df[att_df.index <= last_time]
    att_only_df = att_df[att_df.ATTACK == att]
    benign_only_df = att_df[att_df.ATTACK == 'Benign']

    combined_df = pd.concat([att_only_df, benign_only_df])
    sorted_combined_df = combined_df.sort_index()
    return sorted_combined_df


def convert_to_datetime(data, col_name: str, new_col: str):
    data[new_col] = data[col_name]
    data[new_col] = data[new_col].apply(lambda x: datetime.utcfromtimestamp(x).strftime('%Y-%m-%d %H:%M:%S'))
    data[new_col] = pd.to_datetime(data[new_col])
    return data


def single_attack_filtering(data, att,Benign=True):
    if not Benign:
        att_list = data.ATTACK.unique().tolist()
        att_list.remove(att)
        filtered_data = data[~data['ATTACK'].isin(att_list)]
        att_df = filtered_data[filtered_data.ATTACK == att]
        sorted_df = att_df.sort_values(by='FLOW_START_MILLISECONDS')
    else:
        att_list = data.ATTACK.unique().tolist()
        att_list.remove(att)
        filtered_data = data[~data['ATTACK'].isin(att_list)]
        att_df = filtered_data[filtered_data.ATTACK == att]
        first_time = att_df['FIRST_SWITCHED'].min()
        last_time = att_df['LAST_SWITCHED'].max()

        new_data = data[data['FIRST_SWITCHED'] >= first_time]
        new_data = new_data[new_data['LAST_SWITCHED'] <= last_time]  # one sign change everything
        att_list.remove('Benign')
        filtered_data = new_data[~new_data['ATTACK'].isin(att_list)]
        sorted_df = filtered_data.sort_values(by='FLOW_START_MILLISECONDS')

    return sorted_df



def empty_folder(folder_path):
    # Iterate over all files in the folder
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        try:
            if os.path.isfile(file_path):
                # Remove the file
                os.unlink(file_path)
            # If it's a directory, remove it recursively
            elif os.path.isdir(file_path):
                os.rmdir(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")
    try:
        shutil.rmtree(folder_path)
        print(f"Deleted folder: {folder_path}")
    except Exception as e:
        print(f"Error deleting folder {folder_path}: {e}")


color_map = {
    'DDoS': '#1f77b4',
    'DDOS': '#1f77b4',
    'Benign': '#2ca02c',
    'DoS': '#d62728',
    'DOS': '#d62728',
    'Fuzzers': '#9467bd',
    'Exploits': '#8c564b',
    'Reconnaissance': '#e377c2',
    'Generic': '#7f7f7f',
    'Shellcode': '#bcbd22',
    'Worms': '#17becf',
    'Backdoor': '#aec7e8',
    'Analysis': '#ff7f0e',
    'XSS': '#008b8b',
    'Password': '#556b2f',
    'Theft':'#ff1493',
    'Ransomware':'#FFA500',
    'MITM':'#FF69B4',
    'Injection':'#00CED1',
    'Scanning':'#7FFF00',
    'BruteForce':'#8c564b',
    'Web-Attack':'#17becf',
    'Infiltration':'#9467bd',
    'BoT':'#ff7f0e'
}
