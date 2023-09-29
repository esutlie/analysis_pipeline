from get_session_list import get_session_list
import os
from flatten_list import flatten_list
from load_data import load_data
from get_data_path import get_data_path
from collections import Counter
import numpy as np

def get_all_regions():
    all_regions = []
    files = get_session_list()
    for session in files:
        spikes, pi_events, cluster_info = load_data(session)
        if 'area' not in cluster_info.keys():
            continue
        all_regions.append(cluster_info.area.values)
    all_regions = flatten_list(all_regions)

    regions = list(Counter(all_regions).keys())
    region_counts = list(Counter(all_regions).values())
    return regions, region_counts

if __name__ == '__main__':
    regions, region_counts = get_all_regions()
    print(f'{len(regions)} regions')
    for i,j in zip(regions, region_counts):
        print(f'{i}: {j/sum(region_counts)*100:.2f}%')
