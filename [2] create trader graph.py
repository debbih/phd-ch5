#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 07:29:55 2022

@author: debbie
"""

import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import date, datetime
import matplotlib.pyplot as plt
import networkx as nx
import json

"""
    Function to build one graph per origin
"""

from multiprocessing import Pool

def save_json(lt):
    
    lt_txns = subset[subset.origin==lt]
    
    nodes = range(len(lt_txns))
    
    
    # convert string of same pool to integer value
    l = list(lt_txns.pool)
    features = [map_[x] for x in l]
    
    features = dict(zip(nodes, features))

    # column 1 --> timestamp
    # total_seconds() --> this is a float
    edges = [
        (n1, n2, abs((subset.iloc[n2, 1] - subset.iloc[n1, 1]).total_seconds())) \
            for n1 in nodes for n2 in nodes[n1+1:]
            ]
        
    json_ = {"edges": edges, "features": features}
    
    #with open("Desktop/graph2vec/graph2vec-modif/1-create-graphs/datasetJanJune/"+lt+".json", "w") as outfile:
    #    json.dump(json_, outfile)
        
    with open('/scratch/Debbie_Miori/final_pools/graphs'+name+'/'+lt+'.json', 'w') as outfile:
        json.dump(json_, outfile)
    
    return None

##################################################################
##################################################################

mins_ = {'A': 60, 'B1': 30, 'B2': 30, 'C1': 20, 'C2': 20, 'C3': 20}
maxs_ = {'A': 15_000, 'B1': 5_000, 'B2': 11_000, 
         'C1': 4_000, 'C2': 5_000, 'C3': 9_000}

##################################################################
##################################################################

#name = 'A'

#for name in list(mins_.keys()):
for name in ['A']:

    min_ = mins_[name]
    max_ = maxs_[name]
    create_graphs = True
    plot = False
    
    #with open('Desktop/graph2vec/graph2vec-modif/1-create-graphs/final_pools/final_pools'+name+'.pkl', 'rb') as handle:
    #    pools = pickle.load(handle)
        
    with open('/scratch/Debbie_Miori/crypto/final_pools_SWAP_time/final_pools'+name+'.pkl', 'rb') as handle:
        pools = pickle.load(handle)
        
    map_ = dict([(y,x+1) for x,y in enumerate(sorted(set(pools)))])
        
    print(len(pools.keys()))
    
    for k in pools.keys():
        p = pools[k]
        p['pool'] = [k]*len(p)
        pools[k] = p.loc[:, ['id', 'timestamp', 'origin', 'pool',
                             'amountUSD', 'transaction_blockNumber'
                             ]]
        
    full = pd.concat(list(pools.values()), 
                     ignore_index=False).sort_values(by=['transaction_blockNumber'])
    full = full.astype(
        {'amountUSD': float, 'transaction_blockNumber': int}
        )
    
    subset = full
    #subset = full[full.timestamp >= datetime(2022, 1, 1, 0, 0, 0)]
    #subset = subset[subset.timestamp < datetime(2022, 7, 1, 0, 0, 0)]

    count_txn = subset.groupby('origin').count()
    
    if plot:
        count_txn10 = count_txn[count_txn.id>=10]
        count_txn25 = count_txn[count_txn.id>=25]
        count_txn50_4000 = count_txn[count_txn.id>=min_]
        count_txn50_4000 = count_txn50_4000[max_>=count_txn50_4000.id]
        
        print(len(count_txn10))
        print(len(count_txn25))
        print(len(count_txn50_4000))
        
        plt.figure(figsize=(10, 4))
        plt.yscale('log')
        plt.hist(count_txn.id, bins=40, label='all')
        #plt.hist(count_txn10.id, alpha=0.6, bins=30, label='min 10')
        plt.hist(count_txn25.id, alpha=0.8, bins=40, label='min 25')
        plt.hist(count_txn50_4000.id, alpha=0.4, bins=40, label=str(min_)+'-'+str(max_))
        plt.legend()
        plt.grid()
        plt.title('Case '+name)
        plt.xlabel('Total no. txns over the period')
        plt.ylabel('Count of LTs')
        plt.show()
    
    chosen = count_txn[count_txn.id>=min_]
    chosen = chosen[max_>=chosen.id]
    
    print(len(chosen))

    # Now really create the graphs
    
    import glob
    list_graphs = glob.glob('/scratch/Debbie_Miori/final_pools/graphsA/*')
    c = list(chosen.index)
    c = ['/scratch/Debbie_Miori/final_pools/graphsA/'+lt+'.json' for lt in c]
    to_do = list(set(c)-set(list_graphs))
    
    to_do_ids = [i.split('graphsA/')[1].split('.json')[0] for i in to_do]
    print(len(to_do_ids))
    
    if create_graphs:
        with Pool(36) as p:
            #p.map(save_json, chosen.index)
            p.map(save_json, to_do_ids)
    
























