#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 17:04:49 2022

@author: debbie
"""

### Beyond the above, now look at all data together and extract special 
### flows/routing when two actions inside one transaction are two linked swaps 
### that could have been one (bridge). This could give you an insight of 
### important pools for knowledgeable traders/money.

import glob
import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
import networkx as nx

with open('subsets/A.pkl', 'rb') as f:
    pools_dict = pickle.load(f)
name_folder = 'A'

# start the code

full = pd.concat(list(pools_dict.values()), 
                 ignore_index=True).sort_values(by=['timestamp'])

name0, name1, fees = [], [], []
for pair in list(full.pair_name):
    if pair == 'ETH2x-FLI-WETH/3000':
        t0 = 'ETH2x-FLI'
        t1 = 'WETH'
        fee = '3000'
    else:
        two = pair.split('-')
        t0 = two[0]
        t1 = two[1]
        t1, fee = t1.split('/')
    name0.append(t0)
    name1.append(t1)
    fees.append(fee)
    
full['name0'] = name0
full['name1'] = name1
full['feeTier'] = fees

actions_for_txn = full.groupby('transaction_id').count().sort_values(by=['id'], 
                              ascending=False)
# save only the txns that did at least 2 actions. Only 1 cannot tell us 
#anything about flows
txns_of_interest = list(actions_for_txn[actions_for_txn.id>=2].index)

# difficult to follow the exact money.
# Approximation: just count buy/sell and match that if for an action the 
#opposite sign is still "available"

# also, take the sign because difficult to compare the amount of 
#different cryptos

def flows_in_txn(temp):
    
    tokens = list(set(temp.name0) | set(temp.name1))
    balance = dict(zip(tokens, np.zeros(len(tokens))))
    balance =  dict(zip(tokens, [[]]*len(tokens)))
    save_mid = []
    pools_considered =  dict(zip(tokens, [[]]*len(tokens)))

    look = dict(zip(tokens, [[]]*len(tokens)))

    for r in range(len(temp)):
        row = temp.iloc[r, :]
        n0 = row.name0
        n1 = row.name1

        a0 = np.sign(float(row.amount0))
        a1 = np.sign(float(row.amount1))

        pair = row.pair_name

        save = pools_considered[n0].copy()
        save += [pair]
        pools_considered[n0] = save

        save = pools_considered[n1].copy()
        save += [pair]
        pools_considered[n1] = save

        save = balance[n0].copy() 
        save += [a0]
        balance[n0] = save

        save = balance[n1].copy() 
        save += [a1]
        balance[n1] = save
    
    
        """
            I need to find when the thing that now goes to zero became 
            first non-zero.
            Use this bridges are the things that tell me where the money goes!!!
        """

        try:
            if balance[n0][-2] == -1 and  balance[n0][-1] == +1:
                save_mid.append((n0, pair))

                save = look[n0].copy() 
                save += [len(balance[n0])-1]
                look[n0] = save

            if balance[n1][-2] == -1 and  balance[n1][-1] == +1:
                save_mid.append((n1, pair))

                save = look[n1].copy() 
                save += [len(balance[n1])-1]
                look[n1] = save
        except:
            None
        
    #balance[n0] += a0
    #balance[n1] += a1
    
#print(balance, save_mid)

    flows = []
    if len(save_mid) > 0:
        for c in range(len(save_mid)):
            curr = save_mid[c]
            idx = look[curr[0]][0]
            flow = [pools_considered[curr[0]][idx-1], 
                    pools_considered[curr[0]][idx]]

            flows.append(flow)
            
    return flows

# difficult to follow the exact money.
# Approximation: just count buy/sell and match that if for an 
#action the opposite sign is still "available"

# also, take the sign because difficult to compare the 
#amount of different cryptos

def flows_(txns):
    f = []
    for t in txns:
        
        temp = full[full.transaction_id == t]
        temp.loc[:, 'logIndex'] = [float(l) for l in list(temp.loc[:, 
                'logIndex'])]
        temp = temp.sort_values('logIndex')
        
        f += flows_in_txn(temp)
    return f


### We need to assume one jump maximum in this case.
        
        
def compute_all_flows(t):
    temp = full[full.transaction_id == t]
    temp.loc[:, 'logIndex'] = [float(l) for l in list(temp.loc[:, 'logIndex'])]
    temp = temp.sort_values('logIndex')
    
    flows = flows_in_txn(temp)
    return flows

print('here')

from multiprocessing import Pool
with Pool(36) as p:
    all_flows = p.map(compute_all_flows, tqdm(txns_of_interest))

with open('/scratch/Debbie_Miori/bridges/'+name_folder+'.pickle', 'wb') as handle:
    pickle.dump(all_flows, handle)





