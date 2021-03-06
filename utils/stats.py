import numpy as np
import pandas as pd
from utils import utils
from IPython.display import display # DEBUG

def get_longest_simpoints(weights):
    idx = (weights.groupby('trace')['weight'].transform(max) == weights['weight'])
    traces = weights[idx].trace
    return traces

def _process_prefetcher(stats, df, weights, tr, pf):
    wt = weights[weights.trace == tr][['simpoint', 'weight']]
    data = df[(df.trace == tr) & (df.all_pref == pf)]
    data = data.merge(wt, on='simpoint')
    weights = data['weight'] / sum(data['weight'])

    stats['trace'] = np.append(stats['trace'], tr)
    stats['all_pref'].append(pf)
    stats['simpoint'] = np.append(stats['simpoint'], 'weighted')
    
    if len(data) == 0:
        print(f'[DEBUG] {pf} {tr} not found')
        for metric in utils.metrics:
            stats[f'{metric}'] = np.append(stats[f'{metric}'], np.nan)
        return
    
    for metric in utils.metrics:
        target = data[metric].item() if len(data) <= 1 else utils.mean(data[metric], metric, weights=weights)
        stats[f'{metric}'] = np.append(stats[f'{metric}'], target)
        #print('[DEBUG]', pf, metric, data[metric].to_list(), weights.to_list(), stats[f'{metric}'][-1])
        
def _process_phase_combined(stats, df, weights, tr):
    wt = weights[(weights.trace == tr)]
    data = df[(df.trace == tr)]
    
    # Filter out opportunity prefetchers
    data = data[~(data.LLC_pref.str.contains('pc_') | data.L2C_pref.str.contains('pc_') | data.L1D_pref.str.contains('pc_'))]
    
    stats['trace'] = np.append(stats['trace'], tr)
    stats['all_pref'].append(('no', 'no', 'offline_phase'))
    stats['simpoint'] = np.append(stats['simpoint'], 'weighted')            
    for metric in utils.metrics:
        best_metrics = data.groupby('simpoint')[metric].max().to_frame()
        best_metrics = best_metrics.merge(wt, on='simpoint')
        best_metrics['weight'] = best_metrics['weight'] / best_metrics['weight'].sum()

        try:
            target = utils.mean(best_metrics[metric], metric, weights=best_metrics['weight'])
        except:
            #print(f'[DEBUG] Could not compute phase_combined {tr} {metric}')
            target = np.nan
            
        stats[f'{metric}'] = np.append(stats[f'{metric}'], target)
    
def get_weighted_statistics(df, weights, add_phase_combined=True):
    stats = {
        'trace': np.array([]),
        'all_pref': [],
        'simpoint': np.array([]),
        'LLC_accuracy': np.array([]),
        'LLC_coverage': np.array([]),
        'ipc_improvement': np.array([]),
        'LLC_mpki_reduction': np.array([]),
        'dram_bw_reduction': np.array([])
    }
    
    for tr in df.trace.unique():
        for pf in df.all_pref.unique():
            _process_prefetcher(stats, df, weights, tr, pf)
                
        if add_phase_combined:
            _process_phase_combined(stats, df, weights, tr)
               
    return pd.DataFrame(stats)