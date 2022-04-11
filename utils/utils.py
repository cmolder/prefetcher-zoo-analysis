import pandas as pd
import numpy as np
from scipy import stats

metrics = ['accuracy', 'coverage', 'ipc_improvement', 'mpki_reduction', 'dram_bw_reduction']
amean_metrics = ['accuracy', 'coverage', 'dram_bw_reduction']

def read_weights_file(path):
    weights = pd.read_csv(path, sep=' ', header=None)
    weights.columns = ['full_trace', 'weight']
    
    trace = []
    simpoint = []
    
    for tr in weights.full_trace:
        tokens = tr.split('_')
        
        if len(tokens) == 3: # Cloudsuite
            trace.append(tokens[0] + '_' + tokens[2])
            simpoint.append(tokens[1])
        if len(tokens) == 2: # SPEC '06
            trace.append(tokens[0])
            simpoint.append(tokens[1])
        if len(tokens) == 1: # Gap
            trace.append(tokens[0])
            simpoint.append('default')
    
    weights['trace'] = trace
    weights['simpoint'] = simpoint
    
    return weights

def read_data_file(path):
    df = pd.read_csv(path)
    df.prefetcher = df.prefetcher.replace({
        'scooby': 'pythia',
        'spp_dev2': 'spp',
        'bop': 'bo'
    }, regex=True)
    return df

def mean(values, metric, weights=None):
    if type(weights) is np.ndarray:
        assert np.isclose(np.sum(weights), 1), 'Weights should sum to 1'
    if metric in amean_metrics:
        return np.average(values, weights=weights)
    else:
        if metric == 'ipc_improvement':
            # Add 100 to prevent negative values (so that 100 = no prefetcher baseline)
            return stats.gmean(values + 100, weights=weights) - 100 
        if metric == 'mpki_reduction':
            # Take gmean of relative misses instead of MPKI reduction to prevent negative values
            return 100 - stats.gmean(100 - values, weights=weights) 
        
def rank_prefetchers(df, metric, count=None):
    """Return the <count> best prefetchers, in order of maximum <metric>.
    """
    pf_avgs = []
    for i, (pf, df_pf) in enumerate(df.groupby('prefetcher')):
        avg = mean(df_pf[metric], metric)
        pf_avgs.append((avg, pf))
        
    best = sorted(pf_avgs)[::-1]
    if count != None:
        best = best[:count]
    
    return [pf for _, pf in best]