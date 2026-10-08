import copy
import os
import sys
from dotenv import find_dotenv
sys.path.append(os.path.dirname(find_dotenv()))
import itertools

# Set env variables for numpy proc limits, 
# since CARC has a limit in these on login nodes.
os.environ["OMP_NUM_THREADS"] = "1" 
os.environ["OPENBLAS_NUM_THREADS"] = "1" 
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1" 

import numpy as np
from maestrowf.datastructures.core import ParameterGenerator

# https://stackoverflow.com/a/40623158
def dict_product(dicts):
    """
    >>> list(dict_product(dict(number=[1,2], character='ab')))
    [{'character': 'a', 'number': 1},
     {'character': 'a', 'number': 2},
     {'character': 'b', 'number': 1},
     {'character': 'b', 'number': 2}]
    """
    return (dict(zip(dicts.keys(), x)) for x in itertools.product(*dicts.values()))

# https://stackoverflow.com/questions/5558418/list-of-dicts-to-from-dict-of-lists
def listofdict_to_dictoflist(lod):
    return {k: [dic[k] for dic in lod] for k in lod[0]}

def dictoflist_to_listofdict(dol):
    return [dict(zip(dol,t)) for t in zip(*dol.values())]

def get_custom_generator(env, **kwargs):
    """
    Create a custom populated ParameterGenerator.
    :params env: A StudyEnvironment object containing custom information.
    :params kwargs: A dictionary of keyword arguments this function uses.
    :returns: A ParameterGenerator populated with parameters.
    """
    p_gen = ParameterGenerator()
    
    param_dict = {
        'SPLIT_STRATEGY': ['minMc_freq', 'n_training_wf'],
        'N_TILES': list(range(2, 11)),
        'N_FREQ_OVERLAP_ONESIDE': [2, 8, 16],
    }
    param_label_mapping = {
        'SPLIT_STRATEGY': 'strategy_%%',
        'N_TILES': 'ntiles_%%',
        'N_FREQ_OVERLAP_ONESIDE': 'nfreq_overlap_%%'
    }
    
    assert set(param_dict.keys()) <= set(param_label_mapping.keys())
    
    cartesian_prod_param = listofdict_to_dictoflist(list(dict_product(param_dict)))
    
    for k, v in cartesian_prod_param.items():
        p_gen.add_parameter(k, v, param_label_mapping[k])

    return p_gen
