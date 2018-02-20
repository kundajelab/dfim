### Null Model for DFIM scores
### Peyton Greenside
### Kundaje Lab, Stanford University

import os, sys
import pandas as pd
import numpy as np
import operator

import deeplift
from deeplift.conversion import keras_conversion as kc

BASES = ['A','C','G','T']

def shuffle_seq_from_fasta(seq_fastas, dinuc=False):

    ### DINUCLEOTIDE
    if dinuc:
        print('Using dinucleotide shuffle')
        shuf_seq_fastas = [deeplift.dinuc_shuffle.dinuc_shuffle(s) for s in seq_fastas]

    ### RANDOM SHUFFLE
    else:
        print('Using random shuffle')
        random.seed(1)
        shuf_seq_fastas = [''.join(random.sample(s,len(s))) for s in seq_fastas]

    shuf_sequences = np.swapaxes(util.setOfSeqsTo2Dimages(shuf_seq_fastas).squeeze(1), 1, 2)
    shuf_sequences = shuf_sequences.astype('float32')

    return shuf_sequences

def shuffle_seq_from_one_hot(sequences, dinuc=False):

    ### DINUCLEOTIDE
    if dinuc:
        raise ValueError('Not implemented')
        # Convert to fasta to use DL function?

    ### RANDOM SHUFFLE
    else:
        print('Using random shuffle')
        np.random.seed(1)
        if len(sequences.shape) == 3:
            # Assume (Batch, SEQ_LEN, 4)
            SHUF_AXIS = 1
            shuf_sequences = sequences[:, np.random.permutation(sequences.shape[SHUF_AXIS]), :]
        elif len(sequences.shape) == 4:
            # Assume (Batch, 1, 4, SEQ_LEN)
            SHUF_AXIS = 3
            shuf_sequences = sequences[:, :, :, np.random.permutation(sequences.shape[SHUF_AXIS])]
        else:
            raise ValueError(
                    '''Not sure how to deal with sequences 
                    of shape {0}'''.format(sequences.shape))

    return shuf_sequences


def flatten_nested_dict(score_dict):
    """
    Arguments:
        score_dict: dictionary with keys as tasks 
                    and leaves data_frames

    Returns:
        score_leaves: list of numpy arrays with scores
    """
    def find_leaf(score_dict):
        for key, val in score_dict.iteritems():
            key_history.append(key)
            if isinstance(val, dict):
                find_leaf(val)
            else:
                score_leaves.append(val)

    score_leaves = []
    key_history = []
    find_leaf(score_dict)

    return score_leaves


# Get from dictionary from list
def get_from_dict(data_dict, map_list):
    return reduce(operator.getitem, map_list, data_dict)

# Set value in dictionary from list
def set_in_dict(data_dict, map_list, value):
    get_from_dict(data_dict, map_list[:-1])[map_list[-1]] = value


def restore_nested_dict(score_dict, scores_list): 
    """
    Arguments:
        score_dict: dictionary with keys as tasks 
                    (nested for Graph models)
                    and leaves of data frames as scores_list
        scores_list: single list of scores

    Returns:
        new_score_dict: dictionary in same shape as score_dict
                        with scores_list instead of original content
    """

    key_tree = {}
    new_score_dict = {}
    scores_start = 0

    def restore_leaf(score_dict, descend=0, key_tree=None, scores_start=scores_start):
        for key, val in score_dict.iteritems():
            if isinstance(val, dict):
                if descend not in key_tree:
                    key_tree[descend] = [key]
                else:
                    key_tree[descend].append(key)
                current_key_tree = [key_tree[d][-1] for d in range(descend)
                                   ] + [key] if descend > 0 else [key]
                set_in_dict(new_score_dict, current_key_tree, {})
                restore_leaf(val, descend = descend+1, key_tree = key_tree)
            else:
                final_key_tree = [key_tree[d][-1] for d in range(descend)] + [key]
                set_in_dict(new_score_dict, final_key_tree, scores_list[scores_start])
                scores_start += 1

    restore_leaf(score_dict, key_tree=key_tree)

    return new_score_dict

def assign_empirical_pval(real_values, null_values, remove_zeros=True):
    '''
    remove zeros gets rid of diagonals
    '''
    def empirical_pvalue(val, null_values=null_values):
        pval = float(1+sum(float(val) <= np.array(null_values)))/(len(null_values)+1)
        return pval

    pval_df = real_values.applymap(empirical_pvalue)
    np.fill_diagonal(pval_df.values, 1)

    return pval_df

def assign_fit_pval(real_values, null_values, remove_zeros=True):
    '''
    remove zeros gets rid of diagonals
    '''
    import scipy

    WARNING_NULL_SIZE = 100

    if len(null_values) < WARNING_NULL_SIZE:
        print("""WARNING, fitting null distribution 
                 with less than {0} values""".format(WARNING_NULL_SIZE))

    mu, std = scipy.stats.norm.fit(null_values)

    def gaussian_pvalue(val, null_values=null_values):
        # z_score = (mu - val) / std
        pval = 1 - scipy.stats.norm.cdf(val, mu, std)
        return pval

    pval_df = real_values.applymap(gaussian_pvalue)
    np.fill_diagonal(pval_df.values, 1)

    return pval_df

def assign_pval(dfim_dict, null_dict, null_level='per_map',
                null_type='fit', diagonal_value = 0):

    '''
    Arguments:
        null_level - 
            'per_map' - for long sequences when pvalues are assigned per map
            'per_task' - extract scores and build a null for each task
            'global' - extract all scores in entire dict
        diagonal_value 
            value that DFIM diagonal contains that is meaningless and should be removed

    Returns:
        dfim_pval_dict - dictionary same shape as dfim_dict except with p-values

    '''

    pval_func = assign_empirical_pval if null_type == 'empirical' else assign_fit_pval

    NULL_LEVEL_OPTIONS = ['per_map', 'per_task', 'global']
    NULL_TYPE_OPTIONS = ['empirical', 'fit']

    assert null_level in NULL_LEVEL_OPTIONS
    assert null_type in NULL_TYPE_OPTIONS

    dfim_pval_dict = {}

    if null_level == 'per_map':

        for task in dfim_dict.keys():

            dfim_pval_dict[task] = {}

            for seq in dfim_dict.keys():

                flat_real_values = dfim_dict[task][seq].values().flatten()
                flat_null_values = null_dict[task][seq].values().flatten()

                flat_pvalues = pval_func(flat_real_values, flat_null_values)

                flat_pvalues.reshape(dfim_dict[task][seq].shape)

                dfim_pval_dict[task][seq] = flat_pvalues

    elif null_level == 'per_task':

        for task in dfim_dict.keys():

            dfim_pval_dict[task] = {}

            list_real_values = flatten_nested_dict(dfim_dict[task])
            list_null_values = flatten_nested_dict(null_dict[task])

            flat_null_values = [el for df in list_null_values 
                                   for el in df.values.flatten() if el != diagonal_value]

            list_pvalues = [pval_func(df, flat_null_values) for df in list_real_values]

            pvalue_dict = restore_nested_dict(dfim_dict[task], list_pvalues)

            dfim_pval_dict[task] = pvalue_dict

    elif null_level == 'global':

        list_real_values = flatten_nested_dict(dfim_dict)
        list_null_values = flatten_nested_dict(null_dict)

        flat_null_values = [el for df in list_null_values 
                               for el in df.values.flatten() if el != diagonal_value]

        list_pvalues = [pval_func(df, flat_null_values) for df in list_real_values]

        pvalue_dict = restore_nested_dict(dfim_dict, list_pvalues)

        dfim_pval_dict = pvalue_dict

    else:
        raise ValueError('Please provide null level in {0}'.format(NULL_LEVEL_OPTIONS))

    return dfim_pval_dict


