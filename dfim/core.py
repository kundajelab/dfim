### Deep Feature Interaction Maps (DFIM) core functionalities
### Peyton Greenside
### Kundaje Lab, Stanford University

### Imports

import pandas as pd
import numpy as np
import copy

import deeplift
from deeplift.conversion import keras_conversion as kc

BASES = ['A','C','G','T']

def get_orig_letter(one_hot_vec):
    assert(len(np.where(one_hot_vec!=0)[0]) == 1)
    if one_hot_vec[0] != 0:
        return 'A'
    elif one_hot_vec[1] != 0:
        return 'C'
    elif one_hot_vec[2] != 0:
        return 'G'
    elif one_hot_vec[3] != 0:
        return 'T'

def get_letter_index(letter):
    if letter == 'A':
        return 0
    elif letter == 'C':
        return 1
    elif letter == 'G': 
        return 2
    elif letter == 'T':
        return 3


def generate_mutants_and_key(sequences, mut_loc_dict, sequence_index=None, mutants=BASES):
    """
    mut_loc_dict = {name: {'seq': 'mut_start': int, 'mut_end' int, 'resp_start': [], resp_end: []}}
    sequence_index: used to preserve index used in mut_loc_dict while allowing analysis only on correct ones, subset, etc.
    """
    if sequence_index is None:
        sequence_index = range(sequences.shape[0])

    print('Generating mutated sequences')

    mutated_seq_list = []
    ind = 0
    total_index_size = len([name for name in mut_loc_dict.keys() 
                                 if mut_loc_dict[name]['seq'] in sequence_index]
                          ) * 3
    mutated_seq_key = pd.DataFrame(columns=['array_ind', 'label', 'seq', 'mut_key', 'mut_start'], 
                                   index=range(total_index_size))
    # for seq in range(numSeq):
    for seq in sequence_index:
        # If there are two embedded motifs
        seq_muts = [name for name in mut_loc_dict.keys() if mut_loc_dict[name]['seq'] == seq]
        if len(seq_muts) > 0:
            # Add original sequence to the list of sequences
            mutated_seq_list.append(sequences[seq])
            mutated_seq_key.iloc[ind,:]={'array_ind': ind, 'seq': seq, 'label': 'original', 
                                         'mut_key': 'original', 'mut_start': -1}
            ind += 1
            for m in seq_muts:
                mut_start = mut_loc_dict[m]['mut_start']
                mut_end = mut_loc_dict[m]['mut_end']
                # If the mutation size is greater than 1 then just set to 0s or GC (?)
                if mut_end - mut_start > 1:
                    mutated_seq = copy.deepcopy(sequences[seq])
                    mutated_seq[mut_start:mut_end, :] = 0 
                    mutated_seq_list.append(mutated_seq)
                    label = 'mut;' + m + ';' + 'seq_'+ str(seq) + '_' + 'loc{0}-{1}'.format(
                                                                str(mut_start), str(mut_end))
                    mutated_seq_key.iloc[ind,:] = {'array_ind':ind, 'seq': seq, 'label': label,
                                                   'mut_key': m, 'mut_start': mut_start}
                    ind += 1
                # If mutation is a single base
                elif (mut_end - mut_start) == 1:
                    orig_letter  = get_orig_letter(sequences[seq][:,mut_start:mut_end])
                    seq_mutants = [el for el in mutants if el != orig_letter]
                    for mut_letter in seq_mutants:
                        mutated_seq = copy.deepcopy(sequences[seq])
                        mut_index = get_letter_index(mut_letter)
                        mutated_seq[mut_start:mut_end, :] = 0
                        mutated_seq[mut_start:mut_end, mut_index] = 1
                        label = 'mut;{0};seq_{1}_{2}to{3}_at{4}'.format(
                                            m, str(seq), orig_letter, 
                                            mut_letter, str(mut_start))
                        mutated_seq_key.iloc[ind,:]={'array_ind':ind, 'seq': seq, 'label': label, 
                                                     'mut_key': m, 'mut_start': mut_start}
                        ind += 1
                else:
                    raise ValueError('mut_start - mut_end == %s'%( (mut_start - mut_end)))

    mutated_seq_key = mutated_seq_key.iloc[range(ind),:]
    mutated_seq_array = np.array(mutated_seq_list)

    return (mutated_seq_array, mutated_seq_key)

def compute_importance(model, sequences, tasks,
                       score_type='gradient_input',
                       find_scores_layer_idx=0,
                       target_layer_idx=-2):

    ### Compute deepLIft
    print('Calculating Importance Scores')

    importance_method = {
        "rescale_conv_revealcancel_fc": deeplift.blobs.NonlinearMxtsMode.DeepLIFT_GenomicsDefault,
        "rescale_all_layers": deeplift.blobs.NonlinearMxtsMode.Rescale,
        "revealcancel_all_layers": deeplift.blobs.NonlinearMxtsMode.RevealCancel,
        "gradient_input": deeplift.blobs.NonlinearMxtsMode.Gradient,
        "guided_backprop": deeplift.blobs.NonlinearMxtsMode.GuidedBackprop
    }

    importance_model = kc.convert_sequential_model(model,
                        nonlinear_mxts_mode=importance_method[score_type])

    importance_func = importance_model.get_target_contribs_func(
                                find_scores_layer_idx=find_scores_layer_idx,
                                target_layer_idx=target_layer_idx)

    importance_score_dict = {}
    for task in tasks:
        importance_score_dict[task] = np.array(importance_func(task_idx=0,
                                             input_data_list=[sequences],
                                             batch_size=10,
                                             progress_update=1000))

    return importance_score_dict


def compute_delta_profiles(score_dict, mutated_seq_key, 
                           mut_loc_dict, tasks, sequence_index, 
                           mutated_seq_preds=None,
                           capture_seqs_max_thresh=None):
    ### For each 
    delta_dict = {}
    for task in tasks:
        delta_dict[task] = {}
        for seq in sequence_index:
            delta_dict[task][seq] = {}
            # Get importance scores of original sequence
            orig_ind = mutated_seq_key[mutated_seq_key.seq == seq].loc[
                                       mutated_seq_key.label == 'original','array_ind'].tolist()[0]
            # Get all mutants of that sequence
            seq_mutants = [name for name in mut_loc_dict.keys() if mut_loc_dict[name]['seq'] == seq]
            # Iterate through mutants  
            for m in seq_mutants:
                delta_dict[task][seq][m] = {}
                resp_starts = mut_loc_dict[m]['resp_start']
                resp_ends = mut_loc_dict[m]['resp_end']
                resp_names = mut_loc_dict[m]['resp_names']
                mut_ind = mutated_seq_key[mutated_seq_key.seq == seq].loc[
                                          mutated_seq_key.mut_key == m,'array_ind'].tolist()[0]
                mut_start = mutated_seq_key[mutated_seq_key.seq == seq].loc[
                                          mutated_seq_key.mut_key == m,'mut_start'].tolist()[0]
                for r in range(len(resp_starts)):
                    response_mut_profile = score_dict[task][mut_ind][resp_starts[r]:resp_ends[r], :]
                    orig_profile = score_dict[task][orig_ind][resp_starts[r]:resp_ends[r], :]
                    ddl_profile = orig_profile - response_mut_profile
                    if mutated_seq_preds is not None:
                        pred_diff = (mutated_seq_preds[orig_ind] - mutated_seq_preds[mut_ind])[0]
                    else:
                        pred_diff = None
                    key = 'mut_{0};response_{1}_at_{2}'.format(m, resp_names[r], str(resp_starts[r]))
                    if capture_seqs_max_thresh is None:
                        delta_dict[task][seq][m][key] = {'orig_profile': orig_profile,
                                                         'mut_profile': response_mut_profile,
                                                         'ddl_profile': ddl_profile,
                                                         'prediction_diff': pred_diff,
                                                         'mut_start': mut_start,
                                                         'resp_start': resp_starts[r]
                                                        }                        
                    elif capture_seqs_max_thresh is not None and abs(ddl_profile).max() > capture_seqs_max_thresh:
                        print('Max score exceeded threshold, adding sequence')
                        delta_dict[task][seq][m][key] = {'orig_profile': orig_profile,
                                                         'mut_profile': response_mut_profile,
                                                         'ddl_profile': ddl_profile,
                                                         'prediction_diff': pred_diff,
                                                         'mut_start': mut_start,
                                                         'resp_start': resp_starts[r]
                                                        }
                    else:
                        continue
    return delta_dict



def compute_dfim(delta_dict, sequence_index, tasks,
                 operations=[np.sum, np.max], operation_axes=[1, 0],
                 absolute_value=True,
                 annotate=False, mutated_seq_key=None,
                 diagonal_value=-1):
    """
    operations - MOTIFS: first take sum over all letters, then take mean over positions (FOR MOTIFS)
               - INDIVIDUAL BASES: max over base position
    absolute_value - will take absolute value of delta_profile before operations
    annotate - False will just use position as the index/column name
               True will capture the name

    """
    assert len(operations) == len(operation_axes)

    dfim_dict = {}
    # For each sequence and task
    for task in tasks:
        dfim_dict[task] = {}
        for seq in sequence_index:
            # Capture all the mutations and responses
            all_mut_pos = np.unique([delta_dict[task][seq][m][r]['mut_start'
                                         ] for m in delta_dict[task][seq].keys()
                                           for r in delta_dict[task][seq][m].keys()])
            all_resp_pos = np.unique([delta_dict[task][seq][m][r]['resp_start'
                                         ] for m in delta_dict[task][seq].keys()
                                           for r in delta_dict[task][seq][m].keys()])            
            # Allocate data frame with all mutant names and response names
            dfim_df = pd.DataFrame(index=all_mut_pos,
                                   columns=all_resp_pos,
                                   data=diagonal_value)

            # Extract all the differential profiles
            for m in delta_dict[task][seq].keys():
                for r in delta_dict[task][seq][m].keys():

                    if absolute_value:
                        ddl_profile = abs(delta_dict[task][seq][m][r]['ddl_profile'])
                    else:
                        ddl_profile = delta_dict[task][seq][m][r]['ddl_profile']

                    i_score = ddl_profile

                    for o in range(len(operations)):

                        # Perform operations
                        i_score = np.apply_along_axis(operations[o],
                                                      operation_axes[o],
                                                      i_score)
                    # Combine into a map
                    dfim_df.loc[delta_dict[task][seq][m][r]['mut_start'],
                                delta_dict[task][seq][m][r]['resp_start']] = i_score.tolist() 

            if annotate:

                # Add annotations to the rows
                # Current depends on simdna structure, generalize

                assert mutated_seq_key is not None

                seq_df = mutated_seq_key[mutated_seq_key.seq==seq]
                mut_annots = [seq_df.loc[seq_df.mut_start == p, 'mut_key'
                                        ].tolist()[0].split('-')[1]
                                for p in dfim_df.index]
                resp_annots = [seq_df.loc[seq_df.mut_start == p, 'mut_key'
                                        ].tolist()[0].split('-')[1]
                                for p in dfim_df.columns]
                dfim_df.index = mut_annots
                dfim_df.columns = resp_annots

            # Return the map for each seq and task
            dfim_dict[task][seq] = dfim_df

    return dfim_dict











