### Deep Feature Interaction Maps (DFIM) core functionalities
### Peyton Greenside
### Kundaje Lab, Stanford University

### Imports

import pandas as pd
import numpy as np
import copy
import scipy
import cPickle as pickle

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


def generate_mutants_and_key(sequences, mut_loc_dict, sequence_index=None, 
                             mutants=BASES, per_base_map=False):
    """
    mut_loc_dict = {name: {'seq': 'mut_start': int, 'mut_end' int, 'resp_start': [], resp_end: []}}
    sequence_index: used to preserve index used in mut_loc_dict 
                    while allowing analysis only on correct ones, subset, etc.
    if per_base_map:
        mutant key needs 
    """
    if sequence_index is None:
        sequence_index = range(sequences.shape[0])

    print('Generating mutated sequences')

    mutated_seq_list = []
    ind = 0

    mut_sizes = [mut_loc_dict[name]['mut_end'] - mut_loc_dict[name]['mut_start']
                        for name in mut_loc_dict.keys() 
                        if mut_loc_dict[name]['seq'] in sequence_index]

    if per_base_map:

        if np.max(mut_sizes) > 1:
            print('''Detected mutations of max size {0}, '''
                  '''but iterating through each base '''
                  '''per argument per_base_map=True'''.format(
                     np.max(mut_sizes)))

        total_index_size = np.sum(mut_sizes) * len(mutants)
    else:
        total_index_size = len(mut_sizes) * len(mutants)

    if per_base_map:
        key_columns = ['array_ind', 'label', 'seq',  'mut_key', 'mut_start',
                      'orig_letter', 'mut_letter']
    else:
        key_columns = ['array_ind', 'label', 'seq', 'mut_key', 'mut_start']

    mutated_seq_key = pd.DataFrame(columns=key_columns, 
                                   index=range(total_index_size))

    # for seq in range(numSeq):
    for seq in sequence_index:
        # If there are two embedded motifs
        seq_muts = [name for name in mut_loc_dict.keys() 
                         if mut_loc_dict[name]['seq'] == seq]
        if len(seq_muts) > 0:

            # Add original sequence to the list of sequences
            mutated_seq_list.append(sequences[seq])
            orig_dict = {'array_ind': ind, 'seq': seq, 'label': 'original', 
                         'mut_key': 'original', 'mut_start': -1}
            if per_base_map:
                orig_dict['orig_letter'] = -1; orig_dict['mut_letter'] = -1; 

            mutated_seq_key.iloc[ind,:] = orig_dict
            ind += 1

            for m in seq_muts:
                mut_start = mut_loc_dict[m]['mut_start']
                mut_end = mut_loc_dict[m]['mut_end']

                if per_base_map and mut_end - mut_start == 1:

                        # If mutation is a single base
                        orig_letter  = get_orig_letter(sequences[seq][mut_start, :])
                        seq_mutants = [el for el in mutants if el != orig_letter]
                        for mut_letter in seq_mutants:
                            mutated_seq = copy.deepcopy(sequences[seq])
                            mut_index = get_letter_index(mut_letter)
                            mutated_seq[mut_start, :] = 0
                            mutated_seq[mut_start, mut_index] = 1
                            mutated_seq_list.append(mutated_seq)
                            label = 'seq_{0}_{1}to{2}_at{3}'.format(
                                            str(seq), orig_letter, 
                                            mut_letter, str(mut_start))
                            mutated_seq_key.iloc[ind, :]={'array_ind':ind, 'seq': seq, 'label': label, 
                                                          'mut_key': m, 'mut_start': mut_start,
                                                          'orig_letter': orig_letter, 
                                                          'mut_letter': mut_letter}
                            ind += 1

                elif per_base_map and mut_end - mut_start > 1 :

                    # raise ValueError('Currently may not ask for a --per_base_map with mutation size > 1')
                    ### Issue here is that basically need to re-write  mutant_loc_dict
                    ### Because otherwise all the different mutations have the same "name" m
                    for current_mut_start in range(mut_start, mut_end):
                        # In this case need new names because otherwise they overwrite
                        # new_m = '{0}_base{1}'.format(m, current_mut_start)
                        # Iterate through each base from mut_start to mut_end
                        orig_letter  = get_orig_letter(sequences[seq][current_mut_start, :])
                        seq_mutants = [el for el in mutants if el != orig_letter]
                        for mut_letter in seq_mutants:
                            mutated_seq = copy.deepcopy(sequences[seq])
                            mut_index = get_letter_index(mut_letter)
                            mutated_seq[current_mut_start, :] = 0
                            mutated_seq[current_mut_start, mut_index] = 1
                            mutated_seq_list.append(mutated_seq)
                            label = 'seq_{0}_{1}to{2}_at{3}'.format(
                                            str(seq), orig_letter, 
                                            mut_letter, str(current_mut_start))
                            mutated_seq_key.iloc[ind, :]={'array_ind':ind, 'seq': seq, 'label': label, 
                                                          'mut_key': m, 'mut_start': current_mut_start,
                                                          'orig_letter': orig_letter, 
                                                          'mut_letter': mut_letter}
                            ind += 1

                # If the mutation size is greater than 1 then just set to 0s or GC (?)
                elif per_base_map == False:
                    assert mut_end - mut_start > 1
                    mutated_seq = copy.deepcopy(sequences[seq])
                    mutated_seq[mut_start:mut_end, :] = 0 
                    mutated_seq_list.append(mutated_seq)
                    label = m + ';' + 'seq_'+ str(seq) + '_' + 'loc{0}-{1}'.format(
                                                            str(mut_start), str(mut_end))
                    mutated_seq_key.iloc[ind, :] = {'array_ind':ind, 'seq': seq, 'label': label,
                                                   'mut_key': m, 'mut_start': mut_start}
                    ind += 1

                else:
                    raise ValueError('per_base_map is {0} but mut_end - mut_start is {1} '.format(
                                     (mut_end - mut_start)))


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

    ### For each task and sequence
    delta_dict = {}
    for task in tasks:
        delta_dict[task] = {}
        for seq in sequence_index:
            delta_dict[task][seq] = {}
            # Get importance scores of original sequence
            orig_ind = mutated_seq_key[mutated_seq_key.seq == seq].loc[
                                       mutated_seq_key.label == 'original', 'array_ind'].tolist()[0]
            # Get all mutants of that sequence
            # seq_mutants = mutated_seq_key[mutated_seq_key.seq == seq].loc[
            #                               mutated_seq_key.label != 'original', 'mut_key'].tolist()
            seq_mutant_indices = mutated_seq_key[mutated_seq_key.seq == seq].loc[
                                          mutated_seq_key.label != 'original', :].index.tolist()
            # Iterate through mutants
            for mut_i in seq_mutant_indices:  
                m = mutated_seq_key.loc[mut_i, 'mut_key']
                m_label = mutated_seq_key.loc[mut_i, 'label']
                delta_dict[task][seq][m_label] = {}
                resp_starts = mut_loc_dict[m]['resp_start']
                resp_ends = mut_loc_dict[m]['resp_end']
                mut_name = mut_loc_dict[m]['mut_name']
                resp_names = mut_loc_dict[m]['resp_names']
                mut_ind = mutated_seq_key.loc[mut_i, 'array_ind']
                mut_start = mutated_seq_key.loc[mut_i, 'mut_start']
                mut_end = mutated_seq_key.loc[mut_i, 'mut_end']
                for r in range(len(resp_starts)):
                    response_mut_profile = score_dict[task][mut_ind][resp_starts[r]:resp_ends[r], :]
                    orig_profile = score_dict[task][orig_ind][resp_starts[r]:resp_ends[r], :]
                    # Set orig profile to zero at mutation (for motifs this means delta will be 0, 
                    # for bases it means differential will be magnitude of mutant)
                    # orig_profile[mut_start:mut_end, :] = 0
                    delta_profile = orig_profile - response_mut_profile
                    # Set delta profile to 0 at mutation
                    delta_profile[mut_start:mut_end, :] = 0
                    if mutated_seq_preds is not None:
                        pred_diff = (mutated_seq_preds[orig_ind] - mutated_seq_preds[mut_ind])[0]
                    else:
                        pred_diff = None
                    r_key = 'resp_{0}_{1}to{2}'.format(resp_names[r], str(resp_starts[r]), str(resp_ends[r]))
                    # r_key = 'mut_{0};response_{1}_at_{2}'.format(m, resp_names[r], str(resp_starts[r]))
                    ### Combine code below
                    if capture_seqs_max_thresh is None:
                        delta_dict[task][seq][m_label][r_key] = {'orig_profile': orig_profile,
                                                                 'mut_profile': response_mut_profile,
                                                                 'delta_profile': delta_profile,
                                                                 'prediction_diff': pred_diff,
                                                                 'mut_start': mut_start,
                                                                 'resp_start': resp_starts[r],
                                                                 'resp_end': resp_ends[r],
                                                                 'mut_key': m,
                                                                 'mut_name': mut_name,
                                                                 }                        
                    elif capture_seqs_max_thresh is not None and abs(delta_profile).max() > capture_seqs_max_thresh:
                        print('Max score exceeded threshold, adding sequence')
                        delta_dict[task][seq][m_label][r_key] = {'orig_profile': orig_profile,
                                                                 'mut_profile': response_mut_profile,
                                                                 'delta_profile': delta_profile,
                                                                 'prediction_diff': pred_diff,
                                                                 'mut_start': mut_start,
                                                                 'resp_start': resp_starts[r],
                                                                 'resp_end': resp_ends[r],
                                                                 'mut_key': m,
                                                                 'mut_name': mut_name,
                                                                 }
                    else:
                        continue

    return delta_dict


def dfim_per_element(delta_dict, task, seq, all_mut_pos, 
                     all_resp_pos, diagonal_value,
                     operations, operation_axes,
                     absolute_value):

    # Allocate data frame with all mutant names and response names
    dfim_df = pd.DataFrame(index=all_mut_pos,
                           columns=all_resp_pos,
                           data=diagonal_value)

    # Extract all the differential profiles
    for m in delta_dict[task][seq].keys():
        for r in delta_dict[task][seq][m].keys():

            if absolute_value:
                delta_profile = abs(delta_dict[task][seq][m][r]['delta_profile'])
            else:
                delta_profile = delta_dict[task][seq][m][r]['delta_profile']

            i_score = delta_profile

            for o in range(len(operations)):

                # Perform operations
                i_score = np.apply_along_axis(operations[o],
                                              operation_axes[o],
                                              i_score)
            # Combine into a map
            dfim_df.loc[delta_dict[task][seq][m][r]['mut_start'],
                        delta_dict[task][seq][m][r]['resp_start']] = i_score.tolist() 

    return dfim_df

def dfim_per_base(dfim_key, resp_size, diagonal_value,
                  delta_dict, task, seq, absolute_value,
                  operations, operation_axes, 
                  summarize_per_pos=False):

    # Pandas data frame with element and then per base response  
    # Need every response to be the same length 
    # Currently don't use operations

    mut_starts = dfim_key.mut_start.unique().tolist()

    dfim_array = np.zeros((len(mut_starts), 4, 4, resp_size, 4))

    for i in dfim_key.index:

        m = dfim_key.loc[i, 'mut_key']
        label = dfim_key.loc[i, 'label']
        start_ind = np.where(np.array(mut_starts) == dfim_key.loc[i, 'mut_start'])[0][0]
        orig_letter_ind = get_letter_index(dfim_key.loc[i, 'orig_letter'])
        mut_letter_ind = get_letter_index(dfim_key.loc[i, 'mut_letter'])
        if len(delta_dict[task][seq][label].keys()) == 1:
            r = delta_dict[task][seq][label].keys()[0]
        else:
            print("Unsure how to deal with multiple response locations for per-base-map")

        if absolute_value:
            delta_profile = abs(delta_dict[task][seq][label][r]['delta_profile'])
        else:
            delta_profile = delta_dict[task][seq][label][r]['delta_profile']

        dfim_array[start_ind, orig_letter_ind, mut_letter_ind, :, :] = delta_profile


    if operations is not None:

        assert len(operation_axes) == len(operations)

        for o in range(len(operations)):

            # Perform operations
            dfim_array = np.apply_along_axis(operations[o],
                                             operation_axes[o],
                                             dfim_array)

    return dfim_array


def compute_dfim(delta_dict, sequence_index, tasks,
                 operations=[np.sum, np.max], operation_axes=[1, 0],
                 absolute_value=True,
                 annotate=False, mutated_seq_key=None,
                 diagonal_value=-1,
                 per_base_map=False):

    """
    operations - MOTIFS: first take sum over all letters, then take mean over positions (FOR MOTIFS)
               - INDIVIDUAL BASES: max over base position
    absolute_value - will take absolute value of delta_profile before operations
    annotate - False will just use position as the index/column name
               True will capture the name

    Breaks into:
    - element level summary
    - base level summary
      - mutate motif look at per-base effect (just look at effect)
      - mutate base look at per-base effect (need to keep original base -> base)

    """

    assert len(operations) == len(operation_axes)

    dfim_dict = {}

    # For each sequence and task
    for task in tasks:

        dfim_dict[task] = {}

        for seq in sequence_index:

            if per_base_map:

                dfim_dict[task][seq] = {}

                mut_keys = mutated_seq_key[mutated_seq_key.seq == seq].loc[
                                           mutated_seq_key.label != 'original', 
                                           'mut_key'].unique().tolist()

                for mkey in mut_keys:

                    all_mut_index = mutated_seq_key.loc[mutated_seq_key.mut_key == mkey, 
                                                         'label'].tolist()

                    resp_sizes = [delta_dict[task][seq][m][r]['resp_end'
                                     ] - delta_dict[task][seq][m][r]['resp_start']
                                     for m in delta_dict[task][seq].keys()
                                     for r in delta_dict[task][seq][m].keys()]
     
                    if len(np.unique(resp_sizes)) == 1:

                        # print('Detected response elements of same size %s, making DFIM'%resp_sizes[0])

                        dfim_array =  dfim_per_base(dfim_key=mutated_seq_key[mutated_seq_key.mut_key == mkey], 
                                                    resp_size=np.unique(resp_sizes)[0], 
                                                    diagonal_value=diagonal_value,
                                                    delta_dict=delta_dict, task=task,
                                                    seq=seq, absolute_value=absolute_value,
                                                    operations=operations, 
                                                    operation_axes=operation_axes)

                        dfim_dict[task][seq][mkey] = dfim_array

                        ### Better is to pass [mutated_seq_key.loc[mutated_seq_key.mut_key == m]
                        ### Then make an array and iterate through each of those [mut_start, orig_letter, mut_letter]
 
                    else:

                        raise ValueError('''Warning: response locations are of variable sizes, '''
                              '''generating per element map by default''')

            else:

                # Capture all the mutations and responses
                all_mut_pos = np.unique([delta_dict[task][seq][m][r]['mut_start'
                                             ] for m in delta_dict[task][seq].keys()
                                               for r in delta_dict[task][seq][m].keys()])
                all_resp_pos = np.unique([delta_dict[task][seq][m][r]['resp_start'
                                             ] for m in delta_dict[task][seq].keys()
                                               for r in delta_dict[task][seq][m].keys()])   


                dfim_df = dfim_per_element(delta_dict=delta_dict, 
                                           task=task, seq=seq, 
                                           all_mut_pos=all_mut_pos, 
                                           all_resp_pos=all_resp_pos, 
                                           diagonal_value=diagonal_value,
                                           operations=operations, 
                                           operation_axes=operation_axes,
                                           absolute_value=absolute_value)

                # Make function to annotate with simdna (supply annotation function as argument)
                if annotate:

                    # Add annotations to the rows
                    # Current depends on simdna structure, useful for motif analysis only

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



def capture_strong_interactions(dfim_dict, delta_dict, 
                                score_thresh=0.01,
                                top_pct=None,
                                pickle_file=None):

    """
    Can automate detection of what a strong threshold would be
    thresholds are applied per task
    set score_thresh=None and top_pct=None to capture all sequences

    """

    capture_dict = {}

    for task in delta_dict.keys():
        capture_dict[task] = {}

        if top_pct is not None:

            # Get distribution of maximal interaction scores
            if top_pct < 1:
                top_pct = top_pct * 100

            all_scores = [abs(delta_dict[t][s][m][r]['delta_profile']).max()
                                for t in delta_dict.keys()
                                for s in delta_dict[t].keys()
                                for m in delta_dict[t][s].keys()
                                for r in delta_dict[t][s][m].keys()]

            score_thresh = scipy.stats.scoreatpercentile(all_scores, top_pct)

        elif score_thresh is not None:

            print('--score_thresh and --top_pct not provided, taking all sequences')
            score_thresh = np.min(all_scores)

        for seq in delta_dict[task].keys():
            capture_dict[task][seq] = {}
            for m_label in delta_dict[task][seq].keys():
                for r in delta_dict[task][seq][m_label].keys():

                    if abs(delta_dict[task][seq][m_label][r][
                                'delta_profile']).max() > score_thresh:

                        capture_key = '{0}_{1}'.format(m_label, r)
                        mut_key = delta_dict[task][seq][m_label][r]['mut_key']

                        capture_dict[task][seq][capture_key] = {
                                    'delta_dict': delta_dict[task][seq][m_label][r],
                                    'dfim': dfim_dict[task][seq][mut_key]}

    if pickle_file is not None:

        pickle.dump(capture_dict, open(pickle_file, 'w'))
        print("Dumped capture_dict to %s"%pickle_file)


    return capture_dict


