### Utils for DFIM
### Peyton Greenside
### Kundaje Lab, Stanford University

import os, sys
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import matplotlib.gridspec as grd

def one_hot_encode(sequence):

    one_hot_array = np.zeros((len(sequence), 4))

    for (pos, char) in enumerate(sequence):

        if (char=="A" or char=="a"):
            char_pos = 0;
        elif (char=="C" or char=="c"):
            char_pos = 1;
        elif (char=="G" or char=="g"):
            char_pos = 2;
        elif (char=="T" or char=="t"):
            char_pos = 3;
        elif (char=="N" or char=="n"):
            continue;
        else:
            raise RuntimeError("Unsupported character: "+str(char));

        one_hot_array[pos, char_pos] = 1

    return one_hot_array



def get_correct_predictions_from_model(model, sequences, labels, 
                                       pos_threshold=0.5, neg_threshold=None, 
                                       label_key_column=3):
    """
    By default just returns positives
    You can supply a neg_threshold (i.e. 0.5) to return correct negatives below
    Labels are supplied
    Returns dictionary with with each key as task and 
        a list with indices of correct predictions
    """
    correct_pred_dict = {}
    predictions = model.predict(sequences)

    for task in range(predictions.shape[1]):
        correct_predictions = []
        task_labels = labels.ix[:, task + label_key_column + 1].tolist()
        task_predictions = predictions[:, task]
        correct_pred_dict[task] = get_correct_predictions(task_labels, task_predictions,
                                                          pos_threshold=pos_threshold, 
                                                          neg_threshold=neg_threshold)
    return correct_pred_dict

def get_correct_predictions(true_labels, predicted_labels,
                            pos_threshold=0.5,
                            neg_threshold=None):
    """
    Just return list of indices where prediction is greater than pos_threshold
    If neg_threshold is not None (is supplied, i.e. 0.5), will also 
    return indices for negative correctly predicted examples
    """
    correct_predictions = []

    for i in xrange(len(true_labels)):
        if true_labels[i] == 1 and predicted_labels[i] > pos_threshold:
            correct_predictions.append(i)
        if neg_threshold is not None:
            if true_labels[i] == 0 and predicted_labels[i] < neg_threshold:
                correct_predictions.append(i)

    return correct_predictions

def process_locations_from_simdata(sequences, simdata_file):
    """ 
    Returns: dictionary with key as:
        seq_{seq_number}_emb_{embedding_label}
        and with entry as a dictionary giving
        the sequence, start, end, and name of mutation 
        and lists with starts, ends, names of response locations
    """
    simdata_df = pd.read_table(simdata_file, compression='gzip')

    assert simdata_df.shape[0] == sequences.shape[0]

    seqlet_loc_dict = {}
    for seq_ind in range(sequences.shape[0]):

        if pd.isnull(simdata_df.loc[seq_ind, 'embeddings']):
            print('No embeddings for seq %s'%seq_ind)
            continue

        embeddings = simdata_df.loc[seq_ind, 'embeddings'].split(',')

        for emb in embeddings:

            pos_start = int(emb.split('_',1)[0].replace('pos-', ''))
            pos_end = pos_start + len(emb.split('-')[-1])
            motif = emb.split('_',1)[1].split('-')[0]
            seq_embed = emb.split('_',1)[1].split('-')[1]

            resp_starts = [int(e.split('-')[1].split('_')[0]) 
                            for e in embeddings if e != emb]
            resp_lengths = [len(e.split('-')[-1]) 
                            for e in embeddings if e != emb]
            resp_ends = [resp_starts[i] + resp_lengths[i] 
                            for i in range(len(resp_starts))]
            resp_names = [e.split('-')[1].split('_')[1] 
                            for e in embeddings if e != emb]

            return_key = 'seq_%s_emb_%s'%(seq_ind, emb)
            return_dict = {'seq': seq_ind,
                           'mut_start': pos_start,
                           'mut_end': pos_end,
                           'mut_name': motif.split('_')[0],
                           'resp_start': resp_starts,
                           'resp_end': resp_ends,
                           'resp_names': resp_names}

            seqlet_loc_dict[return_key] = return_dict

    return seqlet_loc_dict

def plot_delta_tracks(DDL_score_list, title_list, plotPosEvery=1, 
                      heightPerTrack=3, yRangeVec=None, 
                      axvline_loc=None):
    """
    REFACTOR FROM pyDNAbinding and DeepLIFT utils
    """
    plt.clf()
    width = DDL_score_list[0].shape[1]
    fig_width = 20 + width/10
    fig = plt.figure(figsize=(fig_width,heightPerTrack*len(DDL_score_list)))
    yMax = np.max([arr.max() for arr in DDL_score_list]) + .01
    yMin = np.min([arr.min() for arr in DDL_score_list]) - .01
    for (i,array) in enumerate(DDL_score_list):
        ax = fig.add_subplot(len(DDL_score_list),1,i+1)
        if (array.shape[0]==4):
            letter_heights=array.T
            pos_heights = np.copy(letter_heights)
            pos_heights[letter_heights < 0] = 0
            neg_heights = np.copy(letter_heights)
            neg_heights[letter_heights > 0] = 0
            for x_pos, heights in enumerate(letter_heights):
                letters_and_heights = sorted(deepLIFTutils.izip(heights, 'ACGT'))
                y_pos_pos = 0.0
                y_neg_pos = 0.0
                for height, letter in letters_and_heights:
                    if height > 0:
                        deepLIFTutils.add_letter_to_axis(ax, letter, -0.5+x_pos, y_pos_pos, height)
                        y_pos_pos += height
                    else:
                        deepLIFTutils.add_letter_to_axis(ax, letter, -0.5+x_pos, y_neg_pos, height)
                        y_neg_pos += height
            if (i==len(DDL_score_list)):
                ax.set_xlabel('pos')
            ax.set_aspect(aspect='auto', adjustable='box')
            ax.autoscale_view()
        elif (array.shape[0]==1):        
            ax.plot(range(arr.shape[1]), array.squeeze(), 'k', lw=0.5)
        else:
           raise RuntimeError("Unsure how to deal with shape "+str(arr.shape))
        ax.axhline(0, linestyle='dashed', color='black')
        ax.set_title(title_list[i])
        ax.set_ylabel('DDL Score')
        ax.set_xlim(-1, array.shape[1]+1)
        ax.set_ylim([yMin, yMax])
        if axvline_loc is not None:
            ax.axvline(axvline_loc)
        xticks_locations=range(-1,array.shape[1]+1)
        ax.set_xticks(xticks_locations[::plotPosEvery])
        ax_2=ax.twiny()
    fig.tight_layout()
    plt.show()
