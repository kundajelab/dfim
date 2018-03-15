### Utils for DFIM
### Peyton Greenside
### Kundaje Lab, Stanford University

import os, sys
import numpy as np
import pandas as pd
import random

import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import matplotlib.gridspec as grd

import pybedtools
from Bio import SeqIO


def get_orig_letter(one_hot_vec):
    try:
        assert(len(np.where(one_hot_vec!=0)[0]) == 1)
    except:
        return 'N'
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

def load_sequences_from_bed(bed_file=None, 
                            bed=None,
                            genome_file=None,
                            return_fasta=False):

    """
    If working on lab cluster can use
    /mnt/data/annotations/by_organism/human/hg19.GRCh37/hg19.genome.fa
    """

    # t0 = time.time()

    records = SeqIO.to_dict(SeqIO.parse(
                            open(genome_file), 'fasta'))

    if bed_file is not None:
        bed = pd.read_table(bed_file, header=None)

    short_seq_records = []
    for ind in bed.index:
        name = bed.ix[ind, 'chrom'] if 'chrom' \
                        in bed.columns else bed.ix[ind, 0] 
        start = bed.ix[ind, 'start'] if 'start'  \
                        in bed.columns else bed.ix[ind, 1]
        end = bed.ix[ind, 'end'] if 'end' \
                        in bed.columns else bed.ix[ind, 2]
        long_seq_record = records[name]
        long_seq = long_seq_record.seq
        alphabet = long_seq.alphabet
        short_seq = str(long_seq)[start:end]
        short_seq_records.append(short_seq)

    # sequences = util.setOfSeqsTo2Dimages(short_seq_records)
    sequences = np.array([one_hot_encode(s) for s in short_seq_records])
    sequences = sequences.astype('float32')

    # t1 = time.time()
    # print 'Time to load sequences: %s'%(t1-t0)

    if return_fasta:
        return (sequences, short_seq_records)
    else:
        return sequences

def random_shuffle_one_hot(sequence, seed=1):
    random.seed(seed)
    if sequence.shape[-2] > sequence.shape[-1]:
        sample_index = -2
    else:
        sample_index = -1
    sample_size = sequence.shape[sample_index]
    new_index = random.sample(range(sample_size), sample_size)
    new_sequence = sequence[new_index, :] if sample_index == -2 \
                                          else sequence[:, new_index]
    return new_sequence

def random_shuffle_fasta(fasta_seq, seed=1):
    random.seed(seed)
    l = list(fasta_seq)
    random.shuffle(l)
    shuf_fasta_seq = ''.join(l)
    return shuf_fasta_seq


def convert_one_hot_to_fasta(sequences):
    fasta_list = []
    for s in range(sequences.shape[0]):
        fasta_string = ''
        for i in range(sequences[s].shape[-2]):
            letter = get_orig_letter(sequences[s, i, :])
            fasta_string = fasta_string + letter
        assert len(fasta_string) == sequences[s].shape[-2]
        fasta_list.append(fasta_string)
    return fasta_list

def process_seqs_and_locations_from_bed(bed_file, genome_file,
                                        seq_len=1000, flank_size=15):
    """ 
    Returns: dictionary with key as:
        seq_{seq_number}_emb_{embedding_label}
        and with entry as a dictionary giving
        the sequence, start, end, and name of mutation 
        and lists with starts, ends, names of response locations
    """
    bed_df = pd.read_table(bed_file)

    seqlet_loc_dict = {}

    seq_coords = []

    for bed_ind in range(bed_df.shape[0]):

        chrom = bed_df.iloc[bed_ind, 0]
        start = bed_df.iloc[bed_ind, 1]
        end = bed_df.iloc[bed_ind, 2]
        mut_size = end - start
        seq_flank_size = (seq_len - mut_size) / 2.
        if mut_size % 2 != 0:
            pre_mut_flank = int(np.ceil(seq_flank_size))
            post_mut_flank = int(np.floor(seq_flank_size))
        else: 
            pre_mut_flank = post_mut_flank = int(seq_flank_size)

        seq_coords.append([chrom, start - pre_mut_flank, end + post_mut_flank])

        feature_start = start - start + pre_mut_flank
        feature_end = end - start + post_mut_flank

        resp_starts = [pre_mut_flank - flank_size]
        resp_ends = [pre_mut_flank + mut_size + flank_size]
        resp_names = ['flank_%s'%flank_size]

        return_key = 'seq_%s'%(bed_ind)

        return_dict = {'seq': bed_ind,
                       'mut_start': pre_mut_flank,
                       'mut_end': pre_mut_flank + mut_size,
                       'mut_name': '%s:%s-%s'%(chrom, start, end),
                       'resp_start': resp_starts,
                       'resp_end': resp_ends,
                       'resp_names': resp_names}

        seqlet_loc_dict[return_key] = return_dict

    peak_df = pd.DataFrame(seq_coords)

    # Now generate sequences
    sequences = load_sequences_from_bed(bed=peak_df, 
                                        genome_file=genome_file)

    return (sequences, seqlet_loc_dict)


def process_seqs_and_locations_from_bed_and_labels(bed_file, labels_file, 
                                                   genome_file,
                                                   seq_len=1000, flank_size=15):
    """ 
    Returns: dictionary with key as:
        seq_{seq_number}_emb_{embedding_label}
        and with entry as a dictionary giving
        the sequence, start, end, and name of mutation 
        and lists with starts, ends, names of response locations
    """

    # Find intersection of bed coordinates with known labels
    bed_df = pd.read_table(bed_file)
    labels_df = pd.read_table(labels_file)

    bed_bedtool = pybedtools.BedTool.from_dataframe(bed_df)
    labels_bedtool = pybedtools.BedTool.from_dataframe(labels_df)
    intersect_df = labels_bedtool.intersect(bed_bedtool, 
                                            wa=True, wb=True
                                            ).to_dataframe()

    feature_start_col = len(labels_df.columns) + 1
    feature_end_col = len(labels_df.columns) + 2

    seqlet_loc_dict = {}

    seq_coords = []

    for bed_ind in range(intersect_df.shape[0]):

        chrom = intersect_df.iloc[bed_ind, 0]
        start = intersect_df.iloc[bed_ind, 1]
        end = intersect_df.iloc[bed_ind, 2]
        peak_size = end - start
        mut_start = intersect_df.iloc[bed_ind, feature_start_col]
        mut_end = intersect_df.iloc[bed_ind, feature_end_col]
        seq_flank_size = (seq_len - peak_size) / 2.
        if (peak_size % 2) != 0:
            pre_mut_flank = int(np.ceil(seq_flank_size))
            post_mut_flank = int(np.floor(seq_flank_size))
        else: 
            pre_mut_flank = post_mut_flank = int(seq_flank_size)
        seq_coords.append([chrom, start - pre_mut_flank, end + post_mut_flank])

        feature_start = np.max([0, mut_start - start + pre_mut_flank])
        feature_end = np.min([seq_len, mut_end - start + post_mut_flank])

        resp_starts = [np.max([0, feature_start - flank_size])]
        resp_ends = [np.min([seq_len, feature_end + flank_size])]
        resp_names = ['flank_%s'%flank_size]

        return_key = 'seq_%s'%(bed_ind)

        return_dict = {'seq': bed_ind,
                       'mut_start': feature_start,
                       'mut_end': feature_end,
                       'mut_name': '%s:%s-%s;mut_%s_to_%s'%(
                                        chrom, start, end, mut_start, mut_end),
                       'resp_start': resp_starts,
                       'resp_end': resp_ends,
                       'resp_names': resp_names}

        seqlet_loc_dict[return_key] = return_dict

    peak_df = pd.DataFrame(seq_coords)

    # Now generate sequences
    sequences = load_sequences_from_bed(bed=peak_df, 
                                        genome_file=genome_file)

    return (sequences, seqlet_loc_dict)

def rank_by_min_pval(dfim_pval_dict):
    """
    Return task, seq, mut with smallest p-values
    """

    all_results = [((task, seq, mut), np.min(dfim_pval_dict[task][seq][mut]))
                        for task in dfim_pval_dict.keys()
                        for seq in dfim_pval_dict[task].keys()
                        for mut in dfim_pval_dict[task][seq].keys()]

    sorted_results = sorted(all_results, key = lambda x: x[1])

    return sorted_results

def rank_by_max_total_interaction(dfim_dict, dfim_pval_dict=None):
    """
    Return task, seq, mut with smallest p-values
    """

    if dfim_pval_dict is None:

        all_results = [((task, seq, mut), np.sum(dfim_dict[task][seq][mut]))
                            for task in dfim_dict.keys()
                            for seq in dfim_dict[task].keys()
                            for mut in dfim_dict[task][seq].keys()]

    else:

        # Only take sum of significant ones to mitigate noise
        all_results = [((task, seq, mut), np.sum(dfim_dict[task][seq][mut] * (
                                        dfim_pval_dict[task][seq][mut] < 0.05)))
                            for task in dfim_dict.keys()
                            for seq in dfim_dict[task].keys()
                            for mut in dfim_dict[task][seq].keys()]

    sorted_results = sorted(all_results, key = lambda x: -1 * x[1])

    return sorted_results


### Dinucleotide shuffling
### Credit to 
### https://github.com/kundajelab/deeplift/blob/master/deeplift/dinuc_shuffle.py

from collections import defaultdict
from random import shuffle

def prepare_edges(s):
    edges = defaultdict(list)
    for i in xrange(len(s)-1):
        edges[s[i]].append(s[i+1])
    return edges

def shuffle_edges(edges):
    #for each character, remove the last edge, shuffle, add edge back
    for char in edges:
        last_edge = edges[char][-1]
        edges[char] = edges[char][:-1]
        the_list = edges[char]
        shuffle(the_list)
        edges[char].append(last_edge)
    return edges

def traverse_edges(s, edges):
    generated = [s[0]]
    edges_queue_pointers = defaultdict(lambda: 0)
    for i in range(len(s)-1):
        last_char = generated[-1]
        generated.append(edges[last_char][edges_queue_pointers[last_char]])
        edges_queue_pointers[last_char] += 1
    return "".join(generated)

def dinuc_shuffle(s):
    s = s.upper()
    return traverse_edges(s, shuffle_edges(prepare_edges(s)))



