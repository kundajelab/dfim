### Null Model for DFIM scores
### Peyton Greenside
### Kundaje Lab, Stanford University

import os, sys
import pandas as pd
import numpy as np

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
        if sequences.shape == 3:
            # Assume (Batch, SEQ_LEN, 4)
            SHUF_AXIS = 1
            shuf_sequences = sequences[:, np.random.permutation(sequences.shape[SHUF_AXIS]), :]
        elif sequences.shape == 4:
            # Assume (Batch, 1, 4, SEQ_LEN)
            SHUF_AXIS = 3
            shuf_sequences = sequences[:, :, :, np.random.permutation(sequences.shape[SHUF_AXIS])]
        else:
            raise ValueError(
                    '''Not sure how to deal with sequences 
                    of shape {0}'''.format(sequences.shape))

    return shuf_sequences


def generate_dfim_null_distribution(mut_loc_dict, deeplift_func,
                                    sequences=None, seq_fastas=None,
                                    dinuc=False, dl_task_idx=0):
    # Generate shuffled sequences
    if sequences is not None:
        shuf_sequences = shuffle_seq_from_one_hot(sequences, dinuc=dinuc)
    elif seq_fastas is not None:
        shuf_sequences = shuffle_seq_from_fasta(seq_fastas, dinuc=dinuc)
    else:
        raise ValueError('Must provide either --sequences or --seq_fastas')

