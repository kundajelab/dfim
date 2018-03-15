# DFIM
Deep Feature Interaction Maps (DFIM)   
Contact: Peyton Greenside (pgreens @ stanford.edu).

## Description

This code base enables interpretation of interactions learned by a deep learning model and represents these interactions in a Deep Feature Interaction Map (DFIM). It is designed primarily to identify interactions between sequence elements in regulatory DNA sequence, but can be applied to other modalities.

Dependencies can be observed between individual bases (i.e. between variants and each base in the surrounding sequence), between motifs (i.e. between transcription factor binding domains) or between a motif and a set of bases (i.e. effect of mutation on a binding motif). See functions `dfim_per_element`, `dfim_per_base`, and `dfim_element_by_base` in core.py.

DFIM works through creating strategic perturbation of features of interest (or an entire DNA sequence) and using backpropagation-based importance scoring methods to efficiently estimate the effect on the surrounding sequence.

![DFIM_Outline_Figure](/examples/DFIM_description_image.png)

## Usage

You can construct your desired features (source elements to mutate and target elements that respond) with a dictionary of the following format where every key is your desired label for the mutation and every value is a dictionary containing the follow elements:
  
```
mut_dict['seq_0'] = {'seq': 0, 
		     'mut_start': 45,  
		     'mut_end': 55,  
		     'mut_name': chr1:100-200;mut_45_to_55,  
		     'resp_start': [0],  
		     'resp_end': [100],  
		     'resp_names': ['flank_size_100']} 
```
  
This will take the first sequence of length 100, make a mutation at the motif in the center and assess interactions with the surrounding 100 bases. Thus a dictionary with 10 such entries will compute interactions for 10 sequences. `resp_start`, and `resp_end` can be lists if you wish to assess the effect on multiple motifs.

Given your sequences and a mutation dictionary as above, there are several steps to compute a DFIM.

1) Generate mutated sequences given sequences and the mutation dictionary (see `generate_mutants_and_key()` in core.py)
2) Compute importance scores for all original and mutated sequences (see `compute_importance()` in core.py)
	- any available importance scoring method in the deeplift package is available (i.e. DeepLIFT with supplied reference, Gradient Input, Guided Backpropagation, etc.)
3) Compute delta profiles between original and mutated sequence (see `compute_delta_profiles()` in core.py)
4) Compute DFIM from all delta profiles (see `compute_dfim()` in core.py)  
[OPTIONAL for NULL model]
5) Repeat steps 1-4) for shuffled sequences  
6) Compute p-values given DFIM from both original and shuffled sequences (see `assign_pval()` in null_model.py) 
 
We provide utilities to visualize the delta profiles or the DFIM (see `plot_delta_tracks()` and `plot_dfim()` in plot.py)

See the examples/embedded_motif_interaction_example.ipynb notebook for a complete demonstration of how to use the code base. 

## Installation

Installation can be done with `python setup.py install` or through `pip install dfim`. 

Requirements: scipy, numpy>=1.11, pandas, deeplift  
Optional: matplotlib, h5py, biopython
