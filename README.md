# DFIM
Deep Feature Interaction Maps (DFIM)   
Contact: Peyton Greenside (pgreens @ stanford.edu).

Publication: Greenside PG, Shimko T, Fordyce P, Kundaje A. Discovering epistatic feature interactions from neural network models of regulatory DNA sequences
bioRxiv 302711; doi: https://doi.org/10.1101/302711
(Accepted to ECCB 2018, To be published in Bioinformatics)

## Abstract
Motivation: Transcription factors bind regulatory DNA sequences in a combinatorial manner to modulate gene expression. Deep neural networks (DNNs) can learn the cis-regulatory grammars encoded in regulatory DNA sequences associated with transcription factor binding and chromatin accessibility. Several feature attribution methods have been developed for estimating the predictive importance of individual features (nucleotides or motifs) in any input DNA sequence to its associated output prediction from a DNN model. However, these methods do not reveal higher-order feature interactions encoded by the models. 
Results: We present a new method called Deep Feature Interaction Maps (DFIM) to efficiently estimate interactions between all pairs of features in any input DNA sequence. DFIM accurately identifies ground truth motif interactions embedded in simulated regulatory DNA sequences. DFIM identifies synergistic interactions between GATA1 and TAL1 motifs from in vivo TF binding models. DFIM reveals epistatic interactions involving nucleotides flanking the core motif of the Cbf1 TF in yeast from in vitro TF binding models. We also apply DFIM to regulatory sequence models of in vivo chromatin accessibility to reveal interactions between regulatory genetic variants and proximal motifs of target TFs as validated by TF binding quantitative trait loci. Our approach makes significant strides in improving the interpretability of deep learning models for genomics.

## Description

### What is an epistatic feature interaction?
A neural network can learn a complex non-linear mapping from the features in an input instance to its associated output label. If the effect/contribution/importance of one feature on the output of a model depends on another feature, the two features are exhibiting epistatic interactions. These interactions and dependence relationships would manifest as non-additive (non-linear) effects on the output. DFIM is a method for estimating epistatic interaction scores between pairs of features from a trained deep learning model. The interactions are represented using a Deep Feature Interaction Map (DFIM). 

The approach is general and can be applied to any deep neural network and any input data modality. However, the code in this repository was specifically designed to interpret neural network models of regulatory DNA sequence by identifying interactions between sequence features (nucleotides and motifs) in regulatory DNA sequence. The primary motivation is regulatory DNA sequences exhibit widespread cooperative/epistatic binding of multiple regulatory proteins (transcription factors). This cooperative/epistatic binding activity is often mediated by DNA sequence features (motifs). Moreover, nucleotides flanking and within DNA sequence motifs (binding sites) that determine the binding specificity of individual transcription factors also exhibit epistatic interactions. Our goal was to extract these types of epistatic interactions between motifs and/or nucleotides that are potentially learned by neural networks trained to predict transcription factor binding and associated molecular markers (e.g chromatin accessibility) from DNA sequence.

To interpret and score feature interactions in any input instance/example, we first create a perturbation of a source feature of interest (e.g. a nucleotide or motif). We then use efficient backpropagation-based importance scoring methods to estimate the effect of perturbing the source feature on the predictive importance of all other features in the surrounding sequence. The change in the importance score of a target feature due a perturbation in a source feature, provides an estimate of a feature interaction score. Interactions can be computed between individual nucleotides, between motifs (contiguous subsequences) or between nucleotides and motifs. (See functions `dfim_per_element`, `dfim_per_base`, and `dfim_element_by_base` in core.py.)


![DFIM_Outline_Figure](/data/DFIM_description_image.png)

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

## Datasets and models
Datasets and models associated with each section of the associated publication are in the data/ directory in this repository
