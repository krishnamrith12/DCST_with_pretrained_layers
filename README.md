Official code for the paper ["Deep Contextualized Self-training for Low Resource Dependency Parsing"](https://www.mitpressjournals.org/doi/pdf/10.1162/tacl_a_00294).\
If you use this code please cite our paper.

## Requirements
Simply run:

* Python 3.7 \
* Pytorch 1.1.0 \
* Cuda 10.0 

```
pip install -r requirements.txt
```
## Data
Preprocessed in `note` format.

## Multilingual Word Embeddings
Embeddings can be found [here](https://drive.google.com/drive/folders/15z28d-boFhhZMdriJZY4tNcZeHiL-naW?usp=sharing)
Possible word embedding option: ['random', 'fasttext'] \
The multilingual word embedding (.vec extensions) should be placed under the `data/multilingual_word_embeddings` folder.


## Low Resource In-domain Experiments
In order to run the low resource in-domain experiments there are three steps we need to follow:
1. Running the base Biaffine parser
2. Running the sequence tagger(s)
3. Running the combined DCST parser

## Running code
If you want to run complete model then simply run bash script `run_dcsh.sh` otherwise
Refer to corrsoponding section in `run_dcsh.sh` to run corrsopnding segments.

## Input settings:
1. Without POS Tag       : Don't use flag `--use_pos` for all stages, namely, base model, auxiliary tasks, Final ensembled model.
2. With Coarse level Tag : Use the input files from `data` folder from [here](https://drive.google.com/drive/folders/15z28d-boFhhZMdriJZY4tNcZeHiL-naW?usp=sharing) 
3. With POS level Tag    : Shuffle 2nd and 3rd column of all the files in  `data` folder.

## Running the base Biaffine Parser
Note that to run BiAFF classifier on 500 training data set `--set_num_training_samples 500`. And if you want to train on complete trainind data remove this flag. 
Refer to corrsoponding section in `run_dcsh.sh`

## Running the Sequence Tagger
Once training the base parser, we can now run the Sequnece Tagger on any of the three proposed sequence tagging tasks in order to learn the syntactical contextualized word embeddings from the unlabeled data set. \
1. For Auxiliary task set tasks as : 'number_of_children' 'relative_pos_based' 'distance_from_the_root'
2. For Multitask setting set tasks : 'Multitask_case_predict' 'Multitask_POS_predict' 'Multitask_label_predict'

Refer to corrsoponding section in `run_dcsh.sh`

## Final step - Running the Combined DCST Parser
As a final step we can now run the DCST (ensemble) parser:

Refer to corrsoponding section in `run_dcsh.sh`

