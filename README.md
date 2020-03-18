Official code for the paper ["Deep Contextualized Self-training for Low Resource Dependency Parsing"](https://www.mitpressjournals.org/doi/pdf/10.1162/tacl_a_00294).\
If you use this code please cite our paper.

## Requirements
Simply run:

* Python 3.7 
* Pytorch 1.1.0 
* Cuda 10.0 

```
pip install -r requirements.txt
```
## Data
Preprocessed in `note` format. Data folder can be obtained from [here](https://drive.google.com/drive/folders/15z28d-boFhhZMdriJZY4tNcZeHiL-naW?usp=sharing).

## Multilingual Word Embeddings
Set the `word_path="./data/morph.word.200.vec"` and `char_path="./data/morph.char.30.vec"`
Embeddings can be found [here](https://drive.google.com/drive/folders/15z28d-boFhhZMdriJZY4tNcZeHiL-naW?usp=sharing)
Possible word embedding option: ['random', 'fasttext'] \
The multilingual word embedding (.vec extensions) should be placed under the `data/multilingual_word_embeddings` folder.


## Low Resource In-domain Experiments
In order to run the low resource in-domain experiments there are three steps we need to follow:
1. Running the base Biaffine parser
2. Running the sequence tagger(s)
3. Running the combined DCST parser

## Running code
Create `saved_model` empty folder to store the new models.
If you want to run complete model then simply run bash script `run_dcsh.sh` otherwise
Refer to corrsoponding section in `run_dcsh.sh` to run corrsopnding segments.

## Input settings:
1. Without POS Tag       : Don't use flag `--use_pos` for all stages, namely, base model, auxiliary tasks, Final ensembled model.
2. With Coarse level Tag : Use the input files from `data` folder from `--use_pos` flag [here](https://drive.google.com/drive/folders/15z28d-boFhhZMdriJZY4tNcZeHiL-naW?usp=sharing) 
3. With POS level Tag    : Shuffle 2nd and 3rd column of all the files in  `data` folder.

## Mode:
Add pretrained model file from [Here](https://drive.google.com/drive/folders/15z28d-boFhhZMdriJZY4tNcZeHiL-naW?usp=sharing) to path `./utils/morph_tagger/cwlm_lstm_crf_cas_2.model`
1. Pretrained morph tagger layer : Run the `run_dcst.sh`. 
2. Pretrained morph tagger layer with freezing: comment [this](https://github.com/Jivnesh/DCST_with_pretrained_layers/blob/cf22743acfb03b10cb9ef4c67eca1b8faa753039/examples/GraphParser.py#L311) and then run `run_dcst.sh`


