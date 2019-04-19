# UvA Statistical Methods for Natural Language Semantics practical: Sent Infer

### Intro

This repo holds code for the [assignment](https://cl-illc.github.io/semantics/resources/practicals/practical1/smnls_practical.pdf) of UvA course [Statistical Methods for Natural Language Semantics](https://cl-illc.github.io/semantics/).
This is mostly an implementation of a paper by Facebook:
[Supervised Learning of Universal Sentence Representations from 
Natural Language Inference Data](https://www.arxiv-vanity.com/papers/1705.02364/)

### Usage

```bash
# download senteval
./get_data.sh
# install packages
conda env create -n sent -f environment.yml
conda env update -n sent -f environment.yml
source activate sent
pip install -r requirements.txt
# training
python src/train.py <model_type> <model_name> <checkpoint_path> <train_data_path>
# evaluation
python src/eval.py <checkpoint_path> <eval_data_path>
# inference
python src/infer.py <checkpoint_path> <input_file> <output_file>
```

### Overview

- [x] get data
    - [x] SNLI
    - [x] GloVe
- [x] preprocess data:
    - [x] tokenize: NLTK Penn treebank tokenizer
    - [x] filter GloVe to SNLI/EvalSent
- [x] NLI architecture
- [x] models
    - [x] glove baseline
    - [x] uni-directional LSTM
    - [x] bi-directional LSTM
    - [x] bi-directional LSTM + max pooling
- [x] hyperparams: Conneau 3.3
    - [x] SGD
    - [x] learning rate 0.1
    - [x] weight decay 0.99
    - [x] at each epoch, we divide the learning rate by 5 if the dev accuracy decreases
    - [x] use mini-batches of size 64
    - [x] training is stopped when the learning rate goes under the threshold of 10e-5
    - [x] classifier
        - [x] MLP
        - [x] 1 hidden layer of 512 hidden units
- [x] evaluation
    - [x] SNLI macro metric: average of dev accuracies
    - [x] SNLI micro metric: sum of the dev accuracies, weighted by the number of dev samples
    - [ ] SentEval: https://uva-slpl.github.io/ull/resources/practicals/practical3/senteval_example.ipynb
- [x] visualization
    - tensorboard: https://github.com/lanpa/tensorboardX
