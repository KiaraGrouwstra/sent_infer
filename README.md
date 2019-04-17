# UvA Statistical Methods for Natural Language Semantics practical: Sent Infer

### Intro

This is mostly an implementation of a paper by Facebook:
[Supervised Learning of Universal Sentence Representations from 
Natural Language Inference Data](https://www.arxiv-vanity.com/papers/1705.02364/)

### Assignment description
https://cl-illc.github.io/semantics/resources/practicals/practical1/smnls_practical.pdf

### Usage

```bash
./get_data.sh
conda env create -n sent -f environment.yml
conda env update -n sent -f environment.yml
source activate sent
pip install git+https://github.com/PetrochukM/PyTorch-NLP.git
python src/foo.py
```

### Overview

- [ ] get data
    - [ ] SNLI
    - [ ] GloVe
- [ ] preprocess data:
    - [ ] https://github.com/brmson/dataset-sts/tree/master/data/rte/snli
    - [ ] tokenize
    - [ ] filter GloVe to SNLI/EvalSent
    - [ ] NLTK Penn treebank tokenizer
- [ ] combine data
- [ ] NLI architecture
- [ ] models
    - [ ] glove baseline
    - [ ] uni-directional LSTM
    - [ ] bi-directional LSTM
    - [ ] bi-directional LSTM + max pooling
- [ ] hyperparams: Conneau 3.3
    - [ ] SGD
    - [ ] learning rate 0.1
    - [ ] weight decay 0.99
    - [ ] at each epoch, we divide the learning rate by 5 if the dev accuracy decreases
    - [ ] use mini-batches of size 64
    - [ ] training is stopped when the learning rate goes under the threshold of 10e-5
    - [ ] classifier
        - [ ] MLP
        - [ ] 1 hidden layer of 512 hidden units
- [ ] evaluation
    - [ ] SNLI macro metric: average of dev accuracies
    - [ ] SNLI micro metric: sum of the dev accuracies, weighted by the number of dev samples
    - [ ] SentEval: https://uva-slpl.github.io/ull/resources/practicals/practical3/senteval_example.ipynb
- [ ] visualization
    - tensorboard: https://github.com/lanpa/tensorboardX
