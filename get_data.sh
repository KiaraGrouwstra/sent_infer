# snli
wget https://nlp.stanford.edu/projects/snli/snli_1.0.zip
unzip snli_1.0.zip -d data/
rm snli_1.0.zip

# senteval
git clone https://github.com/facebookresearch/SentEval.git
cd SentEval/
python setup.py install
    # datasets
    cd data/downstream/
    ./get_transfer_data.bash
    cd ../..
    # glove
    mkdir pretrained
    cd pretrained
    wget http://nlp.stanford.edu/data/glove.840B.300d.zip
    cd ..
cd ..
