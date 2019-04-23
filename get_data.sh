# senteval
# sudo apt-get install unzip
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
