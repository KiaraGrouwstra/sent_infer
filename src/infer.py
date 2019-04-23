from operator import itemgetter
import argparse
from utils import *
from model_utils import *
from nltk.tokenize import TreebankWordTokenizer
from pdb import set_trace
from data import get_data
tokenizer = TreebankWordTokenizer().tokenize

# constants
GLOVE_DIMS = 300  # glove embedding dimensions

def parse_flags():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_type', type = str, default = 'mean', help='model, baseline (default), lstm, bilstm, maxlstm')
    parser.add_argument('checkpoint_path', type = str, default = 'checkpoint_folder/mean.pth', help='model, default checkpoint_folder/mean.pth')
    flags, unparsed = parser.parse_known_args()
    return flags

def infer():
    flags = parse_flags()
    flag_keys = ['checkpoint_path', 'model_type']
    (checkpoint_path, model_type) = itemgetter(*flag_keys)(vars(flags))

    # dummy values to load in checkpoint
    batch_size = 64
    optim = 'adam'
    lr = 0.001
    weight_decay = 0.0

    (model, optimizer) = make_model(model_type, lr, weight_decay, optim, batch_size)
    checkpoint = torch.load(checkpoint_path)
    state_parts = {
        'model':      model,
        'optimizer':  optimizer,
        'classifier': model.classifier,
        'encoder':    model.encoder,
    }
    for k, v in state_parts.items():
        v.load_state_dict(checkpoint[k])

    pars = model.parameters()
    (snli, text_field, label_vocab, text_embeds) = get_data()
    text_stoi = text_field.vocab.stoi

    while True:
        premise    = input('Please input your premise:\n')
        hypothesis = input('Please input your hypothesis:\n')

        premise_tokens = tokenizer(premise)
        hypothesis_tokens = tokenizer(hypothesis)

        premise_idxs    = torch.LongTensor([text_stoi[token] for token in premise_tokens])
        hypothesis_idxs = torch.LongTensor([text_stoi[token] for token in hypothesis_tokens])

        hypo_len = len(hypothesis_tokens)
        prem_len = len(premise_tokens)
        hypo_lens =  torch.LongTensor([hypo_len])
        prem_lens = torch.LongTensor([prem_len])
        prem_embeds = text_embeds(premise_idxs)
        hypo_embeds  = text_embeds(hypothesis_idxs)
        prem_embeds = prem_embeds.view((prem_len, 1, GLOVE_DIMS))
        hypo_embeds = hypo_embeds.view((hypo_len, 1, GLOVE_DIMS))

        predictions = model.forward(prem_embeds, prem_lens, hypo_embeds, hypo_lens)
        prediction_oh = predictions[0]
        prediction_idx = prediction_oh.argmax().item()
        prediction_str = label_vocab.itos[prediction_idx]

        print({
            'contradiction': 'CONTRADICTION!',
            'neutral': 'How is that related?',
            'entailment': 'Well, obviously.',
        }[prediction_str] + '\n')

if __name__ == '__main__':
    infer()
