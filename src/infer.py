import argparse
from utils import *
from model_utils import *

def parse_flags():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type = str, default = 'checkpoint_folder/mean.pth', help='model, default checkpoint_folder/mean.pth')
    flags, unparsed = parser.parse_known_args()
    return flags

def infer():
    flags = parse_flags()
    flag_keys = ['checkpoint_path']
    (checkpoint_path) = itemgetter(*flag_keys)(vars(flags))

    # model = get_model(enc_type)
    model = torch.load(checkpoint_path)
    pars = model.parameters()

    # embeddings = get_embeds(text_field.vocab.vectors)
    # (snli, text_embeds, label_embeds) = get_data()
    (snli, text_field, label_vocab, text_embeds) = get_data()
    text_stoi = text_field.vocab.stoi
    # , label_embeds

    while True:
        premise    = input('Please input your premise:\n')
        print('\n')
        hypothesis = input('Please input your hypothesis:\n')
        print('\n')

        premise_idx    = text_stoi[premise]
        hypothesis_idx = text_stoi[hypothesis]

        prem_embeds = text_embeds(premise_idx)
        hyp_embeds  = text_embeds(hypothesis_idx)
        hyp_lens = [len(hypothesis)]
        prem_lens = [len(premise)]

        predictions = model.forward(prem_embeds, prem_lens, hyp_embeds, hyp_lens)
        prediction_oh = predictions[0]
        prediction_idx = prediction_oh.argmax().item()
        prediction_str = label_vocab.itos[prediction_idx]

        print({
            'contradiction': 'CONTRADICTION!',
            'neutral': 'How is that related?',
            'entailment': 'Well, obviously.',
        }[prediction_str])

if __name__ == '__main__':
    infer()
