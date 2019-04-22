import os
import argparse
from utils import *

def parse_flags():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type = str, default = 'mean.pth', help='model, default checkpoint_folder/mean.pth')
    parser.add_argument('--eval_data_path', type = str, default = 'snli/train.tsv', help='eval data path, default snli/eval.tsv')
    flags, unparsed = parser.parse_known_args()
    flag_keys = [
        'checkpoint_path',
        'eval_data_path',
    ]
    return itemgetter(*flag_keys)(vars(flags))

def eval(checkpoint_path):
    model = torch.load(checkpoint_path)

    # embeddings = get_embeds(text_field.vocab.vectors)
    (snli, text_field, label_vocab, text_embeds) = get_data()
    # , label_embeds

    (train, dev, test) = splits
    (loss, acc) = eval_dataset(model, test, batch_size)
    # vals = [optim, *[val.detach().cpu().numpy().take(0) for val in metrics]]
    stats = {
        # 'optimizer':  optim,
        'test_acc':   acc,
        'test_loss':  loss,
        # 'learning_rate': lr,
    }
    print(yaml.dump({k: round(i, 3) if isinstance(i, float) else i for k, i in stats.items()}))
    # w.add_scalars('metrics', stats)

def senteval(checkpoint_path, eval_data_path):
    # https://uva-slpl.github.io/ull/resources/practicals/practical3/senteval_example.ipynb
    model = torch.load(checkpoint_path)
    encoder = model.encoder
    results = sent_eval(eval_data_path, encoder).items()
    # for task, result in results:
    #     w.add_scalars(os.path.join('tasks', task), result)
    (micro, macro) = senteval_metrics(results)
    metrics = {
        'micro': micro,
        'macro': macro,
    }
    # w.add_scalars('senteval/metrics', metrics)

    # w.export_scalars_to_json('./scalars.json')
    # return loss

if __name__ == '__main__':
    (
        checkpoint_path,
        eval_data_path,
    ) = parse_flags()
    eval(checkpoint_path)
    senteval(checkpoint_path, eval_data_path)
