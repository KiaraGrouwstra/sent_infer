import os
import argparse
from utils import *
from operator import itemgetter
from model_utils import make_model
from sent_eval import sent_eval, senteval_metrics
from tensorboardX import SummaryWriter

def parse_flags():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type = str, default = 'mean', help='model, baseline (default), lstm, bilstm, maxlstm')
    parser.add_argument('--checkpoint_path', type = str, default = 'checkpoint_folder/mean.pth', help='model, default checkpoint_folder/mean.pth')
    parser.add_argument('--eval_data_path', type = str, default = '../SentEval/data', help='eval data path, default ../SentEval/data')
    flags, unparsed = parser.parse_known_args()
    flag_keys = [
        'model_type',
        'checkpoint_path',
        'eval_data_path',
    ]
    return itemgetter(*flag_keys)(vars(flags))

def senteval(checkpoint_path, eval_data_path):
    batch_size = 64
    optim = 'adam'
    lr = 0.001
    weight_decay = 0.0
    (model, optimizer) = make_model(model_type, lr, weight_decay, optim, batch_size)
    checkpoint = torch.load(checkpoint_path)
    for k, v in {
        'model':      model,
        'optimizer':  optimizer,
        'classifier': model.classifier,
        'encoder':    model.encoder,
    }.items():
        v.load_state_dict(checkpoint[k])
    # print('model', model)
    pars = model.parameters()

    # https://uva-slpl.github.io/ull/resources/practicals/practical3/senteval_example.ipynb
    results = sent_eval(eval_data_path, model.encoder)
    with SummaryWriter('senteval') as w:
        for task, result in results.items():
            print(task, result)
            # w.add_scalars(os.path.join('tasks', task), result)
        # (micro, macro) = senteval_metrics(results)
        # metrics = {
        #     'micro': micro,
        #     'macro': macro,
        # }
        # print(metrics)
        # w.add_scalars('senteval/metrics', metrics)

if __name__ == '__main__':
    (
        model_type,
        checkpoint_path,
        eval_data_path,
    ) = parse_flags()
    senteval(checkpoint_path, eval_data_path)
