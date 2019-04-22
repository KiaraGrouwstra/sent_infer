import argparse

def parse_flags():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint_path', type = str, default = 'mean.th', help='model, default mean.th')
    parser.add_argument('eval_data_path', type = str, default = 'eval.txt', help='eval data path, default eval.txt')

# def eval():
flags = parse_flags()
flag_keys = ['checkpoint_path', 'eval_data_path']
(checkpoint_path, eval_data_path) = itemgetter(*flag_keys)(vars(flags))

# TODO: cut out into eval/infer files, see README
(loss, acc) = eval_dataset(model, test, batch_size)
# vals = [optim, *[val.detach().cpu().numpy().take(0) for val in metrics]]
stats = {
    'optimizer':  optim,
    'test_acc':   acc,
    'test_loss':  loss,
    'learning_rate': lr,
}
print(yaml.dump({k: round(i, 3) if isinstance(i, float) else i for k, i in stats.items()}))
w.add_scalars('metrics', stats)  # TODO: , epoch?

# # SentEval: https://uva-slpl.github.io/ull/resources/practicals/practical3/senteval_example.ipynb
# results = sent_eval(data_path).items()
# for task, result in results:
#     w.add_scalars(f'tasks/{task}', result)  # , epoch?
# (micro, macro) = senteval_metrics(results)
# metrics = {
#     'micro': micro,
#     'macro': macro,
# }
# w.add_scalars('senteval/metrics', metrics)  # , epoch?

# w.export_scalars_to_json('./scalars.json')
return loss

# if __name__ == '__main__':
#     eval()
