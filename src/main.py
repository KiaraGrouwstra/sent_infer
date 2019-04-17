import pickle
import itertools
from functools import reduce
# import torch
# import torch.nn
# import torchnlp
import torch.optim
from torchnlp.datasets import snli_dataset
from torchnlp.word_to_vector import GloVe
from nltk.tokenize import TreebankWordTokenizer

def tokenize(snli_set):
    tokenizer = TreebankWordTokenizer()
    token_set = set()
    for item in snli_set:
        for k in ['premise', 'hypothesis']:
            for token in tokenizer.tokenize(item[k]):
                token_set.add(token)
    return token_set

def get_embeddings(words):
    vectors = GloVe()
    glove_filtered = { k: vectors[k] for k in words if k in vectors }
    return glove_filtered

def main():
    snli_dir = 'data/snli_1.0/'

    train = snli_dataset(snli_dir, train=True)
    dev   = snli_dataset(snli_dir, dev=True)
    test  = snli_dataset(snli_dir, test=True)
    snli = {
        'train': train,
        'dev': dev,
        'test': test,
    }

    # words = tokenize(test)  # all
    words = reduce(lambda x, y: {*x, *y}, map(tokenize, snli.items()))
    embeddings = get_embeddings(words)

    n_inputs = 300  # glove embedding dimensions
    # encoder = ???
    model = Model(n_inputs, encoder)
    lr = 0.1
    weight_decay = 0.99
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    batch_size = 64
    prev_acc = 0.0

    for step in range(FLAGS.max_steps):
        optimizer.zero_grad()

        # batch
        # x_train_np, y_train_np = cifar10['train'].next_batch(batch_size)
        # x_train_flat = x_train_np.reshape((batch_size, n_inputs))
        x_train_torch = torch.from_numpy(x_train_flat)     .to(device)
        y_train_torch = torch.from_numpy(y_train_np).long().to(device)
        idx_train = torch.argmax(y_train_torch, dim=-1).long()

        # results
        train_predictions = model.forward(x_train_torch)
        train_loss = ce(train_predictions, idx_train)
        train_acc = accuracy(train_predictions, idx_train)

        # evaluate
        if step % FLAGS.eval_freq == 0:
            # test_predictions = model.forward(x_test_torch)
            test_loss = ce(test_predictions, idx_test)
            test_acc = accuracy(test_predictions, idx_test)
            metrics = [train_acc, test_acc, train_loss, test_loss]
            vals = [optim, *[val.detach().cpu().numpy().take(0) for val in metrics]]
            stats = dict(zip(cols, vals))
            print(stats)
            results.append(stats)

        # training is stopped when the learning rate goes under the threshold of 10e-5
        threshold = 1e-5
        if lr < threshold:
            break
        
        # at each epoch, we divide the learning rate by 5 if the dev accuracy decreases
        if dev_acc > prev_acc:
            lr /= 5
        prev_acc = dev_acc

        train_loss.backward()
        optimizer.step()



if __name__ == '__main__':
    main()
