import pickle
from model_utils import *

def test_train():
    model_type = 'baseline'
    lr = 0.001
    weight_decay = 0.0
    optim = 'adam'
    (model, optimizer) = make_model(model_type, lr, weight_decay, optim)
    with open('batch_ex.pkl', 'rb') as f:
        batch = pickle.load(f)
    (prem_embeds, prem_lens, hyp_embeds, hyp_lens, labels) = batch_cols(batch)
    predictions = model.forward(prem_embeds, prem_lens, hyp_embeds, hyp_lens)
    train_loss = loss_fn(predictions, labels)
    train_acc = accuracy(predictions, labels)
