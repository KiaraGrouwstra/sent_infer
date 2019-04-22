from model_utils import *

def test_get_encoder():
    assert get_encoder('baseline')().forward

# def test_eval_dataset():
#     enc_type = 'lstm'
#     model = get_model(enc_type)
#     # assert eval_dataset(model, dataset, 1, text_embeds)

def test_get_optimizer():
    optim = 'adam'
    lr = 1e-5
    weight_decay = 1e-3
    enc_type = 'lstm'
    model = get_model(enc_type)
    pars = model.parameters()
    optim_pars = {'params': pars, 'lr': lr, 'weight_decay': weight_decay}
    optimizer = get_optimizer(optim, model.parameters())
    assert optimizer

def test_get_model():
    assert get_model('lstm').forward
