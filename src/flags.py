def parse_flags():
    parser = argparse.ArgumentParser()
    for conf in [
        {'dest': 'learning_rate', 'type': float, 'required': False, 'default': 0.1, 'help': 'Learning rate'},
        {'dest': 'max_epochs', 'type': int, 'default': 500, 'help': 'Number of epochs to run trainer.'},
        {'dest': 'batch_size', 'type': int, 'default': 64, 'help': 'Batch size to run trainer.'},
        {'dest': 'eval_freq', 'type': int, 'default': 25, 'help': 'Frequency of evaluation on the test set'},
        {'dest': 'weight_decay', 'type': float, 'default': 1e-3, 'help': 'weight decay used in the optimizer, default 1e-3'},
        {'dest': 'learning_decay', 'type': int, 'default': 5, 'help': 'by what to divide the LR when accuracy improves, default 5'},
        {'dest': 'learning_threshold', 'type': int, 'default': 1e-5, 'help': 'at which learning rate to stop the experiment, default 1e-5'},
        {'dest': 'optimizer_type', 'type': str, 'default': 'SGD', 'help': 'optimizer, default SGD, also supports adam, adagrad, rmsprop, adadelta'},
        {'dest': 'encoder_type', 'type': str, 'default': 'baseline', 'help': 'encoder, default BoW baseline, also supports lstm, bilstm, maxlstm'},
        # {'dest': 'data_dir', 'type': str, 'default': 'results/', 'help': 'Directory for storing input data'},
        {'dest': 'auto', 'help': 'automagically optimize hyperparameters using an evolutionary algorithm'},
        # , 'action': 'store_true'
    ]:
        dest = conf['dest']
        parser._add_action(argparse.Action(**conf, option_strings=[f'--{dest}']))

    flags, unparsed = parser.parse_known_args()
    return flags

