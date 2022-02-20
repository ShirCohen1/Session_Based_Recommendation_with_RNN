
def get_hyperparams():
  hypers = dict(
        hidden_size = 100, #1000
        num_layers = 1, #3
        batch_size = 50,
        dropout_input = 0,
        dropout_hidden = 0, #0.5
        n_epochs = 5,
        k_eval = 20,
        optimizer_type = 'Adagrad',
        final_act = 'tanh',
        lr = 0.01,
        weight_decay = 0,
        momentum = 0,
        eps = 1e-6,
        seed = 22,
        sigma = -1,
        embedding_dim = -1,
        loss_type = 'TOP1-max',
        data_folder = 'dataset/',
        train_data = 'recSys15TrainOnly.txt',
        valid_data = 'recSys15Valid.txt',

        )

  return hypers

