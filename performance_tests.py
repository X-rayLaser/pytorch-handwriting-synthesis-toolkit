import time
import models
import torch


def get_running_time(model, *inputs):
    y_hat = model(*inputs)

    t0 = time.time()
    trials = 20
    for _ in range(trials):
        y_hat = model(*inputs)

    return (time.time() - t0) / trials


input_size = 64
hidden_size = 128
steps = 20


def test_lstm():
    print('LSTM test')
    x = torch.Tensor(1, steps, input_size)
    lstm = models.PeepholeLSTM(input_size, hidden_size)

    h0 = torch.Tensor(1, hidden_size)
    c0 = torch.Tensor(1, hidden_size)
    states = (h0, c0)
    run_time = get_running_time(lstm, x, states)
    print(f'Took on average {run_time} seconds per 1 pass')


def test_batchless_lstm():
    print('Batchless LSTM test')
    x = torch.Tensor(steps, input_size)
    lstm = models.BatchLessLSTM(input_size, hidden_size)

    h0 = torch.Tensor(hidden_size)
    c0 = torch.Tensor(hidden_size)
    states = (h0, c0)
    run_time = get_running_time(lstm, x, states)
    print(f'Took on average {run_time} seconds per 1 pass')


def test_batchless_mixture():
    print('Batchless Mixture LSTM test')
    x = torch.Tensor(steps, input_size)
    lstm = models.BatchLessMixture(input_size, hidden_size)
    run_time = get_running_time(lstm, x)
    print(f'Took on average {run_time} seconds per 1 pass')


def test_synthesis_network():
    print('Synthesis network test')
    alphabet_size = 57
    num_chars = 80
    device = torch.device("cpu")
    model = models.SynthesisNetwork(input_size, hidden_size, alphabet_size, device)

    x = torch.Tensor(1, steps, input_size)
    c = torch.Tensor(1, num_chars, alphabet_size)

    run_time = get_running_time(model, x, c)
    print(f'Took on average {run_time} seconds per 1 pass')


test_lstm()
test_batchless_lstm()
test_batchless_mixture()
test_synthesis_network()
