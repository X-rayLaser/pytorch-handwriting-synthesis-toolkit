import unittest
import torch
import shutil
import os
import numpy as np
import training
import models
import lstm_test_utils


class MovingAverageTests(unittest.TestCase):
    def test_after_initialization(self):
        ma = training.MovingAverage()
        self.assertEqual(ma.value, 0)

    def test_after_single_update(self):
        ma = training.MovingAverage()
        ma.update(25)
        self.assertEqual(ma.value, 25)

    def test_after_few_updates(self):
        ma = training.MovingAverage()
        ma.update(3)
        ma.update(5)
        ma.update(1)
        self.assertEqual(ma.value, 3)

    def test_reset_erases_previous_value(self):
        ma = training.MovingAverage()
        ma.update(3)
        ma.update(5)
        ma.reset()
        self.assertEqual(ma.value, 0)


class MetricTests(unittest.TestCase):
    def test_mse_name(self):
        self.assertEqual(training.MSE().name, 'MSE')

    def test_mse_is_zero_after_initialization(self):
        metric = training.MSE()
        self.assertEqual(metric.value, 0)

    def test_mse_on_a_single_example(self):
        metric = training.MSE()

        y_hat = torch.tensor([[2., 3.]])
        ground_true = torch.tensor([[5., 7.]])
        metric.update(y_hat, ground_true)
        # 12.5 = (5 - 2)**2 + (7 - 3) ** 2
        self.assertEqual(metric.value, 12.5)

    def test_mse_on_a_batch(self):
        metric = training.MSE()

        y_hat = torch.tensor([[2., 3.], [1., 1.]])
        ground_true = torch.tensor([[5., 7.], [2., 2.]])
        metric.update(y_hat, ground_true)
        self.assertEqual(27 / 4, metric.value)


class TrainingLoopTests(unittest.TestCase):
    def test_loop_prints_epochs(self):
        ds = [(i, i**2) for i in range(3)]
        loop = training.TrainingLoop(ds, batch_size=1)

        output_device = training.InMemoryDevice()
        loop.set_output_device(output_device)
        loop.start(epochs=2)
        self.assertIn(f'Epoch {0:4} finished', output_device.lines[0])
        self.assertIn(f'Epoch {1:4} finished', output_device.lines[1])

    def test_loop_prints_epoch_progress(self):
        ds = [(i, i**2) for i in range(2)]
        loop = training.TrainingLoop(ds, batch_size=1)

        output_device = training.InMemoryDevice()
        loop.set_output_device(output_device)

        it = loop.get_iterator(epochs=1)

        next(it)
        self.assertIn(f'Epoch {0:4} 1/2 batches', output_device.lines[0])

        next(it)
        self.assertIn(f'Epoch {0:4} 2/2 batches', output_device.lines[0])

        next(it)
        self.assertIn(f'Epoch {0:4} finished', output_device.lines[0])

    def test_loop_prints_metrics_after_every_epoch(self):
        ds = [(i, i**2) for i in range(2)]
        loop = training.TrainingLoop(ds, batch_size=1)

        dummy_loss = 23.32
        task = training.DummyTask(dummy_loss)

        output_device = training.InMemoryDevice()
        loop.set_output_device(output_device)
        loop.set_training_task(task)
        loop.start(epochs=2)

        expected = [
            f"\rEpoch {0:4} finished. Loss   23.32.",
            f"\rEpoch {0:4} finished. Loss   23.32."
        ]
        self.assertIn(expected[0], output_device.lines[0])

    def test_callback_is_invoked(self):
        ds = [(0, 0), (1, 1)]
        loop = training.TrainingLoop(ds, batch_size=1)

        on_iteration_args = []
        on_epoch_args = []

        class Foo(training.Callback):
            def on_iteration(self, epoch, epoch_iteration, iteration):
                on_iteration_args.append((epoch, epoch_iteration, iteration))

            def on_epoch(self, epoch):
                on_epoch_args.append(epoch)

        loop.add_callback(Foo())
        loop.start(epochs=2)

        self.assertEqual([(0, 0, 0), (0, 1, 1), (1, 0, 2), (1, 1, 3)], on_iteration_args)
        self.assertEqual([0, 1], on_epoch_args)


class CallbackTests(unittest.TestCase):
    def setUp(self):
        self._save_dir = 'test_model_checkpoints'
        if os.path.isdir(self._save_dir):
            shutil.rmtree(self._save_dir)

    def tearDown(self):
        if os.path.isdir(self._save_dir):
            shutil.rmtree(self._save_dir)

    def test_callback_will_not_save_on_iteration(self):
        model = torch.nn.Sequential(torch.nn.Linear(1, 1))

        checkpoint = training.EpochModelCheckpoint(model, self._save_dir, save_interval=1)

        checkpoint.on_iteration(epoch=0, epoch_iteration=10, iteration=25)
        self.assertEqual(len(list(os.listdir(self._save_dir))), 0)

    def test_callback_will_not_save_on_epoch(self):
        model = torch.nn.Sequential(torch.nn.Linear(1, 1))

        checkpoint = training.IterationModelCheckpoint(model, self._save_dir, save_interval=1)

        checkpoint.on_epoch(epoch=4)
        self.assertEqual(len(list(os.listdir(self._save_dir))), 0)

    def test_epoch_checkpoint_will_save_model_on_epoch_end_at_right_intervals(self):
        model = torch.nn.Sequential(torch.nn.Linear(1, 1))
        checkpoint = training.EpochModelCheckpoint(model, self._save_dir, save_interval=3)
        checkpoint.on_epoch(0)
        save_path = os.path.join(self._save_dir, f'model_at_epoch_{0}.pt')
        self.assertFalse(os.path.exists(save_path))

        epoch = 2
        checkpoint.on_epoch(epoch)
        save_path = os.path.join(self._save_dir, f'model_at_epoch_{3}.pt')
        self.assertTrue(os.path.isfile(save_path))

        checkpoint.on_epoch(5)
        save_path = os.path.join(self._save_dir, f'model_at_epoch_{6}.pt')
        self.assertTrue(os.path.isfile(save_path))

    def test_iteration_checkpoint_will_save_model_periodically_after_every_few_iterations(self):
        #self.fail('Finish it')
        pass


class FormulasTests(unittest.TestCase):
    def test_gate_formula_on_1_d_input(self):
        w_x_value = 2.
        w_h_value = 3.
        w_c_value = 4.
        b_value = 5.
        x_value = 10.

        h_value = 0.1
        c_value = 0.5

        w_x = torch.tensor([[w_x_value]])
        w_h = torch.tensor([[w_h_value]])
        w_c = torch.tensor([[w_c_value]])
        b = torch.tensor([b_value])
        gate = lstm_test_utils.Gate(w_x, w_h, w_c, b)

        x = torch.tensor([x_value]).unsqueeze(0)
        h0 = torch.tensor([h_value]).unsqueeze(0)
        c0 = torch.tensor([c_value]).unsqueeze(0)
        state = (h0, c0)

        y_hat = gate(x, state)

        expected = torch.sigmoid(
            torch.tensor([w_x_value * x_value + w_h_value * h_value + w_c_value * c_value + b_value])
        )

        y_hat = y_hat.detach().numpy()
        expected = expected.numpy()
        self.assertTrue(np.allclose(y_hat, expected))

    def test_input_gate_factory_method(self):
        input_size = 5
        hidden_size = 10
        gate = lstm_test_utils.Gate.random_gate(input_size, hidden_size)
        self.assertTupleEqual(gate.w_x.shape, (input_size, hidden_size))
        self.assertTupleEqual(gate.w_h.shape, (hidden_size, hidden_size))
        self.assertTupleEqual(gate.w_c.shape, (hidden_size,))
        self.assertTupleEqual(gate.b.shape, (hidden_size,))

    def test_input_gate_on_general_input(self):
        batch_size = 3
        input_size = 5
        hidden_size = 10
        gate = lstm_test_utils.Gate.random_gate(input_size, hidden_size)

        x = torch.rand(batch_size, input_size)
        h = torch.rand(batch_size, hidden_size)
        c = torch.rand(batch_size, hidden_size)
        state = (h, c)
        y_hat = gate(x, state)
        y_hat = y_hat.detach().numpy()

        expected = torch.sigmoid(x @ gate.w_x + h @ gate.w_h + c * gate.w_c + gate.b)
        expected = expected.detach().numpy()

        self.assertTupleEqual(y_hat.shape, (batch_size, hidden_size))
        self.assertTupleEqual(expected.shape, y_hat.shape)
        self.assertTrue(np.allclose(expected, y_hat))

    def test_cell_formula_factory_method(self):
        input_size = 5
        hidden_size = 10
        cell_formula = lstm_test_utils.CellFormula.new_formula(input_size, hidden_size)
        self.assertTupleEqual(cell_formula.w_x.shape, (input_size, hidden_size))
        self.assertTupleEqual(cell_formula.w_h.shape, (hidden_size, hidden_size))
        self.assertTupleEqual(cell_formula.b.shape, (hidden_size,))

    def test_cell_formula_on_general_input(self):
        batch_size = 3
        input_size = 5
        hidden_size = 10
        formula = lstm_test_utils.CellFormula.new_formula(input_size, hidden_size)

        x = torch.rand(batch_size, input_size)
        f = torch.rand(batch_size, hidden_size)
        i = torch.rand(batch_size, hidden_size)

        h = torch.rand(batch_size, hidden_size)
        c = torch.rand(batch_size, hidden_size)
        state = (h, c)
        y_hat = formula(x, i, f, state)
        y_hat = y_hat.detach().numpy()

        expected = f * c + i * torch.tanh(x @ formula.w_x + h @ formula.w_h + formula.b)
        expected = expected.detach().numpy()

        self.assertTupleEqual(y_hat.shape, (batch_size, hidden_size))
        self.assertTupleEqual(expected.shape, y_hat.shape)
        self.assertTrue(np.allclose(expected, y_hat))

    def test_output_formula(self):
        formula = lstm_test_utils.OutputFormula()
        o = torch.tensor([[2., 1.], [3., 4.]])
        c = torch.tensor([[1., 0.5], [1., 1.]])

        expected = o * np.tanh(c)

        y_hat = formula(o, c)
        y_hat = y_hat.detach().numpy()

        self.assertTrue(np.allclose(y_hat, expected))

    def test_output_formula_with_batch_size_1(self):
        formula = lstm_test_utils.OutputFormula()
        o = torch.tensor([[2., 1.]])
        c = torch.tensor([[1., 0.5]])

        first = o[0, 0] * torch.tanh(c[0, 0])
        second = o[0, 1] * torch.tanh(c[0, 1])
        expected = np.array([[first, second]])

        y_hat = formula(o, c)
        y_hat = y_hat.detach().numpy()

        self.assertTrue(np.allclose(y_hat, expected))


class LstmCellTests(unittest.TestCase):
    def test(self):
        batch_size = 3
        input_size = 5
        hidden_size = 10

        cell = lstm_test_utils.LSTMCell.random_cell(input_size, hidden_size)

        x = torch.rand(batch_size, input_size)
        h = torch.rand(batch_size, hidden_size)
        c = torch.rand(batch_size, hidden_size)
        state = (h, c)

        y_hat, (new_h, new_c) = cell(x, state)
        y_hat = y_hat.detach().numpy()
        new_h = new_h.detach().numpy()
        new_c = new_c.detach().numpy()

        i = cell.input_gate(x, state)
        f = cell.forget_gate(x, state)
        c = cell.cell_formula(x, i, f, state)
        o = cell.output_gate(x, (h, c))
        expected = cell.output_formula(o, c)
        expected = expected.detach().numpy()

        self.assertTrue(np.allclose(expected, y_hat))

        self.assertTrue(np.allclose(new_h, expected))
        self.assertTrue(np.allclose(new_c, c))
        # todo: sometimes nans are produced

    def test_numerical_stability(self):
        self.fail('Finish it')


class LSTMTests(unittest.TestCase):
    def test_get_initial_state(self):
        batch_size = 8
        input_size = 3
        hidden_size = 32
        lstm = models.PeepholeLSTM(input_size, hidden_size)

        h0, c0 = lstm.get_initial_state(batch_size)

        expected = np.zeros((batch_size, hidden_size))
        self.assertTrue(np.allclose(h0.numpy(), expected))
        self.assertTrue(np.allclose(c0.numpy(), expected))

    def test_outputs_correct_shape(self):
        batch_size = 8
        steps = 10
        input_size = 3
        hidden_size = 32

        lstm = models.PeepholeLSTM(input_size, hidden_size)
        x = torch.Tensor(batch_size, steps, input_size)
        h0 = torch.zeros(batch_size, hidden_size)
        c0 = torch.zeros_like(h0)
        y_hat, states = lstm(x, (h0, c0))
        self.assertEqual(y_hat.shape, (batch_size, steps, hidden_size))

        h, c = states
        expected_shape = (batch_size, hidden_size)
        self.assertTupleEqual(h.shape, expected_shape)
        self.assertTupleEqual(c.shape, expected_shape)

    def test_single_step(self):
        batch_size = 1
        input_size = 1
        hidden_size = 1

        x_value = 2.
        x = torch.tensor([
            [[x_value]]
        ])

        lstm = models.PeepholeLSTM(input_size, hidden_size)

        h0_value = 22.
        c0_value = 3.
        h0 = torch.tensor([[h0_value]])
        c0 = torch.tensor([[c0_value]])

        weight = 1
        bias = weight
        lstm.set_weights(weight)
        i = torch.sigmoid(
            torch.tensor([weight * x_value + weight * h0_value + weight * c0_value + bias])
        )

        f = torch.sigmoid(
            torch.tensor([weight * x_value + weight * h0_value + weight * c0_value + bias])
        )

        c = f * c0 + i * torch.tanh(
            torch.tensor([weight * x_value + weight * h0_value + bias])
        )

        o = torch.sigmoid(
            torch.tensor([weight * x_value + weight * h0_value + weight * c0_value + bias])
        )

        expected_output = o * torch.tanh(c)
        expected_output = expected_output.numpy()

        expected_h = expected_output
        expected_c = c.numpy()

        output, states = lstm(x, states=(h0, c0))
        output = output.squeeze(0).detach().numpy()

        h, c = states
        h, c = h.squeeze(0), c.squeeze(0)
        h, c = h.detach().numpy(), c.detach().numpy()

        print(output)
        print(expected_output)
        self.assertTrue(np.allclose(output, expected_output))
        self.assertTrue(np.allclose(h, expected_h))
        self.assertTrue(np.allclose(c, expected_c))

    def test_single_step_for_general_case(self):
        batch_size = 3
        input_size = 5
        hidden_size = 10

        lstm = models.PeepholeLSTM(input_size, hidden_size)
        x = torch.rand(batch_size, 1, input_size)
        state = lstm.get_initial_state(batch_size)

        lstm_cell = self.get_lstm_cell(input_size, lstm)

        expected, (expected_h, expected_c) = lstm_cell(x.squeeze(1), state)
        expected = expected.unsqueeze(1).detach().numpy()
        expected_h = expected_h.detach().numpy()
        expected_c = expected_c.detach().numpy()

        y_hat, (h, c) = lstm(x, states=state)
        y_hat = y_hat.detach().numpy()
        h = h.detach().numpy()
        c = c.detach().numpy()

        self.assertTrue(np.allclose(expected, y_hat))
        self.assertTrue(np.allclose(expected_h, h))
        self.assertTrue(np.allclose(expected_c, c))

    def test_by_feeding_batch_of_sequences(self):
        batch_size = 3
        steps = 20
        input_size = 5
        hidden_size = 10

        lstm = models.PeepholeLSTM(input_size, hidden_size)
        x = torch.rand(batch_size, steps, input_size)
        state = lstm.get_initial_state(batch_size)

        lstm_cell = self.get_lstm_cell(input_size, lstm)
        slow_lstm = lstm_test_utils.SlowPeepholeLstm(lstm_cell)

        y_hat, (h, c) = lstm(x, state)
        expected_y, (expected_h, expected_c) = slow_lstm(x, state)

        y_hat = y_hat.detach().numpy()
        h = h.detach().numpy()
        c = c.detach().numpy()

        expected_y = expected_y.detach().numpy()
        expected_h = expected_h.detach().numpy()
        expected_c = expected_c.detach().numpy()

        print(y_hat, '\n', expected_y)

        self.assertTrue(np.allclose(y_hat, expected_y))
        self.assertTrue(np.allclose(h, expected_h))
        self.assertTrue(np.allclose(c, expected_c))

    def get_lstm_cell(self, input_size, lstm):
        W_x = lstm.W_xh[:input_size]
        W_h = lstm.W_xh[input_size:]

        W_xi, W_xf, W_xc, W_xo = W_x.chunk(4, 1)
        W_hi, W_hf, W_hc, W_ho = W_h.chunk(4, 1)

        input_gate = lstm_test_utils.Gate(W_xi, W_hi, lstm.W_ci, lstm.b_i)
        forget_gate = lstm_test_utils.Gate(W_xf, W_hf, lstm.W_cf, lstm.b_f)
        output_gate = lstm_test_utils.Gate(W_xo, W_ho, lstm.W_co, lstm.b_o)

        cell_formula = lstm_test_utils.CellFormula(W_xc, W_hc, lstm.b_c)
        return lstm_test_utils.LSTMCell(input_gate, forget_gate, cell_formula, output_gate)


if __name__ == '__main__':
    unittest.main()


# todo: test save and load model weights
# todo: metrics
# todo: loss
# todo: visualization (PDF heatmap, window)
# todo: dataset
# todo: model
