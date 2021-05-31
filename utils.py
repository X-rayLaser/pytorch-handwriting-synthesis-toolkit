import torch


class BatchAdapter:
    def prepare_inputs(self, batch):
        pass

    def prepare_loss_inputs(self, batch):
        pass


class PaddedSequencesBatch:
    def __init__(self, sequences, padding=0):
        """

        :param sequences: List[List[Tuple]]
        :param padding: value used to pad sequences to be of max_length length
        """
        if not sequences or not sequences[0]:
            raise BadInputError()

        self._seqs = sequences

        self._max_len = max([len(s) for s in sequences])

        self._inner_dim = len(sequences[0][0])

        self._batch_size = len(sequences)

        self._tensor = torch.ones(
            self._batch_size, self._max_len, self._inner_dim, dtype=torch.float32
        ) * padding

        mask = []

        for i in range(self._batch_size):
            seq_len = len(sequences[i])
            mask.extend([True] * seq_len)
            mask.extend([False] * (self._max_len - seq_len))
            for j in range(seq_len):
                self._tensor[i, j] = torch.tensor(sequences[i][j])

        self._mask = torch.tensor(mask, dtype=torch.bool)

    @property
    def max_length(self):
        return self._max_len

    @property
    def tensor(self):
        return self._tensor

    @property
    def mask(self):
        return self._mask

    def concatenated(self):
        """
        Concatenates all sequences together along dimension 1 while skipping padded values

        :return: Tensor of shape (total_num_steps, inner_dim)
        """
        return self.concatenate_batch(self.tensor)

    def concatenate_batch(self, batch):
        """
        Method that is similar to concatenate, but it works on passed batch

        :param batch: Tensor of self.tensor shape
        :return: Tensor of shape (total_num_steps, inner_dim)
        """

        if batch.shape != self.tensor.shape:
            raise BadInputError(
                f'Expected batch to be of shape {self.tensor.shape}. Got {batch.shape}'
            )

        t = batch.reshape(self._batch_size * self._max_len, self._inner_dim)
        return t[self.mask]


class BadInputError(Exception):
    pass
