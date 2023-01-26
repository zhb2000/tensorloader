import unittest
from collections import namedtuple

import torch

from tensorloader import TensorLoader


class TensorLoaderTest(unittest.TestCase):
    def test_no_shuffle(self):
        data = torch.tensor([0, 1, 2, 3, 4, 5, 6])

        def test_with_batch_size(batch_size: int):
            dataloader = TensorLoader(data, batch_size=batch_size)
            batches = list(dataloader)
            self.assertEqual(
                data.tolist(),
                torch.concat(batches).tolist()
            )

        test_with_batch_size(1)
        test_with_batch_size(3)
        test_with_batch_size(6)
        test_with_batch_size(7)

    def test_shuffle(self):
        data = [0, 1, 2, 3, 4, 5, 6]

        def test_with_batch_size(batch_size: int):
            dataloader = TensorLoader(torch.tensor(data), batch_size=batch_size, shuffle=True)
            while True:
                batches = list(dataloader)
                result = torch.concat(batches).tolist()
                if result != data:
                    break
            self.assertCountEqual(data, result)

        test_with_batch_size(1)
        test_with_batch_size(3)
        test_with_batch_size(6)
        test_with_batch_size(7)

    def test_no_shuffle_tuple(self):
        X = [0, 1, 2, 3, 4, 5, 6]
        Y = [6, 5, 4, 3, 2, 1, 0]

        def test_with_batch_size(batch_size: int):
            dataloader = TensorLoader((torch.tensor(X), torch.tensor(Y)), batch_size=batch_size)
            batches = list(dataloader)
            x_result = torch.concat([x for x, _ in batches]).tolist()
            y_result = torch.concat([y for _, y in batches]).tolist()
            self.assertEqual(X, x_result)
            self.assertEqual(Y, y_result)

        test_with_batch_size(1)
        test_with_batch_size(3)
        test_with_batch_size(6)
        test_with_batch_size(7)

    def test_shuffle_tuple(self):
        X = [0, 1, 2, 3, 4, 5, 6]
        Y = [6, 5, 4, 3, 2, 1, 0]

        def test_with_batch_size(batch_size: int):
            dataloader = TensorLoader(
                (torch.tensor(X), torch.tensor(Y)),
                batch_size=batch_size,
                shuffle=True
            )
            while True:
                batches = list(dataloader)
                x_result = torch.concat([x for x, _ in batches]).tolist()
                y_result = torch.concat([y for _, y in batches]).tolist()
                if x_result != X:
                    self.assertNotEqual(Y, y_result)
                    break
            self.assertCountEqual(list(zip(X, Y)), list(zip(x_result, y_result)))

        test_with_batch_size(1)
        test_with_batch_size(3)
        test_with_batch_size(6)
        test_with_batch_size(7)

    def test_tensor_type(self):
        data = torch.zeros(10, 5)
        for batch in TensorLoader(data):
            self.assertIsInstance(batch, torch.Tensor)

    def test_tuple_type(self):
        X = torch.zeros(10, 5)
        Y = torch.zeros(10)
        for batch in TensorLoader((X, Y)):
            self.assertIsInstance(batch, tuple)
            x, y = batch
            self.assertIsInstance(x, torch.Tensor)
            self.assertIsInstance(y, torch.Tensor)

    def test_namedtuple_type(self):
        X = torch.zeros(10, 5)
        Y = torch.zeros(10)
        Batch = namedtuple('Batch', ['x', 'y'])
        for batch in TensorLoader(Batch(X, Y), unpack_args=True):
            self.assertIsInstance(batch, Batch)
            x, y = batch
            self.assertIsInstance(x, torch.Tensor)
            self.assertIsInstance(y, torch.Tensor)

    def test_shape(self):
        data = torch.zeros(10, 5)
        self.assertEqual(
            [(3, 5), (3, 5), (3, 5), (1, 5)],
            [batch.shape for batch in TensorLoader(data, batch_size=3)]
        )

    def test_tuple_shape(self):
        X = torch.zeros(10, 5)
        Y = torch.zeros(10, 3)
        dataloader = TensorLoader((X, Y), batch_size=3)
        shapes = [(x.shape, y.shape) for x, y in dataloader]
        self.assertEqual([
            ((3, 5), (3, 3)),
            ((3, 5), (3, 3)),
            ((3, 5), (3, 3)),
            ((1, 5), (1, 3))
        ], shapes)

    def test_drop_last(self):
        data = torch.zeros(10, 5)
        self.assertEqual(
            [(3, 5), (3, 5), (3, 5)],
            [x.shape for x in TensorLoader(data, batch_size=3, drop_last=True)]
        )

    def test_tuple_length_mismatch(self):
        X = torch.zeros(10, 5)
        Y = torch.zeros(11)
        with self.assertRaises(ValueError):
            _ = TensorLoader((X, Y))
