import typing
from typing import Generic, TypeVar, Iterator, Tuple, Union

import torch
from torch import Tensor

T = TypeVar('T', bound=Union[Tensor, Tuple[Tensor, ...]])


class TensorLoader(Generic[T]):
    """`TensorLoader` is similar to the combination of PyTorch's `TensorDataset` and `DataLoader`."""

    __slots__ = (
        '__data',
        '__tensor',
        'batch_size',
        'shuffle',
        'drop_last',
        'unpack_args'
    )

    def __init__(
            self,
            data: T,
            *,
            batch_size: int = 1,
            shuffle: bool = False,
            drop_last: bool = False,
            unpack_args: bool = False
    ) -> None:
        """
        :param data: tensor, tuple of tensor or namedtuple of tensor.
            Each sample will be retrieved by indexing tensors along the first dimension.
            When using a namedtuple as data, set `unpack_args=True`.
        :param batch_size: how many samples per batch to load. (default: `1`)
        :param shuffle: shuffle the data at every epoch. (default: `False`)
        :param drop_last: set to `True` to drop the last incomplete batch if the
            dataset size is not divisible by the batch size. If `False` and the
            size of dataset is not divisible by the batch size, then the last batch
            will be smaller. (default: `False`)
        :param unpack_args: set to `True` to unpack the argument of the constructor of `T`.
            Keep it as `False` when using a tuple as data. Set it to `True` when using a
            namedtuple as data. (default: `False`)
        """
        self.__data = data
        if isinstance(data, tuple):
            if any(len(x) != len(data[0]) for x in data):
                raise ValueError('length of tensors are not the same')
            self.__tensor = data[0]
        else:
            self.__tensor = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.unpack_args = unpack_args

    def __iter__(self) -> Iterator[T]:
        if isinstance(self.__data, tuple):
            return self.__tuple_iter()
        else:
            return self.__tensor_iter()

    def __len__(self) -> int:
        batch_num, remainder = divmod(len(self.__tensor), self.batch_size)
        if remainder > 0 and not self.drop_last:
            batch_num += 1
        return batch_num

    def __tensor_iter(self) -> Iterator[Tensor]:
        if typing.TYPE_CHECKING:
            assert isinstance(self.__data, Tensor)
        if self.shuffle:
            indices = torch.randperm(len(self.__tensor), device=self.__tensor.device)
            self.__data = self.__data[indices].contiguous()
        for start in range(0, len(self) * self.batch_size, self.batch_size):
            stop = start + self.batch_size
            yield self.__data[start:stop]

    def __tuple_iter(self) -> Iterator[Tuple[Tensor, ...]]:
        if typing.TYPE_CHECKING:
            assert isinstance(self.__data, tuple)
        TupleType = type(self.__data)
        if self.shuffle:
            indices = torch.randperm(len(self.__tensor), device=self.__tensor.device)
            self.__data = TupleType(x[indices] for x in self.__data)
        for start in range(0, len(self) * self.batch_size, self.batch_size):
            stop = start + self.batch_size
            iterable = (x[start:stop] for x in self.__data)
            yield TupleType(*iterable) if self.unpack_args else TupleType(iterable)
