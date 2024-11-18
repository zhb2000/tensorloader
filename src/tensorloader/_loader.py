import typing
from typing import Generic, TypeVar, Iterator, Tuple, Union

import torch
from torch import Tensor

T = TypeVar('T', bound=Union[Tensor, Tuple[Tensor, ...]])


class TensorLoader(Generic[T]):
    """`TensorLoader` is similar to the combination of PyTorch's `TensorDataset` and `DataLoader`."""

    __slots__ = (
        '__data',
        '__length',
        '__data_type',
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
        self.__data: Union[Tensor, Tuple[Tensor, ...]]
        if isinstance(data, tuple):
            self.__data = tuple(data)
            self.__length = len(data[0])
            for i, x in enumerate(data):
                if len(x) != self.__length:
                    raise ValueError(
                        f'length of tensors are not the same, length of the first tensor is {self.__length}, '
                        f'but the length of tensor at index {i} is {len(x)}'
                    )
        elif isinstance(data, Tensor):
            self.__data = data
            self.__length = len(data)
        else:
            raise TypeError(f'unsupported data type: {type(data)}')
        self.__data_type = type(data)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.unpack_args = unpack_args

    def __iter__(self) -> Iterator[T]:
        if issubclass(self.__data_type, tuple):
            return self.__tuple_iter()
        else:
            return self.__tensor_iter()

    def __len__(self) -> int:
        batch_num, remainder = divmod(self.__length, self.batch_size)
        if remainder > 0 and not self.drop_last:
            batch_num += 1
        return batch_num

    def __tensor_iter(self) -> Iterator[Tensor]:
        if typing.TYPE_CHECKING:
            assert isinstance(self.__data, Tensor)
        if self.shuffle:
            indices = torch.randperm(self.__length, device=self.__data.device)
            self.__data = self.__data[indices].contiguous()
        for start in range(0, len(self) * self.batch_size, self.batch_size):
            stop = start + self.batch_size
            yield self.__data[start:stop]

    def __tuple_iter(self) -> Iterator[Tuple[Tensor, ...]]:
        if typing.TYPE_CHECKING:
            assert isinstance(self.__data, tuple)
            assert issubclass(self.__data_type, tuple)
        if self.shuffle:
            indices = torch.randperm(self.__length, device=self.__data[0].device)
            self.__data = self.__data_type(x[indices.to(x.device)].contiguous() for x in self.__data)
        for start in range(0, len(self) * self.batch_size, self.batch_size):
            stop = start + self.batch_size
            iterable = (x[start:stop] for x in self.__data)
            yield self.__data_type(*iterable) if self.unpack_args else self.__data_type(iterable)
