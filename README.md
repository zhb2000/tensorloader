# Tensor Loader

![PyPI](https://img.shields.io/pypi/v/tensorloader?logo=pypi&logoColor=white) ![PyPI - Python Version](https://img.shields.io/pypi/pyversions/tensorloader?logo=python&logoColor=white) ![PyPI - License](https://img.shields.io/pypi/l/tensorloader)

`TensorLoader` is similar to the combination of PyTorch's `TensorDataset` and `DataLoader`. It is faster and has better type hints.

## Installation

Install from PyPI:

```shell
pip install tensorloader
```

Install from source:

```shell
git clone https://github.com/zhb2000/tensorloader.git
cd tensorloader
pip install .
```

## Usage

This package only contains a `TensorLoader` class.

```python
from tensorloader import TensorLoader
```

Use a single tensor as data:

```python
X = torch.tensor(...)
dataloader = TensorLoader(X)
for x in dataloader:
    ...
```

Use a tuple of tensor as data:

```python
X = torch.tensor(...)
Y = torch.tensor(...)
dataloader = TensorLoader((X, Y))
for x, y in dataloader:  # unpack the batch tuple as x, y
    ...
```

Use a namedtuple of tensor as data:

```python
from collections import namedtuple

Batch = namedtuple('Batch', ['x', 'y'])
X = torch.tensor(...)
Y = torch.tensor(...)
# set unpack_args=True when using a namedtuple as data
dataloader = TensorLoader(Batch(X, Y), unpack_args=True)
for batch in dataloader:
    assert isinstance(batch, Batch)
    assert isinstance(batch.x, torch.Tensor)
    assert isinstance(batch.y, torch.Tensor)
    x, y = batch
    ...
```

PS: Namedtuples are similar to common tuples and they allow field access by name which makes code more readable. For more information, see the [documentation](https://docs.python.org/3/library/collections.html#collections.namedtuple) of namedtuple.

## Speed Test

`TensorLoader` is much faster than `TensorDataset` + `DataLoader`, for it uses vectorized operations instead of creating costly Python lists.

```python
import timeit
import torch
from torch.utils.data import TensorDataset, DataLoader
from tensorloader import TensorLoader

def speed_test(epoch_num: int, **kwargs):
    sample_num = int(1e6)
    X = torch.zeros(sample_num, 10)
    Y = torch.zeros(sample_num)
    tensorloader = TensorLoader((X, Y), **kwargs)
    torchloader = DataLoader(TensorDataset(X, Y), **kwargs)

    def loop(loader):
        for _ in loader:
            pass

    t1 = timeit.timeit(lambda: loop(tensorloader), number=epoch_num)
    t2 = timeit.timeit(lambda: loop(torchloader), number=epoch_num)
    print(f'TensorLoader: {t1:.4g}s, TensorDatset + DataLoader: {t2:.4g}s.')
```

```
>>> speed_test(epoch_num=10, batch_size=128, shuffle=False)
TensorLoader: 0.363s, TensorDatset + DataLoader: 54.39s.
>>> speed_test(epoch_num=10, batch_size=128, shuffle=True)
TensorLoader: 0.9296s, TensorDatset + DataLoader: 56.54s.
>>> speed_test(epoch_num=10, batch_size=10000, shuffle=False)
TensorLoader: 0.005262s, TensorDatset + DataLoader: 55.57s.
>>> speed_test(epoch_num=10, batch_size=10000, shuffle=True)
TensorLoader: 0.5682s, TensorDatset + DataLoader: 57.71s.
```
