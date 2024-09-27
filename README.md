# Fork of [a2032/a2032](https://github.com/a2032/a2032)

This fork of [the original implementation of HistNet](https://github.com/a2032/a2032) is used to produce unittest cases for other implementations of HistNet. The concept of these unittest cases is to define the desired output for some input data. Other implementations of HistNet, e.g., implementations in frameworks other than PyTorch, can then use these pairs of inputs and outputs to assert their correctness.

To install this project, call

```sh
conda create --name histnet
conda activate histnet
conda install pytorch==1.12.1 torchvision cudatoolkit=10.2 pandas tensorboard wandb scikit-learn==1.0.1 scipy==1.7.1 tqdm quadprog cvxpy mkl==2024.0 numpy==1.22 -c pytorch -c conda-forge
conda develop .
```

To generate the unittest cases in a file `histnet_unittest_cases.npy`, call

```sh
conda activate histnet
python generate_unittest_cases.py --module histnet histnet_unittest_cases.npy
python generate_unittest_cases.py --module mab mab_unittest_cases.npy
```

And to load the data elsewhere for unit testing, use

```python
import numpy as np
cases = np.load("histnet_unittest_cases.npy", allow_pickle=True)
```
