import pandas as pd
import numpy as np
import os
import random
import logging
from typing import Any, Dict, Generator, Optional

# Pytorch etc
import torch
# import pytorch_lightning as pl
# from pytorch_lightning.loggers import CSVLogger

# visualisation
import matplotlib.pyplot as plt
import seaborn as sns

##################################################################################

### SAVING THINGS
def save_data_info(path, datamodule):
    """
    Creates a new directory "data_info" and saves the loaded overview of used datapoints
    (participant ID, age, file location) and the indices used for the train, validation,
    and test set in separate files in that new directory.
    """
    # create data_info directory
    os.makedirs(path+'data_info', exist_ok=True)
    # save overview dataframe as csv
    datamodule.data.labels.to_csv(path+'data_info/overview.csv', index=False)
    # save train/val/test indices as integers ('%i') in csv
    np.savetxt(path+'data_info/train_idx.csv', datamodule.train_idx, fmt='%i')
    np.savetxt(path+'data_info/val_idx.csv', datamodule.val_idx, fmt='%i')
    np.savetxt(path+'data_info/test_idx.csv', datamodule.test_idx, fmt='%i')

### REPRODUCIBILITY
# increase reproducitility by defining a seed worker
# instructions from https://pytorch.org/docs/stable/notes/randomness.html#reproducibility
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# copy seed_everything function provided in Lightning vers. 1.6.3 that is not 
# available anymore in 2.0.2 from 
# https://pytorch-lightning.readthedocs.io/en/1.6.3/_modules/pytorch_lightning/utilities/seed.html#seed_everything
log = logging.getLogger(__name__)
def seed_everything(seed: Optional[int] = None, workers: bool = False) -> int:
    """Function that sets seed for pseudo-random number generators in: pytorch, numpy, python.random In addition,
    sets the following environment variables:

    - `PL_GLOBAL_SEED`: will be passed to spawned subprocesses (e.g. ddp_spawn backend).
    - `PL_SEED_WORKERS`: (optional) is set to 1 if ``workers=True``.

    Args:
        seed: the integer value seed for global random state in Lightning.
            If `None`, will read seed from `PL_GLOBAL_SEED` env variable
            or select it randomly.
        workers: if set to ``True``, will properly configure all dataloaders passed to the
            Trainer with a ``worker_init_fn``. If the user already provides such a function
            for their dataloaders, setting this argument will have no influence. See also:
            :func:`~pytorch_lightning.utilities.seed.pl_worker_init_function`.
    """
    max_seed_value = np.iinfo(np.uint32).max
    min_seed_value = np.iinfo(np.uint32).min

    if seed is None:
        env_seed = os.environ.get("PL_GLOBAL_SEED")
        if env_seed is None:
            seed = _select_seed_randomly(min_seed_value, max_seed_value)
            rank_zero_warn(f"No seed found, seed set to {seed}")
        else:
            try:
                seed = int(env_seed)
            except ValueError:
                seed = _select_seed_randomly(min_seed_value, max_seed_value)
                rank_zero_warn(f"Invalid seed found: {repr(env_seed)}, seed set to {seed}")
    elif not isinstance(seed, int):
        seed = int(seed)

    if not (min_seed_value <= seed <= max_seed_value):
        rank_zero_warn(f"{seed} is not in bounds, numpy accepts from {min_seed_value} to {max_seed_value}")
        seed = _select_seed_randomly(min_seed_value, max_seed_value)

    # using `log.info` instead of `rank_zero_info`,
    # so users can verify the seed is properly set in distributed training.
    log.info(f"Global seed set to {seed}")
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    os.environ["PL_SEED_WORKERS"] = f"{int(workers)}"

    return seed

#### VISUALISATIONS
def get_metrics(trainer, show=False):
    """
    Gets the metrics of a current trainer object (aka the training + validation information
    of the trained with the trainer) and returns them in a dataframe.
    """
    metrics = pd.read_csv(f'{trainer.logger.log_dir}/metrics.csv')
    # set epoch as index
    metrics.set_index('epoch', inplace=True)
    # move step to first position
    step = metrics.pop('step')
    metrics.insert(0, 'step', step)
    # display first 5 rows if required
    if show:
        display(metrics.dropna(axis=1, how='all').head())
    return metrics

def plot_training(data, yscale='log', title=''):
    """
    Plots the training process of a model. 
    Expects the metrics_df from get_metrics as input.
    """
    # only include train + validation values in plot
    visualisation = sns.relplot(data=data.loc[:,:'val_mae'], kind='line')
    # set scale and title
    visualisation.set(yscale=yscale)
    plt.title(title)
    plt.plot()
    