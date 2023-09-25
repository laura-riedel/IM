import pandas as pd
import numpy as np
import os
import random
import logging
from typing import Any, Dict, Generator, Optional

# Pytorch etc
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

# import data + model modules
from ukbb_package import ukbb_data
from ukbb_package import ukbb_ica_models

# visualisation
import matplotlib.pyplot as plt
import seaborn as sns

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
    
# define a function for easier set-up
def make_reproducible(random_state=43):
    """
    Function that calls on 'seed_everything' to set seeds for pseudo-random number generators
    and that enforeces more deterministic behaviour in torch.
    Input:
        random_state: seed to set
    """
    seed_everything(seed=random_state, workers=True)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed(random_state)

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

# call reproducibility function
make_reproducible()

#### MODEL INITIALISATION
# logger
def logger_init(save_dir):
    logger = CSVLogger(save_dir=save_dir, name='Logs')
    return logger

# callbacks
def checkpoint_init(save_dir):
    checkpoint = ModelCheckpoint(dirpath=save_dir+'Checkpoint/',
                                 filename='models-{epoch:02d}-{val_loss:.2f}',
                                 monitor='val_loss',
                                 save_top_k=1,
                                 mode='min')
    return checkpoint

def wandb_checkpoint_init():
    checkpoint = ModelCheckpoint(monitor='val_loss',
                                 save_top_k=1,
                                 mode='min')
    return checkpoint

def earlystopping_init(patience=15):
    early_stopping = EarlyStopping(monitor='val_loss',
                                   patience=patience)
    return early_stopping

# initialise trainer
def trainer_init(device, logger, log_steps=10, max_epochs=175, callbacks=[]):
    trainer = pl.Trainer(accelerator='gpu',
                         devices=[device], 
                         logger=logger,
                         log_every_n_steps=log_steps,
                         max_epochs=max_epochs,
                         callbacks=callbacks,
                         deterministic=True)
    return trainer

#### SWEEP FUNCTION
# define a run function
def model_run(config=None):
    # initialise W&B run
    with wandb.init(config=config):
        # if called by wandb.agent, 
        # this config will be set by Sweep Controller
        config = wandb.config

        # initialise logger
        wandb_logger = WandbLogger(project=config.project,
                                  tags=['all components'], # 'best components'
                                  log_model=True, # logs at the end of training
                                  )

        # initialise callbacks
        checkpoint = utils.wandb_checkpoint_init()
        early_stopping = utils.earlystopping_init(patience=config.patience)

        # initialise trainer
        trainer = pl.Trainer(accelerator='cpu', 
                             logger=wandb_logger,
                             log_every_n_steps=config.log_steps,
                             max_epochs=config.max_epochs,
                             callbacks=[checkpoint, early_stopping],
                             deterministic=True)

        # initialise DataModule
        datamodule = ukbb_data.UKBBDataModule('/ritter/share/data/UKBB/ukb_data/')

        # initialise model
        variable_CNN = ukbb_ica_models.variable1DCNN(in_channels=config.in_channels,
                                            kernel_size=config.kernel_size,
                                            lr=config.lr,
                                            depth=config.depth,
                                            start_out=config.start_out,
                                            stride=config.stride,
                                            conv_dropout=config.conv_dropout,
                                            final_dropout=config.final_dropout,
                                            weight_decay=config.weight_decay,
                                            dilation=config.dilation,
                                            double_conv=config.double_conv,
                                            batch_norm=config.batch_norm)

        # train model
        trainer.fit(variable_CNN, datamodule=datamodule)

#### TRAIN + TEST MODELS WITH CONFIG DICT
def train_model(log_path, data_path, config, device, execution='nb'):
    """
    Fuction for training a variable 1D-CNN model on a GPU using external config information.
    Outputs a trained model.
    Input:
        log_path: path to where logs, checkpoints and data info should be saved
        data_path: path to location where data is saved (expectations see ukbb_data.DataModule)
        config: configuration dictionary of the form 
                {'project': '', 'model': '', 'parameters': {all parameters except for execution}}
        device: which GPU to run on
        execution: whether model is called from a Jupyter Notebook ('nb') or the terminal ('t'). 
                    Teriminal call cannot handle dilation. Default: 'nb'.
    Output:
        trainer: trained model
        datamodule: PyTorch Lightning UKBB DataModule
    """
    full_log_path = log_path+config['project']+'/'+config['model']+'/'
    
    # initialise model
    variable_CNN = ukbb_ica_models.variable1DCNN(in_channels=config['parameters']['in_channels'],
                                                kernel_size=config['parameters']['kernel_size'],
                                                lr=config['parameters']['lr'],
                                                depth=config['parameters']['depth'],
                                                start_out=config['parameters']['start_out'],
                                                stride=config['parameters']['stride'],
                                                conv_dropout=config['parameters']['conv_dropout'],
                                                final_dropout=config['parameters']['final_dropout'],
                                                weight_decay=config['parameters']['weight_decay'],
                                                dilation=config['parameters']['dilation'],
                                                double_conv=config['parameters']['double_conv'],
                                                batch_norm=config['parameters']['batch_norm'],
                                                execution=execution)

    # initialise logger
    logger = logger_init(save_dir=full_log_path)

    # set callbacks
    early_stopping = earlystopping_init(patience=config['parameters']['patience'])

    checkpoint = checkpoint_init(save_dir=full_log_path)

    # initialise trainer
    trainer = trainer_init(device=device,
                                logger=logger,
                                log_steps=config['parameters']['log_steps'],
                                max_epochs=config['parameters']['max_epochs'],
                                callbacks=[early_stopping, checkpoint])

    # initialise DataModule
    datamodule = ukbb_data.UKBBDataModule(data_path,
                                         ica=config['parameters']['ica'],
                                         good_components=config['parameters']['good_components'])

    # train model
    trainer.fit(variable_CNN, datamodule=datamodule)
    print('Training complete.')

    # save info on which data was used + what the train/val/test split was
    save_data_info(path=full_log_path, datamodule=datamodule)
    
    return trainer, datamodule

def test_model(trainer, datamodule, config):
    """
    Fuction for using the same model and testing set-up using external config information.
    Outputs a test score and a plot visualising the training progression.
    Input:
        trainer: the trained model
        datamodule: PyTorch Lightning DataModule instance
        config: configuration dictionary of the form 
                {'project': '', 'model': '', 'parameters': {all parameters except for execution}}
    """
    model_info = config['model']
    ica_info = config['parameters']['ica']
    if config['parameters']['good_components'] == True:
        gc = 'good components only'
    else:
        gc = 'all components'
    if config['project'] == 'FinalModels_IM':
        data_info = '(data subset)'
    else:
        data_info = '(full data)'
    
    # test model
    print(f'\nTesting model "{model_info}" {data_info}...')
    trainer.test(ckpt_path='best', datamodule=datamodule)
    
    # visualise training
    print(f'\nVisualise training of model "{model_info}" {data_info}...')    
    metrics = get_current_metrics(trainer, show=True)
    plot_training(data=metrics, title=f'Training visualisation of the ICA{ica_info} 1D-CNN with {gc} {data_info}.')

#### VISUALISATIONS
def get_current_metrics(trainer, show=False):
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

def get_saved_metrics(logging_path, show=False):
    """
    Gets the metrics of saved training + validation information
    and returns them in a dataframe.
    """
    metrics = pd.read_csv(f'{logging_path}metrics.csv')
    # set epoch as index
    metrics.set_index('epoch', inplace=True)
    # move step to first position
    step = metrics.pop('step')
    metrics.insert(0, 'step', step)
    # display first 5 rows if required
    if show:
        display(metrics.dropna(axis=1, how='all').head())
    return metrics

def load_datainfo(path_to_data_info, info_type):
    """
    Loads information from the data_info directory.
    Input:
        path_to_data_info: path to where data_info is saved (without data_info itself in path)
        info_type: 'train_idx', 'val_idx', 'test_idx', or 'overview'
    Output:
        info: if info_type is one of the split indices: numpy array of the split indices
              if info_type is overview: overview.csv as dataframe  
    """
    if info_type.endswith('idx'):
        info = np.loadtxt(path_to_data_info+'data_info/'+info_type+'.csv', dtype='int')
    else:
        info = pd.read_csv(path_to_data_info+'data_info/overview.csv')
    return info

def plot_training(data, yscale='log', title='', xmin=None, xmax=None):
    """
    Plots the training process of a model. 
    Expects the metrics_df from get_metrics as input.
    """
    if xmin == None:
        xmin = data.index.min()
    if xmax == None:
        xmax = data.index.max()
    # only include train + validation values in plot
    visualisation = sns.relplot(data=data.loc[xmin:xmax,'train_loss':'val_mae'], kind='line')
    # set scale and title
    visualisation.set(yscale=yscale)
    plt.title(title)
    plt.show()

