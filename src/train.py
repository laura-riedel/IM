"""
Script for running hyperparameter sweep with Weights & Biases.
"""
# PyTorch modules
import torch
from torch import optim, nn
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

# Weights and Biases, argparse, sys
import wandb
import argparse
import yaml
import sys
 
# import data + model modules
import ukbb_data
import ukbb_ica_models

# import custom functions
import utils

###################

with wandb.init() as run: # project name set through yaml
    config = wandb.config

    # increase reproducibility
    utils.make_reproducible()

    # make sure only one CPU thread is used
    torch.set_num_threads(1)
    
    # initialise model
    # catch cases where the model architecture is impossible
    # or would cause memory issues
    try:
        variable_CNN = ukbb_ica_models.variable1DCNN(in_channels=config.in_channels,
                                                    kernel_size=config.kernel_size,
                                                    # activation=config.activation,
                                                    # loss=config.loss,
                                                    lr=config.lr,
                                                    depth=config.depth,
                                                    start_out=config.start_out,
                                                    stride=config.stride,
                                                    conv_dropout=config.conv_dropout,
                                                    final_dropout=config.final_dropout,
                                                    weight_decay=config.weight_decay,
                                                    double_conv=config.double_conv,
                                                    batch_norm=config.batch_norm,
                                                    execution='t') # specifying 'terminal' doesn't set dilation in conv layer
    except:     
        # add tag for later filtering
        run.tags = run.tags + ('impossible_architecture_or_oom',) 
        # finish run 
        wandb.finish(exit_code=555)
        sys.exit()
    
    # catch models with too many parameters
    pytorch_total_params = sum(torch.numel(p) for p in variable_CNN.parameters())
    # threshold: 5Mio
    if pytorch_total_params > 250000:
        # add tag for later filtering
        run.tags = run.tags + ('too_many_params',) 
        # finish run 
        wandb.finish(exit_code=555)
        sys.exit()

    # initialise logger
    wandb_logger = WandbLogger(log_model=False)

    # initialise callbacks
    # checkpoint = utils.wandb_checkpoint_init()
    early_stopping = utils.earlystopping_init(patience=config.patience)

    # initialise trainer
    trainer = pl.Trainer(accelerator='cpu', 
                        logger=wandb_logger,
                        log_every_n_steps=config.log_steps,
                        max_epochs=config.max_epochs,
                        callbacks=[early_stopping], # ,checkpoint
                        enable_checkpointing=False,
                        deterministic=True,
                        max_time='00:03:00:00') # stop after 3h

    # initialise DataModule
    datamodule = ukbb_data.UKBBDataModule(data_path='/ritter/share/data/UKBB/ukb_data/',
                                        ica=config.ica,
                                        good_components=config.good_components) 
        

    # train model
    trainer.fit(variable_CNN, datamodule=datamodule)  
    
    wandb.finish()