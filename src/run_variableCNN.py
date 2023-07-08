"""
Script for running models with Weights & Biases.
"""
# PyTorch modules
import torch
from pytorch_lightning.loggers import CSVLogger, WandbLogger

# Weights and Biases
import wandb

# import data + model modules
import ukbb_data
import ukbb_ica_models

# import custom functions
import utils

######################################################################################

# initialise W&B run
wandb.init() # brauche ich das mit PL?

# increase reproducibility
utils.make_reproducible()

# initialise logger
wandb_logger = WandbLogger(project=,
                          run=, #?
                          log_model=, #True? all? best?
                          )

# initialise callbacks
checkpoint = utils.wandb_checkpoint_init()
early_stopping = utils.earlystopping_init()

# initialise trainer
trainer = pl.Trainer(accelerator='gpu', ### ADAPT
                     devices=[device], 
                     logger=wandb_logger,
                     log_every_n_steps=log_steps,
                     max_epochs=max_epochs,
                     callbacks=callbacks,
                     deterministic=True)


# end wandb
wandb.finish()