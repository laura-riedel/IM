"""
Script for running models with Weights & Biases.
"""
# PyTorch modules
import torch
from torch import optim, nn
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

# Weights and Biases, argparse
import wandb
import argparse
import yaml
 
# import data + model modules
import ukbb_data
import ukbb_ica_models

# import custom functions
import utils

######################################################################################

parser = argparse.ArgumentParser(prog='RunVariableCNN',
                                description='Trains and tests a model configuration with variableCNN on UKBB data.',
                                epilog='Training + testing complete.')
# wandb setup
parser.add_argument('project', type=str, help='Enter the name of the current Weights & Biases project.')
parser.add_argument('tags', type=list, help='Enter tags to help classify this run in the future. Expects list of strings.')

# trainer setup
# accelerator + devices?
# parser.add_argument()
parser.add_argument('--log_steps', type=int, default=10, help='Enter after how many steps a logging shall take place. Default: 10')
parser.add_argument('--max_epochs', type=int, default=175, help='Enter maximum amount of epochs the model is allowed to train. Default: 175')
parser.add_argument('--patience', type=int, default=15, help='Enter number of epochs the model continues to train without improving the validation loss without calling early stopping. Defautl: 15')

# dataset setup
parser.add_argument('--ica', type='str', default='25', help='Enter whether to load ICA25 ("25") or ICA100 ("100"). Default: "25".')
parser.add_argument('--good_components', type=bool, default=True, help='Enter whether to use the good components only. Default: True.')

# model hyperparameters
parser.add_argument('--in_channels', type=int, default=25, help='Enter number of ICA components. Default: 25')
parser.add_argument('--kernel_size', type=int, default=5, help='Enter the width of the kernel that is applied. Default: 5')
parser.add_argument('--activation', default=nn.ReLU(), help='Enter activation function to be used. Default: torch.nn.ReLU()')
parser.add_argument('--loss', default=nn.MSELoss(), help='Enter loss to be used. Default: torch.nn.MSELoss()')
parser.add_argument('--lr', type=float, default=1e-3, help='Enter learning rate for training. Default: 1e-3') #, metavar='learning rate'
parser.add_argument('--depth', type=int, default=4, help='Enter the model depth (number of convolutional layers).')
parser.add_argument('--start_out', type=int, default=32, help='Enter the output dimensionality for first convolutional layer that will be scaled up (power of 2).')
parser.add_argument('--stride', type=int, default=2, help='Enter the stride for pooling layers. Default: 2')
parser.add_argument('--weight_decay', type=float, default=0, help='Enter the weight decay for the AdamW optimiser. If the value is 0, no weight decay is applied. Default: 0')
parser.add_argument('--dilation', type=int, default=1, help='Enter the spacing between the kernel points. Default: 1')
parser.add_argument('--conv_dropout', type=float, default=0, help='Enter the probability for channel dropout after each convolutional layer. If the value is 0, no dropout is applied. Default: 0')
parser.add_argument('--final_dropout', type=float, default=0, help='Enter the probability for channel dropout before the final linear layer. If the value is 0, no dropout is applied. Default: 0')
parser.add_argument('--double_conv', type=bool, default=False, help='Enter boolean flag to turn on or off whether to have two convolutional layers before pooling. Default: False')
parser.add_argument('--batch_norm', type=bool, default=False, help='Enter boolean flag to turn batch normalisation on or off.')

args = parser.parse_args()

######################################################################################
# initialise W&B run
wandb.init(project=args.project, 
          tags=args.tags) # brauche ich das mit PL?


# increase reproducibility
utils.make_reproducible()

# make sure only one CPU thread is used
torch.set_num_threads(1)

# initialise logger
wandb_logger = WandbLogger(project=args.project,
                         log_model='all', #True? all? best?
                         )
# # initialise logger
# path = '../tracking/ICA25/ModelDepth1DCNN/terminaltest'
# logger = utils.logger_init(save_dir=path)

# initialise callbacks
checkpoint = utils.wandb_checkpoint_init()
# checkpoint = utils.checkpoint_init(save_dir=path)
early_stopping = utils.earlystopping_init(patience=args.patience)

# initialise trainer
trainer = pl.Trainer(accelerator='cpu', 
                     logger=wandb_logger,
                    #  logger=logger,
                     log_every_n_steps=args.log_steps,
                     max_epochs=args.max_epochs,
                     callbacks=[checkpoint, early_stopping],
                     deterministic=True)

# initialise DataModule
datamodule = ukbb_data.UKBBDataModule('/ritter/share/data/UKBB/ukb_data/', # data path
                                      args.ica,
                                      args.good_components,
                                      '../data/') # good components path

# initialise model
variable_CNN = ukbb_ica_models.variable1DCNN(in_channels=args.in_channels,
                                            kernel_size=args.kernel_size,
                                            activation=args.activation,
                                            loss=args.loss,
                                            lr=args.lr,
                                            depth=args.depth,
                                            start_out=args.start_out,
                                            stride=args.stride,
                                            conv_dropout=args.conv_dropout,
                                            final_dropout=args.final_dropout,
                                            weight_decay=args.weight_decay,
                                            dilation=args.dilation,
                                            double_conv=args.double_conv,
                                            batch_norm=args.batch_norm,
                                            execution='t') # specifying 'terminal' doesn't set dilation in conv layer

# train model
trainer.fit(variable_CNN, datamodule=datamodule)

# test model
trainer.test(ckpt_path='best', datamodule=datamodule)  