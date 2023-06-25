# PyTorch modules
import torch
from torch import optim, nn
import pytorch_lightning as pl

# own utils module
import utils

################################################################################################

# define LightningModule
class simple1DCNN(pl.LightningModule):
    """
    Simple 1D Convolutional Neural Network (CNN) that takes ICA timeseries as input and predicts
    (i.e., regresses) a participant's age.
    Input:
        channels: number of ICA components.
        kernel_size: width of the kernel that is applied.
        activation: activation function to be used.
        loss: loss to be used.
        lr: learning rate to be used.
    Output:
        A model.
    """
    def __init__(self, channels=25, kernel_size=5, activation=nn.ReLU(), loss=nn.MSELoss(), lr=1e-3):
        super().__init__()
        self.in_channels = channels
        self.kernel_size = kernel_size
        self.act = activation
        self.loss = loss
        self.lr = lr
        self.save_hyperparameters()
        utils.make_reproducible()        
        
        # define convolutional and maxpool layers
        self.conv1 = nn.Conv1d(self.in_channels, 32, kernel_size=self.kernel_size)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=self.kernel_size)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=self.kernel_size)
        self.conv4 = nn.Conv1d(128, 256, kernel_size=self.kernel_size)
        self.maxpool = nn.MaxPool1d(kernel_size=self.kernel_size, stride=2)
                
        # define model architecture
        self.model = nn.Sequential(self.conv1,
                                    self.act,
                                    self.maxpool,
                                    self.conv2,
                                    self.act,
                                    self.maxpool,
                                    self.conv3,
                                    self.act,
                                    self.maxpool,
                                    self.conv4,
                                    self.act,
                                    self.maxpool,
                                    nn.Flatten(),
                                    nn.Linear(6144,1)
                                    # dropout?
                                    )
    
    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        return self.model(x)
        
    def training_step(self, batch, batch_idx): 
        # training_step defines the train loop, independent of forward
        x, y = batch
        y_hat = self.forward(x)
        y = torch.unsqueeze(y,1)
        loss = self.loss(y_hat, y)
        
        self.log('train_loss', loss)
        return loss
    
    def evaluate(self, batch, stage=None):
        x, y = batch
        y_hat = self.forward(x)
        y = torch.unsqueeze(y,1)
        loss = self.loss(y_hat, y)
        mae = nn.functional.l1_loss(y_hat, y)
        
        if stage:
            self.log(f'{stage}_loss', loss)
            self.log(f'{stage}_mae', mae)
            
        if stage == 'val':
            return {"val_loss": loss, "diff": (y - y_hat), "target": y, 'mae': mae}
    
    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, 'val')
    
    def test_step(self, batch, batch_idx):
        self.evaluate(batch, 'test')
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr) 
        return optimizer

class variable1DCNN(pl.LightningModule):
    """
    1D Convolutional Neural Network (CNN) that takes ICA timeseries as input and predicts
    (i.e., regresses) a participant's age.
    Inspired by simple1DCNN but with setup that enables easy depth and channel modification.
    Input:
        channels: number of ICA components.
        kernel_size: width of the kernel that is applied.
        activation: activation function to be used.
        loss: loss to be used.
        lr: learning rate to be used.
        depth: model depth (number of convolutional layers)
        start_out: output dimensionality for first convolutional layer that will be scaled up
    Output:
        A model.
    """
    def __init__(self, channels=25, kernel_size=5, activation=nn.ReLU(), loss=nn.MSELoss(), lr=1e-4, depth=4, start_out=32):
        super().__init__()
        self.in_channels = channels
        self.kernel_size = kernel_size
        self.act = activation
        self.loss = loss
        self.lr = lr
        self.depth = depth
        self.start_out = start_out
        self.save_hyperparameters()
        utils.make_reproducible()        
        
        # define convolutional and maxpool layers
        self.conv1 = nn.Conv1d(self.in_channels, self.start_out, kernel_size=self.kernel_size)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=self.kernel_size)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=self.kernel_size)
        self.conv4 = nn.Conv1d(128, 256, kernel_size=self.kernel_size)
        
        self.maxpool = nn.MaxPool1d(kernel_size=self.kernel_size, stride=2)
                
        # define model architecture
        channel = self.start_out
        self.encoder = nn.Sequential()
        for layer in range(self.depth):
            if layer == 1:
                conv_layer = nn.Conv1d(self.in_channels, channel, kernel_size=self.kernel_size)
            else:
                new_channel = channel*2
                conv_layer = nn.Conv1d(channel, new_channel, kernel_size=self.kernel_size)
                # set output channel size as new input channel size
                channel = new_channel
            # append to list
            self.encoder.append(conv_layer)
            self.encoder.append(self.act)
            self.encoder.append(self.maxpool)
        self.encoder.append(nn.Flatten())
            
       # self.encoder = nn.Sequential(self.conv1,
        #                            self.act,
         #                           self.maxpool,
          #                          self.conv2,
           #                         self.act,
            #                        self.maxpool,
             #                       self.conv3,
              #                      self.act,
               #                     self.maxpool,
                #                    self.conv4,
                 #                   self.act,
                  #                  self.maxpool,
                   #                 nn.Flatten()
                    #                )
    
    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions          
        self.encoder(x)
        size = x.size(1)
        return nn.Linear(size,1)
        
    def training_step(self, batch, batch_idx): 
        # training_step defines the train loop, independent of forward
        x, y = batch
        y_hat = self.forward(x)
        y = torch.unsqueeze(y,1)
        loss = self.loss(y_hat, y)
        
        self.log('train_loss', loss)
        return loss
    
    def evaluate(self, batch, stage=None):
        x, y = batch
        y_hat = self.forward(x)
        y = torch.unsqueeze(y,1)
        loss = self.loss(y_hat, y)
        mae = nn.functional.l1_loss(y_hat, y)
        
        if stage:
            self.log(f'{stage}_loss', loss)
            self.log(f'{stage}_mae', mae)
            
        if stage == 'val':
            return {"val_loss": loss, "diff": (y - y_hat), "target": y, 'mae': mae}
    
    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, 'val')
    
    def test_step(self, batch, batch_idx):
        self.evaluate(batch, 'test')
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr) 
        return optimizer