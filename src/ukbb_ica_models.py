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
        activation: activation function to be used.
        loss: loss to be used.
        lr: learning rate to be used.
    Output:
        A trained model.
    """
    def __init__(self, in_channels=25, activation=nn.ReLU(), loss=nn.MSELoss(), lr=1e-3):
        super().__init__()
        self.in_channels = in_channels
        self.kernel_size = 5
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
    Inspired by simple1DCNN but with setup that enables easy kernel, depth and channel modification.
    Input:
        in_channels: number of ICA components.
        kernel_size: width of the kernel that is applied.
        activation: activation function to be used.
        loss: loss to be used.
        lr: learning rate to be used.
        depth: model depth (number of convolutional layers).
        start_out: output dimensionality for first convolutional layer that will be scaled up.
        stride: stride for MaxPool layers.
        weight_decay: weight decay for optimiser.
        dilation: spacing between the kernel points.
        conv_dropout: optional dropout after each conv layer.
        final_dropout: optional final dropout before linear layer.
        double_conv: boolean flag to turn on or off whether to have two conv layers before pooling.
        batch_norm: boolean flag to turn batch normalisation on or off.
        execution: whether model is called from a Jupyter Notebook ('nb') or the terminal ('t'). 
                    Teriminal call cannot handle dilation. Default: 'nb'.
    Output:
        A model.
    """
    def __init__(self, in_channels=25, kernel_size=5, activation=nn.ReLU(), loss=nn.MSELoss(), 
                 lr=1e-3, depth=4, start_out=32, stride=2, weight_decay=0, dilation=1,
                 conv_dropout=0, final_dropout=0, double_conv=False, batch_norm=False, execution='nb'):
        super().__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.act = activation
        self.loss = loss
        self.lr = lr
        self.depth = depth
        self.start_out = start_out
        self.stride = stride
        self.conv_dropout = conv_dropout
        self.final_dropout = final_dropout
        self.weight_decay = weight_decay
        self.dilation = dilation
        self.double_conv = double_conv
        self.batch_norm = batch_norm
        self.execution = execution
        # save hyperparameters
        self.save_hyperparameters()
        # make reproducible
        utils.make_reproducible()        
        
        # additional variables
        # for tracking best validation loss during training
        self.best_val_loss = 10000
        # tracking validation step outputs
        self.validation_step_outputs = []
        
        # define maxpool layer
        self.maxpool = nn.MaxPool1d(kernel_size=self.kernel_size, stride=self.stride)
                
        # define model architecture
        # encoder
        channel = self.start_out
        self.encoder = nn.Sequential()
        # add X sequences of convolutional layer, activation + maxpool
        for layer in range(self.depth):
            # treat first layer a bit differently
            if layer == 0:
                self.add_conv_layer(self.in_channels, channel)
            # all other layers
            else:
                new_channel = channel*2
                self.add_conv_layer(channel, new_channel)
                # set output channel size as new input channel size
                channel = new_channel
            # add second conv layer if flag = True (complete with activation (+ BatchNorm + dropout))
            if self.double_conv:
                new_channel = channel*2
                self.add_conv_layer(channel, new_channel)
                # update output channel size again as new input channel size
                channel = new_channel
            # add pooling layer
            if layer == self.depth-1:
                # average pooling after last conv layer
                self.encoder.append(nn.AvgPool1d(self.kernel_size, self.stride))
            else:
                # normally maxpool
                self.encoder.append(self.maxpool)
        # add a final flatten layer at the end
        self.encoder.append(nn.Flatten())
        
        # check: too many parameters?
        total_encoder_params = self.get_num_parameters(self.encoder)   
        # threshold: 5,000,000
        if total_encoder_params > 5000000:
            raise Exception(f'More than 5Mio parameters in encoder! Count: {total_encoder_params}')
        
        # decoder
        flattened_dimension = self.get_flattened_dimension(self.encoder)
        self.decoder = nn.Sequential()
        # add dropout if wanted
        if self.final_dropout:
            self.decoder.append(nn.Dropout1d(p=self.final_dropout))
        self.decoder.append(nn.Linear(flattened_dimension, 1))
        
    def add_conv_layer(self, in_channel, out_channel):
        """ Function to add a convolutional layer and activation function
        (and optionally BatchNorm and conv dropout) to encoder architecture.
        Input:
            in_channel: input channel dimension.
            out_channel: output channel dimension.
        """
        if self.execution == 'nb':
            # Jupyter Notebook can handle dilation
            conv_layer = nn.Conv1d(in_channel, out_channel, kernel_size=self.kernel_size, dilation=self.dilation)
        else: 
            # terminal cannot handle dilation
            conv_layer = nn.Conv1d(in_channel, out_channel, kernel_size=self.kernel_size)
        # append to list
        self.encoder.append(conv_layer)
        
        # add BatchNorm if flag = True
        if self.batch_norm:
            self.encoder.append(nn.BatchNorm1d(out_channel))
            
        # add activation to encoder
        self.encoder.append(self.act)
        
        # add dropout if wanted
        if self.conv_dropout:
            self.encoder.append(nn.Dropout1d(p=self.conv_dropout))

    def get_flattened_dimension(self, model):
        """Function to figure out the dimension after nn.Flatten()
        so that it can be fed into the linear layer.
        Takes a given model, creates dummy input, and checks the
        dimensions after the input has gone through the model.
        Input:
            model: a (sequential) model architecture
        Output:
            input dimension size for linear layer
        """
        input = torch.randn(10, self.in_channels, 490)
        x = model(input)
        return x.size(1)
    
    def get_num_parameters(self, model):
        """
        Calculate how many parameters a model uses overall.
        """
        return sum(torch.numel(p) for p in model.parameters())
    
    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions          
        x = self.encoder(x)
        return self.decoder(x)
        
    def training_step(self, batch, batch_idx): 
        # training_step defines the train loop, independent of forward
        x, y = batch
        y_hat = self.forward(x)
        y = torch.unsqueeze(y,1)
        loss = self.loss(y_hat, y)
        
        self.log('train_loss', loss) #, on_step=True, on_epoch=True, logger=True
        return loss
    
    def evaluate(self, batch, stage=None):
        """Helper function that generalises the steps for validation_step and test_step.
        Calculates loss & MAE and logs both values.
        Input:
            batch: current batch.
            stage: 'val' for validation_step or 'test' for test_step.
        """
        x, y = batch
        y_hat = self.forward(x)
        y = torch.unsqueeze(y,1)
        loss = self.loss(y_hat, y)
        mae = nn.functional.l1_loss(y_hat, y)
        
        if stage:
            self.log(f'{stage}_loss', loss) #, on_step=True, on_epoch=True, logger=True
            self.log(f'{stage}_mae', mae) #, on_step=True, on_epoch=True, logger=True
            
        if stage == 'val':
            # track validation losses in list
            self.validation_step_outputs.append(loss)
            return {"val_loss": loss, "diff": (y - y_hat), "target": y, 'mae': mae}
    
    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, 'val')
        
    def on_validation_epoch_end(self):
        #val_mean = torch.cat([x['val_loss'] for x in outputs]).mean(dim=0)
        val_mean = torch.stack(self.validation_step_outputs).mean()
        # update best val loss if current loss is smaller
        if val_mean < self.best_val_loss:
            self.best_val_loss = val_mean
        # log current best val loss
        self.log('best_val_loss', self.best_val_loss)
        # clear memory
        self.validation_step_outputs.clear()
    
    def test_step(self, batch, batch_idx):
        self.evaluate(batch, 'test')
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay) 
        return optimizer