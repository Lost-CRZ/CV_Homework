import torch
import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self):
        """
        Init function to define the layers and loss function

        Note: Use 'mean' reduction in the loss_criterion. Read Pytorch documention to understand what it means

        """
        super(SimpleNet, self).__init__()

        self.conv_layers = nn.Sequential()
        self.fc_layers = nn.Sequential()
        self.loss_criterion = None

        ############################################################################
        # Student code begin
        ############################################################################
        
        # Define convolutional layers for grayscale images to abstract features
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

        )

        # Define fully connected layers to perform classification
        self.fc_layers = nn.Sequential(
            nn.Linear(in_features=32 * 16 * 16, out_features=128),  # Adjust for flattened size
            nn.ReLU(),
            #nn.Dropout(0.5),  # 50% dropout
            nn.Linear(in_features=128, out_features=15)  # 15 classes for classification
        )

        # Define the loss function with 'mean' reduction
        self.loss_criterion = nn.CrossEntropyLoss(reduction='mean')

        # raise NotImplementedError(
        #     "`__init__` function in "
        #     + "`simple_net.py` needs to be implemented"
        # )

        ############################################################################
        # Student code end
        ############################################################################

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform the forward pass with the net

        Args:
        -   x: the input image [Dim: (N,C,H,W)]
        Returns:
        -   y: the output (raw scores) of the net [Dim: (N,15)]
        """
        model_output = None
        ############################################################################
        # Student code begin
        ############################################################################
        
        #print(x.shape)

        # Forward pass through conv layers
        x = self.conv_layers(x)
        
        # Flatten the output for the fully connected layers
        x = x.view(x.size(0), -1)  # convert from [N,C,H,W]-->[N,C*H*W]
        
        # Forward pass through fully connected layers
        model_output = self.fc_layers(x)

        return model_output
        raise NotImplementedError(
            "`forward` function in "
            + "`simple_net.py` needs to be implemented"
        )

        ############################################################################
        # Student code end
        ############################################################################

        
