
import torch
import torch.nn as nn
import torch.nn.functional as nnf




class resenet_block(nn.Module):
    def __init__(self,channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels,channels,3,padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels,channels,3,padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self,x):
        y = self.conv1(x)
        y = nnf.relu(self.bn1(y))
        y = self.conv2(y)
        y = nnf.relu(self.bn2(x+y))
        return y
        


class Resnet(nn.Module):
    def __init__(self, in_channels, output_dimension=1,downscale_architecture='resnet34'):
    # in_channels - no. of channels in the image input
    # output_dimensions - no. of outputs
    # downscale_architecture - it is a list of integers, where each integer represents the number of resnet blocks before a downscale
    #                          happens. Length of this list represents the number of downscales that happen.
    #                          
    
        super().__init__()
        self.downscale_architecture = downscale_architecture
        self.model_layers = []
        if self.downscale_architecture == 'resnet34':
            self.downscale_architecture = [3,3,5,3]
            
        self.model_layers.append(nn.Conv2d(in_channels,64,7,padding=3))
        self.model_layers.append(nn.BatchNorm2d(64))
        self.model_layers.append(nn.ReLU())
        #add relu
        self.model_layers.append(nn.MaxPool2d(3,stride=2,padding=1))
        in_channels = 64



        for count,resnet_section_length in enumerate(self.downscale_architecture):
            for i in range(resnet_section_length):
                self.model_layers.append(resenet_block(in_channels))
            if count < (len(self.downscale_architecture)-1):
                self.model_layers.append(nn.Conv2d(in_channels,in_channels*2,kernel_size=3,stride=2,padding=1))
                in_channels = in_channels*2
                self.model_layers.append(nn.BatchNorm2d(in_channels))
                self.model_layers.append(nn.ReLU())
                self.model_layers.append(nn.Conv2d(in_channels,in_channels,kernel_size=3,stride=1,padding=1))
                self.model_layers.append(nn.BatchNorm2d(in_channels))
                self.model_layers.append(nn.ReLU())

        self.model_layers.append(nn.AdaptiveAvgPool2d((1,1)))
        #add flatten
        self.model_layers.append(nn.Flatten())
        self.model_layers.append(nn.Linear(in_channels,1000))
        self.model_layers.append(nn.ReLU())
        self.model_layers.append(nn.Linear(1000,output_dimension))

        self.model = nn.Sequential(*self.model_layers)

    def forward(self,x):
        return self.model(x)


    def model_train(self,dataloader,epochs,lr=2e-4):
        self.model.train()
        criterion_loss = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr = lr)

        for i, data in enumerate(dataloader):

            x = data['x']
            y = data['y']

            self.model.zero_grad()

            y_pred = self.model(x)
            loss = criterion_loss(y_pred,y)

            loss.backward()
            optimizer.step()