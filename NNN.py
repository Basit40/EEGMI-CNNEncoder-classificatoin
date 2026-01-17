import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
import numpy as np
#_____________________________________________________________________________________________________
class SENet(nn.Module):
    def __init__(self, channel, reduction=2):
        super(SENet, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ ,_= x.size()
        y = self.avg_pool(x).view(b, c)
        #y=torch.mean(x,dim=(2,3))
        y = self.fc(y).view(b, c, 1,1)
        return x * y.expand_as(x)
#____________________________________________________________________________________________________________
class TemporalFilteringModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):

        super(TemporalFilteringModule, self).__init__()
        self.kernel_size=kernel_size
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, kernel_size),padding='same',groups=in_channels)
        self.senet = SENet(out_channels)
        self.bn = nn.BatchNorm2d(out_channels)
        self.elu = nn.ELU()


    def forward(self, x):
        # x=x.unsqueeze(1)
        #print('===rrrrrr==',x.shape)
        x = self.conv(x)
        #print('===rrrrrr==',x.shape)
        x = self.senet(x)
        x = self.bn(x)
        x = self.elu(x)
        return x
#________________________________________________________________________________________________________
class SpatialFilteringModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(SpatialFilteringModule, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1),padding='valid')
        self.senet = SENet(out_channels)
        self.bn = nn.BatchNorm2d(out_channels)
        self.elu = nn.ELU()
        self.avg_pool = nn.AvgPool2d(kernel_size=(1,8))
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.conv(x)
        x = self.senet(x)
        x = self.bn(x)
        x = self.elu(x)
        x = self.avg_pool(x)
        x = self.dropout(x)
        return x
#__________________________________________________________________________________________________________
class FeatureCompressionModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(FeatureCompressionModule, self).__init__()
        self.kernel_size=kernel_size
        self.depthwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size=( 1,kernel_size), groups=in_channels,padding='same')#
        self.pointwise_conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1),padding='same')
        self.senet = SENet(out_channels)
        self.bn = nn.BatchNorm2d(out_channels)
        self.elu = nn.ELU()
        self.avg_pool = nn.AvgPool2d(kernel_size=(1, 16))
        self.dropout = nn.Dropout(0.2)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        x = self.senet(x)
        x = self.bn(x)
        x = self.elu(x)
        x = self.avg_pool(x)
        x = self.dropout(x)
        x = self.flatten(x)
        #print("================",x.shape)
        
        return x
#__________________________________________________________________________________________________
class Classifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Classifier, self).__init__()

        #self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.elu = nn.ELU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.sftmx=nn.Softmax(dim=1)  # Softmax layer

    def forward(self, x):
        #x = self.flatten(x)
        #print('=====classifier input===',x.shape)
        x = self.fc1(x)
        x = self.elu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.sftmx(x)  # Softmax layer
        return x
#______________________________________________________________________________________________________
class SSCLNet(nn.Module):
    in_channels=1
    num_classes=2    #------------- change according to dataset
    num_electrods=3  #------------- change according to dataset
    def __init__(self,  time_steps,classification=False):
        super(SSCLNet, self).__init__()   

        self.classification=classification 
        self.time_steps=time_steps   
        self.temporal_filtering = TemporalFilteringModule(self.in_channels, 8, 128)
        self.spatial_filtering = SpatialFilteringModule(8, 16,self.num_electrods )
        self.feature_compression = FeatureCompressionModule(16, 16, 32)
        with torch.no_grad():
            dummy_input=torch.randn(1,1,self.num_electrods,time_steps)
            dummy_output=self.feature_compression(self.spatial_filtering(self.temporal_filtering(dummy_input)))
            input_dim=dummy_output.numel()
        self.classifier = Classifier(input_dim, 32, self.num_classes)

    def forward(self, x):
        x=x.unsqueeze(1)
        #print('=>>>>>>>>>> timepoitn',x.shape)
        x = self.temporal_filtering(x)
        #print('=>>>>>>>>>> timepoitn',x.shape)        
        x = self.spatial_filtering(x)
        x = self.feature_compression(x)

        if self.classification:
            x = self.classifier(x)
        return x

#____________________________________________________________________________________________
def ssl(time_steps,**kwargs):
   
    model = SSCLNet(time_steps,**kwargs) 
    #print('=>>>>>>>>>> timepoitn',time_steps)
    #summary(model,input_size=(512,22,time_steps))
    #ooo
    return model
#_____________________________________________________________________________________________
def test():
    # Assuming input EEG data has shape (batch_size, channels, time_steps)
    batch_size, channels, time_steps = 512, 22, 1301
    # input_data = torch.randn(batch_size, channels, time_steps).unsqueeze(1)  # Add channel dimension
    # print('=====',input_data.shape)
    x=torch.randn(batch_size,channels,time_steps).to('cuda')
    print('===x==',x.shape)
    net=ssl(time_steps,classification=True).to('cuda')
    y=net(x).to('cuda')
    print('====y==',y.shape)   
    # model = EEGNet(1, time_steps, num_classes=4)
    # output = model(input_data)
    # print("Output shape:", output.shape)
#_________________________________________________________________________________________________


    
# Example usage
if __name__ == "__main__":
    #print('...............',(int(np.ceil(1000/128)))*16)
    test()