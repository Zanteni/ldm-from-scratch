import torch
import torch.nn as nn
import torch.nn.functional as F
import math
class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        # TODO: store in_channels, out_channels, kernel_size, stride, padding
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
       
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter("bias", None)

        # TODO: create weight parameter, shape (out_channels, in_channels, kernel_size, kernel_size)
        fan_in = in_channels * kernel_size * kernel_size
        self.weight = nn.Parameter(torch.randn(out_channels,in_channels,kernel_size,kernel_size)/ math.sqrt(fan_in))
        
    def forward(self,x):
        assert x.ndim == 3 or x.ndim == 4,f"the input should be a 3D or 4D.got{x.ndim}."
        x = F.pad(x,(self.padding,self.padding,self.padding,self.padding))
        
        if x.ndim == 3:
            B = 1
            C_in, H, W = x.shape
        else:
            B, C_in, H, W = x.shape
        H_out = (H-self.kernel_size)//self.stride+1
        W_out = (W-self.kernel_size)//self.stride+1
        patches = F.unfold(x, kernel_size=self.kernel_size, stride=self.stride)
        weight_flat = self.weight.view(self.out_channels, -1)
        out = weight_flat.unsqueeze(0) @ patches
        out = out.reshape(B, self.out_channels, H_out, W_out)
        if self.bias is not None:
            bias = self.bias.reshape(1,self.out_channels,1,1)
            out = out+bias
        return out