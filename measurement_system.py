import torch
import torch.nn.functional as F

class DownsampleModel():
    def __init__(self, scale=4):
        # define afverage pooling layer
        self.avgpool = torch.nn.AvgPool2d(kernel_size=scale, stride=scale)

    def apply_A(self, x):
        """
        X: torch.Tensor of shape (batch_size, 3, 256, 256)
        """
        y = self.avgpool(x)
        return y

if __name__ == '__main__':
    # Test the DownsampleModel
    model = DownsampleModel()
    X = torch.ones(10, 3, 256, 256)
    Y = model.apply_A(X)
    print(Y.shape) # Should be torch.Size([10, 3, 16, 16])
    print(Y[0, 0, 0, 0]) # Should be 1.0