import torch
import torch.nn as nn


class DistrLoss(nn.Module):

    def __init__(self, channels):
        super(DistrLoss, self).__init__()
        self._channels = channels

    def forward(self, input):
        if input.dim() != 4 and input.dim() != 3 and input.dim() != 2:
            raise ValueError('expected 4D, 3D or 2D input (got {}D input)'
                             .format(input.dim()))
        if input.size()[1] != self._channels:
            raise ValueError('expected {} channels (got {}D input)'
                             .format(self._channels, input.size()[1]))

        if input.dim() == 4:
            mean = input.mean(dim=-1).mean(dim=-1).mean(dim=0)
            var = ((input - mean.unsqueeze(0).unsqueeze(2).unsqueeze(3)) ** 2
                   ).mean(dim=-1).mean(dim=-1).mean(dim=0)
        elif input.dim() == 3:
            mean = input.mean(dim=-1).mean(dim=0)
            var = ((input - mean.unsqueeze(0).unsqueeze(2)) ** 2
                   ).mean(dim=-1).mean(dim=0)
        elif input.dim() == 2:
            mean = input.mean(dim=0)
            var = ((input - mean.unsqueeze(0)) ** 2).mean(dim=0)

        var = var + 1e-10  # to avoid 0 variance
        std = var.abs().sqrt()

        distrloss1 = (torch.min(1 - mean - 0.25*std, 0 + mean - 0.25 *
                                std).clamp(min=0) ** 2).mean() + ((std - 4).clamp(min=0) ** 2).mean()  # Gradient mismatch + Saturation
        distrloss2 = ((mean-0.5) ** 2 - std **
                      2).clamp(min=0).mean()  # Degeneration

        return [distrloss1, distrloss2]

class SALoss(nn.Module):

    def __init__(self, channels, offset=-4, loss_interval=10):
        super(SALoss,self).__init__()
        self._channels = channels
        self.loss_interval = loss_interval # cells interval for loss calculation, usually use sigma * 3
        self.offset = offset

    def forward(self, input):
        if input.dim() != 4 and input.dim() != 3 and input.dim() != 2:
            raise ValueError('expected 4D, 3D or 2D input (got {}D input)'
                             .format(input.dim()))
        if input.size()[1] != self._channels:
            raise ValueError('expected {} channels (got {}D input)'
                             .format(self._channels, input.size()[1]))
        
        # V1
        #SALoss = (((self.loss_interval - input.abs()).clamp(min=0) ** 2).mean(dim=-1).mean(dim=-1).mean(dim=0).mean(dim=0)).sqrt()
        # V2
        #SALoss = ((((self.loss_interval - input.abs()) / self.loss_interval).clamp(min=0) ** 2).mean(dim=-1).mean(dim=-1).mean(dim=0).mean(dim=0)).sqrt()
        # V3
        SALoss = ((((self.loss_interval - input.abs()) / self.loss_interval).clamp(min=0) ** 2).mean(dim=-1).mean(dim=-1).mean(dim=0).mean(dim=0))
        # V4
        #SALoss = ((((self.loss_interval - (input-self.offset).abs()) / self.loss_interval).clamp(min=0) ** 2).mean(dim=-1).mean(dim=-1).mean(dim=0).mean(dim=0))
        return SALoss
    
    def extra_repr(self):
        s = ('offset={offset}, loss_interval={loss_interval}')
        return s.format(**self.__dict__)
    
    def set_loss_interval(self, loss_interval):
        print('---setting SA loss interval to %d cells---' % loss_interval)
        self.loss_interval = loss_interval