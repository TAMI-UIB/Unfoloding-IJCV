import torch
import lpips

class LossMSElpipsCosineColor(torch.nn.Module):
    def __init__(self, device):
        super(LossMSElpipsCosineColor, self).__init__()

        self.device = device
        self.mse = torch.nn.MSELoss()
        self.loss_lpips = lpips.LPIPS(net='alex')
        self.loss_lpips.to(self.device)
        self.alpha = 0.1
        self.alpha2= 0.1
        self.beta = 0.5


        self.l1 = torch.nn.L1Loss()

    def forward(self, image, estimated):
        lpipsloss = self.loss_lpips.forward(image, estimated)
        LL_t_flatten = torch.flatten(image)
        pred_t_flatten1 = torch.flatten(estimated)

        inner_loss = torch.dot(pred_t_flatten1, LL_t_flatten) / (image.shape[1] * image.shape[2] * image.shape[3])

        mse = self.mse(image, estimated)

        return mse + (self.alpha * inner_loss) + (self.alpha2*lpipsloss)