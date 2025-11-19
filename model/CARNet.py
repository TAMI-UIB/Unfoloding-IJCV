import torch
import torch.nn as nn

from utils.layers import ResNetU, NLBPUNetFormer, MultiHeadAttention

class CARNet(nn.Module):
    def __init__(self, channels, batch_size, **kwargs):
        super(CARNet, self).__init__()
        self.channels = channels
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.tau = nn.Parameter(torch.tensor(0.1))
        self.beta = nn.Parameter(torch.tensor(0.5))
        self.mu = nn.Parameter(torch.tensor(0.05))
        self.lambdaN = nn.Parameter(torch.tensor(0.05))
        self.stages = kwargs['stages']
        self.ResNet_L = nn.ModuleList([ResNetU(channels=1, features=32, index=i, stages= self.stages, num_blocks=3) for i in range(self.stages)])
        self.ResNet_R = nn.ModuleList([NLBPUNetFormer(hs_channels=self.channels,
                                                 features=32, patch_size=4, kernel_size=3) for i in range(self.stages)])
        self.nonlocalfidelity = nn.ModuleList([MultiHeadAttention(hs_channels=2*self.channels, patch_size=4, features=32) for _ in
              range(self.stages)])

    def split_gradient(self, u):
        dx = torch.zeros_like(u)
        dy = torch.zeros_like(u)
        dy[:, :, :-1, :] = u[:, :, 1:, :] - u[:, :, :-1, :]
        dx[:, :, :, :-1] = u[:, :, :, 1:] - u[:, :, :, :-1]
        return dy, dx

    def div(self, v):
        vy, vx = v[:, 0:3, :, :], v[:, 3:6, :, :]
        div = torch.zeros_like(vx)

        div[:, :, 1:-1, 1:-1] = (vx[:, :, 1:-1, 1:-1] - vx[:, :, 1:-1, :-2]) + (
                    vy[:, :, 1:-1, 1:-1] - vy[:, :, :-2, 1:-1])

        div[:, :, 0, 1:-1] = vx[:, :, 0, 1:-1] - vx[:, :, 0, :-2] + vy[:, :, 0, 1:-1]
        div[:, :, -1, 1:-1] = vx[:, :, -1, 1:-1] - vx[:, :, -1, :-2] - vy[:, :, -2, 1:-1]

        div[:, :, 1:-1, 0] = vx[:, :, 1:-1, 0] + vy[:, :, 1:-1, 0] - vy[:, :, :-2, 0]
        div[:, :, 1:-1, -1] = -vx[:, :, 1:-1, -2] + vy[:, :, 1:-1, -1] - vy[:, :, :-2, -1]

        div[:, :, 0, 0] = vx[:, :, 0, 0] + vy[:, :, 0, 0]
        div[:, :, 0, -1] = -vx[:, :, 0, -2] + vy[:, :, 0, -1]
        div[:, :, -1, 0] = vx[:, :, -1, 0] - vy[:, :, -2, 0]
        div[:, :, -1, -1] = -vx[:, :, -1, -2] - vy[:, :, -2, -1]

        return div
    def forward(self, image, L0, R0):
        N0= torch.zeros_like(R0)
        L_aux = L0
        R_aux = R0

        N_aux=N0



        for i in range(self.stages):
            dyR, dxR = self.split_gradient(R_aux)
            nablaR = torch.cat([dyR, dxR], dim=1)

            dyI, dxI = self.split_gradient(image)
            nablaI = torch.cat([dyI, dxI], dim=1)

            I_nl = self.nonlocalfidelity[i](nablaI, nablaR)
            argdiv= nablaR-I_nl
            upd_R = self.ResNet_R[i](R_aux - self.tau * self.beta * (L_aux*((R_aux*L_aux)+N_aux - image)-(self.mu * self.div(argdiv))), image)

            upd_L = self.ResNet_L[i](L_aux-(self.tau*self.alpha*torch.sum(upd_R, dim=1, keepdim=True))*
                                     ((L_aux*torch.sum(upd_R, dim=1, keepdim=True))+torch.sum(N_aux, dim=1, keepdim=True)-torch.sum(image, dim=1, keepdim=True)))




            upd_N = (image - (upd_L * upd_R)) / (1 + self.lambdaN)

            L_aux= upd_L
            R_aux= upd_R
            N_aux=upd_N

        return [upd_L, upd_R]