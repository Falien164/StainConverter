import torch
import os

from torch.autograd import Variable


class Pix2Pix:
    def __init__(self, generator, discriminator):
        cuda = bool(os.environ.get('CUDA'))
        lr = float(os.environ.get('LR'))
        beta = (0.5, 0.999)

        self.generator = generator.cuda()
        self.discriminator = discriminator.cuda()

        self.optimizer_gen = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=beta)
        self.optimizer_disc_b = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=beta)

        self.mse_loss = torch.nn.MSELoss()
        self.l1_loss = torch.nn.L1Loss()
        self.adv_lamda = 1
        self.cycle_lambda = 10

        tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
        self.target_real = Variable(tensor(1, 1).fill_(1.0), requires_grad=False)
        self.target_fake = Variable(tensor(1, 1).fill_(0.0), requires_grad=False)

    def train_step(self, batch):
        # Set model input

        real_a = Variable(batch['HE']).cuda()
        real_b = Variable(batch['PAS']).cuda()

        self.optimizer_gen.zero_grad()

        # GAN loss
        fake_b = self.generator(real_a)
        loss_l1 = self.l1_loss(fake_b, real_b) * self.adv_lamda

        # gan loss:
        fake_ab = torch.cat((real_a, fake_b), 1)
        pred_fake = self.discriminator(fake_ab)
        loss_gen_a2b = self.mse_loss(pred_fake, self.target_real) * self.adv_lamda

        # Total loss
        total_loss = loss_l1 + loss_gen_a2b
        total_loss.backward()
        self.optimizer_gen.step()

        self.optimizer_disc_b.zero_grad()
        with torch.no_grad():
            fake_b = self.generator(real_a)
        pred_fake = self.discriminator(torch.cat((real_a, fake_b), 1)) * self.adv_lamda
        pred_real = self.discriminator(torch.cat((real_a, real_b), 1)) * self.adv_lamda
        loss_disc_b = self.mse_loss(pred_fake, self.target_fake) + self.mse_loss(pred_real, self.target_real)

        loss_disc_b.backward()
        self.optimizer_disc_b.step()

    def predict(self, img):
        if torch.is_tensor(img):
            return self.generator(img.cuda()).detach().cpu().numpy()
        else:
            return self.generator(torch.from_numpy(img).cuda()).detach().cpu().numpy()
