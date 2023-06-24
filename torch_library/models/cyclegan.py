import torch
import copy
import itertools
import os

from torch.autograd import Variable


class CycleGAN:
    def __init__(self, generator, discriminator):
        lr = float(os.environ.get('LR'))
        cuda = bool(os.environ.get('CUDA'))
        beta = (0.5, 0.999)

        self.generator_A2B = copy.deepcopy(generator).cuda()
        self.generator_B2A = generator.cuda()

        self.discriminator_B = copy.deepcopy(discriminator).cuda()
        self.discriminator_A = discriminator.cuda()

        self.optimizer_G = torch.optim.Adam(
            itertools.chain(self.generator_A2B.parameters(), self.generator_B2A.parameters()),
            lr=lr, betas=beta)
        self.optimizer_D_B = torch.optim.Adam(self.discriminator_B.parameters(), lr=lr, betas=beta)
        self.optimizer_D_A = torch.optim.Adam(self.discriminator_A.parameters(), lr=lr, betas=beta)

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

        fake_a, fake_b = self.loss_gen(real_a, real_b)

        self.loss_disc_a(real_a, fake_a)
        self.loss_disc_b(real_b, fake_b)

    def loss_gen(self, real_a, real_b):
        self.optimizer_G.zero_grad()

        # GAN loss
        fake_b = self.generator_A2B(real_a)
        pred_fake = self.discriminator_B(fake_b)
        loss_gen_a2b = self.adv_lamda * self.mse_loss(pred_fake, self.target_real)

        fake_a = self.generator_B2A(real_b)
        pred_fake = self.discriminator_A(fake_a)
        loss_gen_b2a = self.adv_lamda * self.mse_loss(pred_fake, self.target_real)

        # Cycle loss
        recovered_a = self.generator_B2A(fake_b)
        loss_cycle_aba = self.cycle_lambda * self.l1_loss(recovered_a, real_a)

        recovered_b = self.generator_A2B(fake_a)
        loss_cycle_bab = self.cycle_lambda * self.l1_loss(recovered_b, real_b)

        # Total loss
        loss_total = loss_gen_a2b + loss_gen_b2a + loss_cycle_aba + loss_cycle_bab
        loss_total.backward()
        self.optimizer_G.step()

        return fake_a, fake_b

    def loss_disc_a(self, real_a, fake_a):
        self.optimizer_D_A.zero_grad()
        # Real loss
        pred_real = self.discriminator_A(real_a)
        loss_disc_real = self.adv_lamda * self.mse_loss(pred_real, self.target_real)
        # Fake loss

        pred_fake = self.discriminator_A(fake_a.detach())
        loss_disc_fake = self.adv_lamda * self.mse_loss(pred_fake, self.target_fake)

        # Total loss
        loss_disc_a = (loss_disc_real + loss_disc_fake)
        loss_disc_a.backward()

        self.optimizer_D_A.step()

    def loss_disc_b(self, real_b, fake_b):
        self.optimizer_D_B.zero_grad()

        # Real loss
        pred_real = self.discriminator_B(real_b)
        loss_disc_real = self.adv_lamda * self.mse_loss(pred_real, self.target_real)

        # Fake loss
        pred_fake = self.discriminator_B(fake_b)
        loss_disc_fake = self.adv_lamda * self.mse_loss(pred_fake, self.target_fake)

        # Total loss
        loss_disc_b = (loss_disc_real + loss_disc_fake)
        loss_disc_b.backward()

        self.optimizer_D_B.step()

    def predict(self, img):
        return self.generator_A2B(torch.from_numpy(img).cuda()).detach().cpu().numpy()
