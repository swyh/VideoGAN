from torch.autograd import Variable
import torch
import torch.nn as nn

def get_gan_loss(dis_real, dis_fake, criterion):
    lables_dis_real = Variable(torch.ones([dis_real.size()[0], 1]))
    lables_dis_fake = Variable(torch.zeros([dis_fake.size()[0], 1]))
    lables_gen = Variable(torch.ones([dis_fake.size()[0], 1]))


    #print(lables_dis_real.shape)

    if torch.cuda:
        lables_dis_real = lables_dis_real.cuda()
        lables_dis_fake = lables_dis_fake.cuda()
        lables_gen = lables_gen.cuda()

    dis_loss = criterion(dis_real, lables_dis_real) * 0.5 + criterion(dis_fake, lables_dis_fake) * 0.5
    gen_loss = criterion(dis_fake, lables_gen)

    return dis_loss, gen_loss


def get_mse_loss(gen_real, gen_fake, criterion):
    #loss = torch.sum(torch.abs(gen_real, gen_fake))

    loss = criterion(gen_fake, gen_real)

    return loss