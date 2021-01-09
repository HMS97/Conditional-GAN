
import torch

from model.GD import Discriminator
from model.GD import Generator
from torchvision import transforms
from torch import nn
from torch.nn import init
from torch.optim import lr_scheduler
import neptune
import numpy as np
import os
from path import Path
import pathlib
from PIL import Image
from torchvision.utils import save_image




def get_scheduler(optimizer, lr_policy, step_size):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 - n_epochs) / float(100 + 1)
            return lr_l
                       
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.9)
    elif lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', lr_policy)
    return scheduler


class CGAN(nn.Module):
    
    def __init__(self,  cfg ):
        super(CGAN,self).__init__()
        self.shuffle = 'train'
        self.cfg = cfg
        self.iteration = 0
        self.start_epoch = 0        
        self.count_times = 0
        self.iou_value = 0
        
        self.device = torch.device('cuda:{}'.format(self.cfg.GPU_IDS[0])) if self.cfg.GPU_IDS else torch.device('cpu')  # get device name: CPU or GPU

        self.optimizers = []
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G', 'D_real','D_fake' ]
        self.suffix = 'predict_images'

        self.model_names = ['G', 'D']

        # define networks (both generator and discriminator)

        self.netG = Generator(self.cfg.NZ).to(self.device)
        self.netD = Discriminator().to(self.device)
        # self.netG = init_net(BBnet(in1_chs=self.cfg.NET.GEN_IN1,in1_out=self.cfg.NET.GEN_OUT1,in2_chs=self.cfg.NET.GEN_IN2,in2_out=self.cfg.NET.GEN_OUT2), init_type='normal', init_gain=0.02, gpu_ids = self.cfg.GPU_IDS)

        # self.netD = define_D(input_nc = self.cfg.NET.D1_IN, ndf = self.cfg.NET.D1_NDF, netD = self.cfg.NET.D1_TYPE, gpu_ids = self.cfg.GPU_IDS )
        self.noise = torch.FloatTensor(self.cfg.BATCH_SIZE, (self.cfg.NZ)).to(self.device)

        # define loss functions
        self.criterion= torch.nn.BCELoss()
        # # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
        self.optimizer_G = torch.optim.SGD(self.netG.parameters(), lr=self.cfg.LR)
        self.optimizer_D = torch.optim.SGD(self.netD.parameters(), lr=self.cfg.LR)



        self.optimizers.append(self.optimizer_G)
        self.optimizers.append(self.optimizer_D)

        SAMPLE_SIZE = 80
        NUM_LABELS = 10
        self.fixed_noise = torch.FloatTensor(SAMPLE_SIZE, self.cfg.NZ).normal_(0,1).to(self.device)
        self.fixed_labels = torch.zeros(SAMPLE_SIZE, NUM_LABELS).to(self.device)
        for i in range(NUM_LABELS):
            for j in range(SAMPLE_SIZE // NUM_LABELS):
                self.fixed_labels[i*(SAMPLE_SIZE // NUM_LABELS) + j, i] = 1.0


        for name in self.loss_names:
            if isinstance(name, str):
                setattr(self, 'loss_' + name, 0)
                setattr(self,  name, 0)

      


    def record_loss(self):
        """ update loss according train and valid status"""
        if self.iteration%self.cfg.SHOW_INTERVAL == 0:
            pathlib.Path(os.path.join(self.cfg.SAVE_SAMPLE )).mkdir(parents=True, exist_ok=True) 

            g_out = self.netG(self.fixed_noise, self.fixed_labels).data.view( 80, 1, 28,28).cpu()   
            save_image(g_out,f'{self.cfg.SAVE_SAMPLE}/{self.iteration}_fixed.png')
    
            # for name in self.loss_names:
            # print((f'{self.shuffle}_loss_' + name, self.iteration, getattr(self,'loss_'+name)))


    def setup(self ):
        """Load and print networks; create schedulers

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """

        if self.cfg.ISTRAIN:
            self.schedulers = [get_scheduler(optimizer, self.cfg.LR_POLICY, self.cfg.STEP_DEC) for optimizer in self.optimizers]
      

    def save_networks(self, which_epoch):
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '%s_net_%s.pth' % (which_epoch, name)
                save_path = os.path.join(self.cfg.SAVE_DIR , save_filename).replace('\\', '/')
                net = getattr(self, 'net' + name)
                optimize = getattr(self, 'optimizer_' + name)
                pathlib.Path(os.path.join(self.cfg.DIR.SAVE_DIR )).mkdir(parents=True, exist_ok=True) 

                if len(self.cfg.GPU_IDS) > 0 and torch.cuda.is_available():
                    torch.save({'net': net.module.cpu().state_dict(), 'optimize': optimize.state_dict()}, save_path)
                    net.cuda(self.cfg.GPU_IDS[0])
                else:
                    torch.save(net.cpu().state_dict(), save_path)

    # helper loading function that can be used by subclasses
    def load_networks(self, which_epoch):
        for name in self.model_names:
            if isinstance(name, str):
                load_filename = '%s_net_%s.pth' % (which_epoch, name)
                load_path = os.path.join(self.cfg.SAVE_DIR , load_filename)

                net = getattr(self, 'net' + name)
                optimize = getattr(self, 'optimizer_' + name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                # if you are using PyTorch newer than 0.4 (e.g., built from
                # GitHub source), you can remove str() on self.device
                state_dict = torch.load(load_path.replace('\\', '/'), map_location=str(self.device))
                optimize.load_state_dict(state_dict['optimize'])
                net.load_state_dict(state_dict['net'])



    def eval(self):
        """Make models eval mode during test time"""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.eval()

    def train(self):
        """Make models eval mode during train time"""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.train()
     




    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        self.real_label = 1
        self.fake_label = 0

        self.input_x = input[0].view(-1, 784).to(self.device)
        self.input_y = input[1].to(self.device)


        self.one_hot_labels = torch.FloatTensor(self.cfg.BATCH_SIZE, self.cfg.NUM_LABELS).to(self.device)
        self.label = torch.FloatTensor(self.cfg.BATCH_SIZE).to(self.device)





    def get_one_hot_label(self, rand = False):

        self.noise.resize_(self.cfg.BATCH_SIZE, self.cfg.NZ).normal_(0,1)

        if rand:
            self.one_hot_labels.zero_()
            rand_y = torch.from_numpy(
                np.random.randint(0, self.cfg.NUM_LABELS, size=(self.cfg.BATCH_SIZE,1))).cuda()
            self.one_hot_labels.scatter_(1, rand_y.view(self.cfg.BATCH_SIZE,1), 1)
            self.label.resize_(self.cfg.BATCH_SIZE).fill_(self.fake_label)
        else:       
            self.one_hot_labels.zero_()
            self.one_hot_labels.scatter_(1, self.input_y.view(self.cfg.BATCH_SIZE,1), 1)
            self.label.resize_(self.cfg.BATCH_SIZE).fill_(self.real_label)


    def backward_D(self):
        """Calculate GAN loss for the discriminator"""


        self.get_one_hot_label(rand = False)

        output = self.netD( self.input_x, self.one_hot_labels)
        self.loss_D_real = self.criterion(output.squeeze(1), self.label)
        self.loss_D_real.backward()

        ## fake discriminator
        self.get_one_hot_label(rand = True)

        g_out = self.netG(self.noise, self.one_hot_labels)
        output = self.netD(g_out, self.one_hot_labels)
        self.loss_D_fake = self.criterion(output.squeeze(1), self.label)

        self.loss_D_fake.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""

        self.get_one_hot_label(rand = False)
        self.label.resize_(self.cfg.BATCH_SIZE).fill_(self.real_label)

        g_out = self.netG(self.noise, self.one_hot_labels)
        output = self.netD(g_out, self.one_hot_labels)
        self.loss_G = self.criterion(output.squeeze(1), self.label)
        # combine loss and calculate gradients
        self.loss_G.backward()




    def optimize_parameters(self,epoch):
        if self.cfg.ISTRAIN!=True:
            with torch.no_grad():
                self.eval()
                self.forward()                   # compute fake images: G(A)
        else:
            self.train()
            # update D
            self.set_requires_grad(self.netD, True)  # enable backprop for D
            self.optimizer_D.zero_grad()     # set D's gradients to zero
            self.backward_D()                # calculate gradients for D
            self.optimizer_D.step()          # update D's weights

            # update G
            self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
            self.optimizer_G.zero_grad()        # set G's gradients to zero
            self.backward_G()                   # calculate graidents for G
            self.optimizer_G.step()# udpate G's weights

        self.iteration += 1
        self.record_loss()


    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad


    def update_learning_rate(self, epoch ):
        """Update learning rates for all the networks; called at the end of every epoch"""
        old_lr = self.optimizers[0].param_groups[0]['lr']

        for scheduler in self.schedulers:
            if self.cfg.LR_POLICY == 'plateau':
                scheduler.step(self.metric)
            else:
                scheduler.step()

        lr = self.optimizers[0].param_groups[0]['lr']

        print(f'current epoch {epoch}  learning rate %.7f -> %.7f' % (old_lr, lr))


