# from networks import Mygan
# from networks_seg_two import Mygan
from CGAN import CGAN
from torch.utils.data import TensorDataset, DataLoader, Dataset,SubsetRandomSampler
from torchvision import transforms
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr
from datetime import date
import torch
from path import Path
import os
from config import cfg
from torchvision import datasets, transforms

def train(cfg):



    print(cfg)

    train_dataset = datasets.MNIST(root='data', train=True,download=True,
                    transform=transforms.ToTensor())
    train_loader = DataLoader(train_dataset, shuffle=True,
                    batch_size=cfg.BATCH_SIZE)

    data_loader = {'train_loader':train_loader}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    epochs = cfg.EPOCH
    continue_epoch = 0
    
    model = CGAN(  cfg = cfg )

    model.setup() 

    #training
    for epoch in range(continue_epoch + 1,continue_epoch + epochs+1):
        model.isTrain = True
        for index, data in enumerate(tqdm(data_loader['train_loader'])):
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters(epoch)

        model.update_learning_rate(epoch)

    model.save_networks('last')

if __name__ == '__main__':
    train(cfg) 
