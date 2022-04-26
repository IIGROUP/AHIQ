from tqdm import tqdm 
import os
import torch 
import numpy as np
import logging
from scipy.stats import spearmanr, pearsonr
import timm
from timm.models.vision_transformer import Block
from timm.models.resnet import BasicBlock,Bottleneck
import time
from torch.utils.data import DataLoader

from utils.util import setup_seed,set_logging,SaveOutput
from script.extract_feature import get_resnet_feature, get_vit_feature
from options.test_options import TestOptions
from model.deform_regressor import deform_fusion, Pixel_Prediction
from data.pipal import PIPAL
from utils.process_image import ToTensor, RandHorizontalFlip, RandCrop, crop_image, Normalize, five_point_crop
from torchvision import transforms

class Test:
    def __init__(self, config):
        self.opt = config
        self.create_model()
        self.init_saveoutput()
        self.init_data()
        self.load_model()
        self.test()

    def create_model(self):
        self.resnet50 =  timm.create_model('resnet50',pretrained=True).cuda()
        if self.opt.patch_size == 8:
            self.vit = timm.create_model('vit_base_patch8_224',pretrained=True).cuda()
        else:
            self.vit = timm.create_model('vit_base_patch16_224',pretrained=True).cuda()
        self.deform_net = deform_fusion(self.opt).cuda()
        self.regressor = Pixel_Prediction().cuda()

    def init_saveoutput(self):
        self.save_output = SaveOutput()
        hook_handles = []
        for layer in self.resnet50.modules():
            if isinstance(layer, Bottleneck):
                handle = layer.register_forward_hook(self.save_output)
                hook_handles.append(handle)
        for layer in self.vit.modules():
            if isinstance(layer, Block):
                handle = layer.register_forward_hook(self.save_output)
                hook_handles.append(handle)
    
    def init_data(self):
        test_dataset = PIPAL(
            ref_path=self.opt.test_ref_path,
            dis_path=self.opt.test_dis_path,
            txt_file_name=self.opt.test_list,
            resize=self.opt.resize,
            size=(self.opt.size,self.opt.size),
            flip=self.opt.flip,
            transform=ToTensor(),
        )
        logging.info('number of test scenes: {}'.format(len(test_dataset)))

        self.test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=self.opt.batch_size,
            num_workers=self.opt.num_workers,
            drop_last=True,
            shuffle=False
        )

    def load_model(self):
        models_dir = self.opt.checkpoints_dir
        if os.path.exists(models_dir):
            if self.opt.load_epoch == -1:
                load_epoch = 0
                for file in os.listdir(models_dir):
                    if file.startswith("epoch"):
                        load_epoch = max(load_epoch, int(file.split('.')[0].split('_')[1]))
                self.opt.load_epoch = load_epoch
                checkpoint = torch.load(os.path.join(models_dir,"epoch"+str(self.opt.load_epoch)+".pth"))
                self.regressor.load_state_dict(checkpoint['regressor_model_state_dict'])
                self.deform_net.load_state_dict(checkpoint['deform_net_model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                self.start_epoch = checkpoint['epoch']+1
                loss = checkpoint['loss']
            else:
                found = False
                for file in os.listdir(models_dir):
                    if file.startswith("epoch"):
                        found = int(file.split('.')[0].split('_')[1]) == self.opt.load_epoch
                        if found: break
                assert found, 'Model for epoch %i not found' % self.opt.load_epoch
        else:
            assert self.opt.load_epoch < 1, 'Model for epoch %i not found' % self.opt.load_epoch
            self.opt.load_epoch = 0

    def test(self):
        f = open(os.path.join(self.opt.checkpoints_dir,self.opt.test_file_name), 'w')
        with torch.no_grad():
            for data in tqdm(self.test_loader):
                d_img_org = data['d_img_org'].cuda()
                r_img_org = data['r_img_org'].cuda()
                d_img_name = data['d_img_name']
                pred = 0
                for i in range(self.opt.n_ensemble):
                    b, c, h, w = r_img_org.size()
                    if self.opt.n_ensemble > 9:
                        new_h = config.crop_size
                        new_w = config.crop_size
                        top = np.random.randint(0, h - new_h)
                        left = np.random.randint(0, w - new_w)
                        r_img = r_img_org[:,:, top: top+new_h, left: left+new_w]
                        d_img = d_img_org[:,:, top: top+new_h, left: left+new_w]
                    elif self.opt.n_ensemble == 1:
                        r_img = r_img_org
                        d_img = d_img_org
                    else:
                        d_img, r_img = five_point_crop(i, d_img=d_img_org, r_img=r_img_org, config=self.opt)
                    d_img = d_img.cuda()
                    r_img = r_img.cuda()
                    _x = self.vit(d_img)
                    vit_dis = get_vit_feature(self.save_output)
                    self.save_output.outputs.clear()

                    _y = self.vit(r_img)
                    vit_ref = get_vit_feature(self.save_output)
                    self.save_output.outputs.clear()
                    B, N, C = vit_ref.shape
                    if self.opt.patch_size == 8:
                        H,W = 28,28
                    else:
                        H,W = 14,14
                    assert H*W==N 
                    vit_ref = vit_ref.transpose(1, 2).view(B, C, H, W)
                    vit_dis = vit_dis.transpose(1, 2).view(B, C, H, W)

                    _ = self.resnet50(d_img)
                    cnn_dis = get_resnet_feature(self.save_output) 
                    self.save_output.outputs.clear()
                    cnn_dis = self.deform_net(cnn_dis,vit_ref)

                    _ = self.resnet50(r_img)
                    cnn_ref = get_resnet_feature(self.save_output)
                    self.save_output.outputs.clear()
                    cnn_ref = self.deform_net(cnn_ref,vit_ref)
                    pred += self.regressor(vit_dis, vit_ref, cnn_dis, cnn_ref)

                pred /= self.opt.n_ensemble
                for i in range(len(d_img_name)):
                    line = "%s,%f\n" % (d_img_name[i], float(pred.squeeze()[i]))
                    f.write(line)

        f.close()

                

if __name__ == '__main__':
    config = TestOptions().parse()
    config.checkpoints_dir = os.path.join(config.checkpoints_dir, config.name)
    setup_seed(config.seed)
    set_logging(config)
    Test(config)
    