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
from options.train_options import TrainOptions
from model.deform_regressor import deform_fusion, Pixel_Prediction
from data.pipal import PIPAL
from utils.process_image import ToTensor, RandHorizontalFlip, RandCrop, crop_image, Normalize, five_point_crop
from torchvision import transforms

class Train:
    def __init__(self, config):
        self.opt = config
        self.create_model()
        self.init_saveoutput()
        self.init_data()
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam([
        {'params': self.regressor.parameters(), 'lr': self.opt.learning_rate,'weight_decay':self.opt.weight_decay}, 
        {'params': self.deform_net.parameters(),'lr': self.opt.learning_rate,'weight_decay':self.opt.weight_decay}
        ])
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.opt.T_max, eta_min=self.opt.eta_min)
        self.load_model()
        self.train()

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
        train_dataset = PIPAL(
            ref_path=self.opt.train_ref_path,
            dis_path=self.opt.train_dis_path,
            txt_file_name=self.opt.train_list,
            transform=transforms.Compose(
                [
                    RandCrop(self.opt.crop_size, self.opt.num_crop),
                    #Normalize(0.5, 0.5),
                    RandHorizontalFlip(),
                    ToTensor(),
                ]
            ),
        )
        val_dataset = PIPAL(
            ref_path=self.opt.val_ref_path,
            dis_path=self.opt.val_dis_path,
            txt_file_name=self.opt.val_list,
            transform=ToTensor(),
        )
        logging.info('number of train scenes: {}'.format(len(train_dataset)))
        logging.info('number of val scenes: {}'.format(len(val_dataset)))

        self.train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=self.opt.batch_size,
            num_workers=self.opt.num_workers,
            drop_last=True,
            shuffle=True
        )
        self.val_loader = DataLoader(
            dataset=val_dataset,
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
                    if file.startswith("epoch_"):
                        load_epoch = max(load_epoch, int(file.split('.')[0].split('_')[1]))
                self.opt.load_epoch = load_epoch
                checkpoint = torch.load(os.path.join(models_dir,"epoch_"+str(self.opt.load_epoch)+".pth"))
                self.regressor.load_state_dict(checkpoint['regressor_model_state_dict'])
                self.deform_net.load_state_dict(checkpoint['deform_net_model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                self.start_epoch = checkpoint['epoch']+1
                loss = checkpoint['loss']
            else:
                found = False
                for file in os.listdir(models_dir):
                    if file.startswith("epoch_"):
                        found = int(file.split('.')[0].split('_')[1]) == self.opt.load_epoch
                        if found: break
                assert found, 'Model for epoch %i not found' % self.opt.load_epoch
        else:
            assert self.opt.load_epoch < 1, 'Model for epoch %i not found' % self.opt.load_epoch
            self.opt.load_epoch = 0

    def train_epoch(self, epoch):
        losses = []
        self.regressor.train()
        self.deform_net.train()
        self.vit.eval()
        self.resnet50.eval()
        # save data for one epoch
        pred_epoch = []
        labels_epoch = []
        
        for data in tqdm(self.train_loader):
            d_img_org = data['d_img_org'].cuda()
            r_img_org = data['r_img_org'].cuda()
            labels = data['score']
            labels = torch.squeeze(labels.type(torch.FloatTensor)).cuda()

            _x = self.vit(d_img_org)
            vit_dis = get_vit_feature(self.save_output)
            self.save_output.outputs.clear()

            _y = self.vit(r_img_org)
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

            _ = self.resnet50(d_img_org)
            cnn_dis = get_resnet_feature(self.save_output)   #0,1,2都是[B,256,56,56]
            self.save_output.outputs.clear()
            cnn_dis = self.deform_net(cnn_dis,vit_ref)

            _ = self.resnet50(r_img_org)
            cnn_ref = get_resnet_feature(self.save_output)
            self.save_output.outputs.clear()
            cnn_ref = self.deform_net(cnn_ref,vit_ref)

            pred = self.regressor(vit_dis, vit_ref, cnn_dis, cnn_ref)

            self.optimizer.zero_grad()
            loss = self.criterion(torch.squeeze(pred), labels)
            losses.append(loss.item())

            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            # save results in one epoch
            pred_batch_numpy = pred.data.cpu().numpy()
            labels_batch_numpy = labels.data.cpu().numpy()
            pred_epoch = np.append(pred_epoch, pred_batch_numpy)
            labels_epoch = np.append(labels_epoch, labels_batch_numpy)
        
        # compute correlation coefficient
        rho_s, _ = spearmanr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))
        rho_p, _ = pearsonr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))

        ret_loss = np.mean(losses)
        print('train epoch:{} / loss:{:.4} / SRCC:{:.4} / PLCC:{:.4}'.format(epoch + 1, ret_loss, rho_s, rho_p))
        logging.info('train epoch:{} / loss:{:.4} / SRCC:{:.4} / PLCC:{:.4}'.format(epoch + 1, ret_loss, rho_s, rho_p))

        return ret_loss, rho_s, rho_p
    
    def train(self):
        best_srocc = 0
        best_plcc = 0
        for epoch in range(self.opt.load_epoch, self.opt.n_epoch):
            start_time = time.time()
            logging.info('Running training epoch {}'.format(epoch + 1))
            loss_val, rho_s, rho_p = self.train_epoch(epoch)
            if (epoch + 1) % self.opt.val_freq == 0:
                logging.info('Starting eval...')
                logging.info('Running testing in epoch {}'.format(epoch + 1))
                loss, rho_s, rho_p = self.eval_epoch(epoch)
                logging.info('Eval done...')

                if rho_s > best_srocc or rho_p > best_plcc:
                    best_srocc = rho_s
                    best_plcc = rho_p
                    print('Best now')
                    logging.info('Best now')
                    self.save_model( epoch, "best.pth", loss, rho_s, rho_p)
                if epoch % self.opt.save_interval == 0:
                    weights_file_name = "epoch_%d.pth" % (epoch+1)
                    self.save_model( epoch, weights_file_name, loss, rho_s, rho_p)
            logging.info('Epoch {} done. Time: {:.2}min'.format(epoch + 1, (time.time() - start_time) / 60))
    
    def eval_epoch(self, epoch):
        with torch.no_grad():
            losses = []
            self.regressor.train()
            self.deform_net.train()
            self.vit.eval()
            self.resnet50.eval()
            # save data for one epoch
            pred_epoch = []
            labels_epoch = []

            for data in tqdm(self.val_loader):
                pred = 0
                for i in range(self.opt.num_avg_val):
                    d_img_org = data['d_img_org'].cuda()
                    r_img_org = data['r_img_org'].cuda()
                    labels = data['score']
                    labels = torch.squeeze(labels.type(torch.FloatTensor)).cuda()
                    
                    d_img_org, r_img_org = five_point_crop(i, d_img=d_img_org, r_img=r_img_org, config=self.opt)

                    _x = self.vit(d_img_org)
                    vit_dis = get_vit_feature(self.save_output)
                    self.save_output.outputs.clear()

                    _y = self.vit(r_img_org)
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

                    _ = self.resnet50(d_img_org)
                    cnn_dis = get_resnet_feature(self.save_output)   #0,1,2都是[B,256,56,56]
                    self.save_output.outputs.clear()
                    cnn_dis = self.deform_net(cnn_dis,vit_ref)

                    _ = self.resnet50(r_img_org)
                    cnn_ref = get_resnet_feature(self.save_output)
                    self.save_output.outputs.clear()
                    cnn_ref = self.deform_net(cnn_ref,vit_ref)

                    pred += self.regressor(vit_dis, vit_ref, cnn_dis, cnn_ref)
                    
                pred /= self.opt.num_avg_val
                # compute loss
                loss = self.criterion(torch.squeeze(pred), labels)
                loss_val = loss.item()
                losses.append(loss_val)

                # save results in one epoch
                pred_batch_numpy = pred.data.cpu().numpy()
                labels_batch_numpy = labels.data.cpu().numpy()
                pred_epoch = np.append(pred_epoch, pred_batch_numpy)
                labels_epoch = np.append(labels_epoch, labels_batch_numpy)
            
            # compute correlation coefficient
            rho_s, _ = spearmanr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))
            rho_p, _ = pearsonr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))
            print('Epoch:{} ===== loss:{:.4} ===== SRCC:{:.4} ===== PLCC:{:.4}'.format(epoch + 1, np.mean(losses), rho_s, rho_p))
            logging.info('Epoch:{} ===== loss:{:.4} ===== SRCC:{:.4} ===== PLCC:{:.4}'.format(epoch + 1, np.mean(losses), rho_s, rho_p))
            return np.mean(losses), rho_s, rho_p

    def save_model(self, epoch, weights_file_name, loss, rho_s, rho_p):
        print('-------------saving weights---------')
        weights_file = os.path.join(self.opt.checkpoints_dir, weights_file_name)
        torch.save({
            'epoch': epoch,
            'regressor_model_state_dict': self.regressor.state_dict(),
            'deform_net_model_state_dict': self.deform_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss
        }, weights_file)
        logging.info('Saving weights and model of epoch{}, SRCC:{}, PLCC:{}'.format(epoch, rho_s, rho_p))

if __name__ == '__main__':
    config = TrainOptions().parse()
    config.checkpoints_dir = os.path.join(config.checkpoints_dir, config.name)
    setup_seed(config.seed)
    set_logging(config)
    # logging.info(config)
    Train(config)
    