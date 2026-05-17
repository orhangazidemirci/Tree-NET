import torch
from torch.autograd import Variable
import os
import argparse
from datetime import datetime
from lib.pvt import PolypPVT
from utils.dataloader import get_loader, test_dataset
from utils.utils import clip_gradient, adjust_lr, AvgMeter
import torch.nn.functional as F
import numpy as np
import logging
import torch.nn as nn
from torchvision import transforms

import matplotlib.pyplot as plt

class EuclideanLoss(nn.Module):
    def __init__(self):
        super(EuclideanLoss, self).__init__()

    def forward(self, predicted, true):
        # Compute Euclidean loss (L2 loss)
        loss = torch.mean((predicted - true) ** 2)
        return loss
    

# class Encoder_x4(nn.Module):
#    def __init__(self):

#        super(Encoder_x4, self).__init__()
#        # self.encoder=nn.Sequential(nn.Conv2d(3,8, 3),nn.ELU(),nn.MaxPool2d(2,2),
#        #                 nn.ELU(),nn.Conv2d(8, 24, 3),nn.ELU(),nn.Conv2d(24, 32, 5,stride=3)
#        #                 ##nn.Conv2d(8, 8, 3)
#        #                 ,nn.ELU(),nn.Conv2d(32, 10, 3,stride=2),nn.ELU(),nn.ConvTranspose2d(10, 10, 3),nn.ELU())
#        self.encoder = nn.Sequential(
#            nn.Conv2d(3, 8, 3, padding=2),
#            nn.BatchNorm2d(8),
#            nn.ELU(),
#            nn.Conv2d(8, 24, 3, padding=1, stride=2),
#            nn.BatchNorm2d(24),
#            nn.ELU(),
#            nn.Conv2d(24, 32, 5, padding=1),
#            nn.BatchNorm2d(32),
#            nn.ReLU(),
#            nn.Conv2d(32, 18, 3, padding=1, stride=2),
#            nn.BatchNorm2d(18),
#            nn.Tanh(),
#            nn.Conv2d(18, 3, 3, padding=1),  # Adjusted padding
#            nn.BatchNorm2d(3),
#            nn.ELU()
#        )

#    #        self.decoder=nn.Sequential(nn.ConvTranspose2d(10, 32, 5,stride=3),nn.ELU(),nn.Conv2d(32, 24, 3),nn.ELU(),nn.Upsample(scale_factor=2, mode='nearest'),
#    # nn.Conv2d(24, 16, 3),nn.ELU(),nn.Upsample(scale_factor=2, mode='nearest'),
#    # nn.Conv2d(16, 8, 3),nn.ELU(), nn.Conv2d(8, 3, 3),nn.ELU())
#        self.decoder = nn.Sequential(
#            nn.ConvTranspose2d(3, 10, 3, padding=1),  # Output: (N, 10, 64, 64)
#            nn.BatchNorm2d(10),
#            nn.ReLU(),
#            nn.ConvTranspose2d(10, 24, 3, padding=1, stride=2),  # Output: (N, 24, 128, 128)
#            nn.BatchNorm2d(24),
#            nn.ELU(),
#            nn.ConvTranspose2d(24, 32, 3, padding=1),  # Output: (N, 32, 128, 128)
#            nn.BatchNorm2d(32),
#            nn.Tanh(),
#            nn.ConvTranspose2d(32, 32, 3, padding=1),  # Output: (N, 32, 256, 256)
#            nn.BatchNorm2d(32),
#            nn.ReLU(),
#            nn.Upsample(scale_factor=2, mode='nearest'),
#            nn.ConvTranspose2d(32, 24, 3, padding=1),  # Output: (N, 24, 256, 256)
#            nn.BatchNorm2d(24),
#            nn.ELU(),
#            nn.ConvTranspose2d(24, 8, 3, padding=1),  # Output: (N, 18, 256, 256)
#            nn.BatchNorm2d(8),
#            nn.ReLU(),
#            nn.ConvTranspose2d(8, 3, 3),  # Output: (N, 3, 256, 256)
#            nn.ELU()
#            )

#    def forward(self, inp):
#        x=self.encoder(inp)
#        ## x=self.bridge(x)+x
       
#        ## x=x-self.res(inp)
#       # # x=self.encoder2(x)
       
#        # x=self.decoder(x)
       
#      #  # out=self.decoder(x)+F.interpolate(self.pooling(inp), scale_factor=8, mode='nearest')
#        return x
        
# encoder_path='./model_pth/encoder_x4_CVC-ClinicDB.pt'
# in_model=Encoder_x4().cuda()
# save_model = torch.load(encoder_path)
# model_dict = in_model.state_dict()
# state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
# model_dict.update(state_dict)
# in_model.load_state_dict(model_dict)
# for param in in_model.parameters():
#     param.requires_grad = False  
    
# class Decoder_x4(nn.Module):
#     def __init__(self):

#         super(Decoder_x4, self).__init__()
#         self.encoder_out= nn.Sequential(
#             nn.Conv2d(1, 8, 3, padding=2),
#             nn.ReLU(),
#             nn.Conv2d(8, 24, 3, padding=1, stride=2),
#             nn.ReLU(),
#             nn.Conv2d(24, 32, 5, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(32, 18, 3, padding=1, stride=2),
#             nn.ReLU(),
#             nn.Conv2d(18, 3, 3, padding=1), 
#             nn.Sigmoid()
#         )

#         self.decoder_out = nn.Sequential(
#             nn.ConvTranspose2d(3, 18, 3, padding=1, stride=2),  # Output: (N, 24, 128, 128)
#             nn.ReLU(),
#             nn.ConvTranspose2d(18, 32, 3, padding=1),  # Output: (N, 32, 128, 128)
#             nn.ReLU(),
#             nn.ConvTranspose2d(32, 32, 3, padding=1),  # Output: (N, 32, 256, 256)
#             nn.ReLU(),
#             nn.Upsample(scale_factor=2, mode='nearest'),
#             nn.ConvTranspose2d(32, 24, 3, padding=1),  # Output: (N, 24, 256, 256)
#             nn.ReLU(),
#             nn.ConvTranspose2d(24, 8, 3, padding=1),  # Output: (N, 18, 256, 256)
#             nn.ReLU(),
#             nn.ConvTranspose2d(8, 1, 3),  # Output: (N, 3, 256, 256)
#             nn.Sigmoid()
#             )
#     def forward(self, x):
#         x=self.encoder_out(x)
#         # x=self.decoder_out(x)
#         return x
    
# decoder_path='./model_pth/decoder_x4_CVC-ClinicDB.pt'

# out_model=Decoder_x4().cuda()
# save_model = torch.load(decoder_path)
# model_dict = out_model.state_dict()
# state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
# model_dict.update(state_dict)
# out_model.load_state_dict(model_dict)

# class Decoder_x4(nn.Module):
#     def __init__(self):

#         super(Decoder_x4, self).__init__()
#         self.encoder_out= nn.Sequential(
#             nn.Conv2d(1, 8, 3, padding=2),
#             nn.ReLU(),
#             nn.Conv2d(8, 24, 3, padding=1, stride=2),
#             nn.ReLU(),
#             nn.Conv2d(24, 32, 5, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(32, 18, 3, padding=1, stride=2),
#             nn.ReLU(),
#             nn.Conv2d(18, 3, 3, padding=1), 
#             nn.Sigmoid()
#         )

#         self.decoder_out = nn.Sequential(
#             nn.ConvTranspose2d(3, 18, 3, padding=1, stride=2),  # Output: (N, 24, 128, 128)
#             nn.ReLU(),
#             nn.ConvTranspose2d(18, 32, 3, padding=1),  # Output: (N, 32, 128, 128)
#             nn.ReLU(),
#             nn.ConvTranspose2d(32, 32, 3, padding=1),  # Output: (N, 32, 256, 256)
#             nn.ReLU(),
#             nn.Upsample(scale_factor=2, mode='nearest'),
#             nn.ConvTranspose2d(32, 24, 3, padding=1),  # Output: (N, 24, 256, 256)
#             nn.ReLU(),
#             nn.ConvTranspose2d(24, 8, 3, padding=1),  # Output: (N, 18, 256, 256)
#             nn.ReLU(),
#             nn.ConvTranspose2d(8, 1, 3),  # Output: (N, 3, 256, 256)
#             nn.Sigmoid()
#             )
#     def forward(self, x):
#         # x=self.encoder_out(x)
#         x=self.decoder_out(x)
#         return x
    
# decoder_path='./model_pth/decoder_x4_Polyp.pt'

# out_model2=Decoder_x4().cuda()
# save_model = torch.load(decoder_path)
# model_dict = out_model2.state_dict()
# state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
# model_dict.update(state_dict)
# out_model2.load_state_dict(model_dict)


# for param in out_model.parameters():
#     param.requires_grad = False   
def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)

    return (wbce + wiou).mean()


def test(model, path, dataset):

    data_path = os.path.join(path, dataset)
    image_root = '{}/images/'.format(data_path)
    gt_root = '{}/masks/'.format(data_path)
    model.eval()
    num1 = len(os.listdir(gt_root))
    test_loader = test_dataset(image_root, gt_root, 384)
    DSC = 0.0
    for i in range(num1):
        image, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()
        
        # # res, res1  = model(image)
        # # down_img=F.upsample(image, size=(int(384/4), int(384/4)), mode='bilinear', align_corners=True)
        # image=in_model(image)
        # image=transform(image)

        res, res1  = model(image)
        # res=out_model2(res)
        # res1=out_model2(res1)

        # eval Dice
        res = F.upsample(res + res1 , size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        input = res
        target = np.array(gt)
        N = gt.shape
        smooth = 1
        input_flat = np.reshape(input, (-1))
        target_flat = np.reshape(target, (-1))
        intersection = (input_flat * target_flat)
        dice = (2 * intersection.sum() + smooth) / (input.sum() + target.sum() + smooth)
        dice = '{:.4f}'.format(dice)
        dice = float(dice)
        DSC = DSC + dice

    return DSC / num1

transform=transforms.Compose([
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Standardize
])

def train(train_loader, model, optimizer, epoch, test_path):
    model.train()
    # in_model.eval()
    # out_model.eval()
    global best
    criterion = EuclideanLoss()
    size_rates = [0.75, 1, 1.25] 
    trainsizes= [256,384,512]
    loss_P2_record = AvgMeter()
    for i, pack in enumerate(train_loader, start=1):
        for size in trainsizes:
            torch.cuda.empty_cache()
            optimizer.zero_grad()
            # ---- data prepare ----
            images, gts = pack
            images = Variable(images).cuda()
            gts = Variable(gts).cuda()
            # ---- rescale ----
            # trainsize = int(round(opt.trainsize * rate / 32) * 32)
            trainsize=size
            if size != 384:
                images = F.upsample(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                gts = F.upsample(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
            # ---- forward ----
            
            # down_img=F.upsample(images, size=(int(trainsize/4), int(trainsize/4)), mode='bilinear', align_corners=True)

            # images=in_model(images)

            # images.requires_grad = False
            # images=transform(images)
            # gts=out_model(gts)
            # gts.requires_grad = False
            
            P1, P2= model(images)
            # ---- loss function ----
            loss_P1 = structure_loss(P1, gts)
            loss_P2 = structure_loss(P2, gts)
            loss = loss_P1 + loss_P2 
            # ---- backward ----
            loss.backward()
            clip_gradient(optimizer, opt.clip)
            optimizer.step()
            # ---- recording loss ----
            if size == 384:
                loss_P2_record.update(loss_P2.data, opt.batchsize)
        # ---- train visualization ----
        if i % 20 == 0 or i == total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                  ' lateral-5: {:0.4f}]'.
                  format(datetime.now(), epoch, opt.epoch, i, total_step,
                         loss_P2_record.show()))
    # save model 
    save_path = (opt.train_save)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    torch.save(model.state_dict(), save_path +str(epoch)+ 'PolypPVT.pth')
    # choose the best model

    global dict_plot
    torch.cuda.empty_cache()
    test1path = './dataset/ISAC2018/'
    if (epoch + 1) % 1 == 0:
        for dataset in ['test']:
            dataset_dice = test(model, test1path, dataset)
            logging.info('epoch: {}, dataset: {}, dice: {}'.format(epoch, dataset, dataset_dice))
            print(dataset, ': ', dataset_dice)
            dict_plot[dataset].append(dataset_dice)
        meandice = test(model, test_path, 'valid')
        dict_plot['test'].append(meandice)
        if meandice > best:
            best = meandice
            torch.save(model.state_dict(), save_path + 'PolypPVT.pth')
            torch.save(model.state_dict(), save_path +str(epoch)+ 'PolypPVT-best.pth')
            print('##############################################################################best', best)
            logging.info('##############################################################################best:{}'.format(best))


def plot_train(dict_plot=None, name = None):
    color = ['red']
    line = ['-', "--"]
    for i in range(len(name)):
        plt.plot(dict_plot[name[i]], label=name[i], color=color[i], linestyle=line[(i + 1) % 2])
        transfuse = {'test':0.83}
        plt.axhline(y=transfuse[name[i]], color=color[i], linestyle='-')
    plt.xlabel("epoch")
    plt.ylabel("dice")
    plt.title('Train')
    plt.legend()
    plt.savefig('eval.png')
    # plt.show()
    
    
if __name__ == '__main__':
    dict_plot = {'test':[]}
    name = ['test']
    ##################model_name#############################
    model_name = 'ISIC-2018'
    ###############################################
    parser = argparse.ArgumentParser()

    parser.add_argument('--epoch', type=int,
                        default=100, help='epoch number')

    parser.add_argument('--lr', type=float,
                        default=1e-4, help='learning rate')

    parser.add_argument('--optimizer', type=str,
                        default='AdamW', help='choosing optimizer AdamW or SGD')

    parser.add_argument('--augmentation',
                        default=True, help='choose to do random flip rotation')

    parser.add_argument('--batchsize', type=int,
                        default=4, help='training batch size')

    parser.add_argument('--trainsize', type=int,
                        default=384, help='training dataset size')

    parser.add_argument('--clip', type=float,
                        default=0.5, help='gradient clipping margin')

    parser.add_argument('--decay_rate', type=float,
                        default=0.1, help='decay rate of learning rate')

    parser.add_argument('--decay_epoch', type=int,
                        default=50, help='every n epochs decay learning rate')

    parser.add_argument('--root_path',   type=str,
                        default='dataset/Synapse',
                        help='root dir for training data')
    parser.add_argument('--list_dir',    type=str,
                        default='dataset/Synapse/lists/lists_Synapse',
                        help='list dir')
    parser.add_argument('--volume_path', type=str,
                        default='dataset/Synapse/test_vol_h5',
                        help='root dir for test volumes')
    parser.add_argument('--output_dir',  type=str,
                        default='snapshots',
                        help='output dir for checkpoints and logs')

    parser.add_argument('--train_save', type=str,
                        default='./model_pth/'+model_name+'/')

    parser.add_argument('--seed',        type=int,   default=1234,
                        help='random seed')

    opt = parser.parse_args()
    logging.basicConfig(filename='train_log.log',
                        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')

    # ---- build models ----
    # torch.cuda.set_device(0) # set your gpu device
    model = PolypPVT().cuda() 
    # path_model='./model_pth/Bests/Encoder_x4_87PolypPVT-best.pth'
    # # path_model='./model_pth/Bests/Original_CVC_66PolypPVT-best.pth'
    # save_model = torch.load(path_model)
    # model.load_state_dict(save_model)

    
    # ####Encoder Update
    # encoder_path='./model_pth/encoder_x4_ISAC2018.pt'
    # encoder = torch.load(encoder_path)

    # in_model_state = model.in_model.state_dict()  # Current weights of in_model
    
    # # Update weights selectively
    # for key in encoder.keys():
    #     if key in in_model_state:
    #         in_model_state[key] = encoder[key]
    
    # model.in_model.load_state_dict(in_model_state)


    # ####Decoder Update
    # decoder_path='./model_pth/decoder_x16_ISAC2018.pt'
    # decoder = torch.load(encoder_path)

    # out_model_state = model.out_model.state_dict()  # Current weights of in_model
    
    # # Update weights selectively
    # for key in decoder.keys():
    #     if key in out_model_state:
    #         out_model_state[key] = decoder[key]
    
    # model.out_model.load_state_dict(out_model_state)
    
    # del encoder,decoder,in_model_state,out_model_state
    # class CombinedModel(nn.Module):
    #     def __init__(self):
    #         super(CombinedModel, self).__init__()
    #         # Assuming model1, model2, and model3 are pre-initialized models
    #         self.model1 = in_model.cuda()
    #         self.model2 = model
    #         self.model3 = out_model.cuda()
    #         for param in in_model.parameters():
    #             param.requires_grad = False  
    #         for param in out_model.parameters():
    #             param.requires_grad = False  
    #     def forward(self, x):
    #         # Forward pass through each model sequentially
    #         x = self.model1(x)   # Pass input through model1
    #         x1,x2 = self.model2(x)   # Pass model1's output to model2
    #         x1 = self.model3(x1)   # Pass model2's output to model3
    #         x2 = self.model3(x2)
    #         return x1,x2
        
    # comb=CombinedModel().cuda()
    best = 0

    params = model.parameters()

    if opt.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(params, opt.lr, weight_decay=1e-4)
    else:
        optimizer = torch.optim.SGD(params, opt.lr, weight_decay=1e-4, momentum=0.9)

    print(optimizer)
    image_root = '{}/images/'.format(opt.train_path)
    gt_root = '{}/masks/'.format(opt.train_path)


    from utils.dataloader import Synapse_dataset, RandomGenerator


    db_train = Synapse_dataset(
        base_dir=opt.root_path,
        list_dir=opt.list_dir,
        split="train",
        transform=None
    )
    print("The length of train set is: {}".format(len(db_train)))
 
    def worker_init_fn(worker_id):
        random.seed(opt.seed + worker_id)
 
    trainloader = DataLoader(
        db_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        worker_init_fn=worker_init_fn
    )
 
    total_step = len(train_loader)

    print("#" * 20, "Start Training", "#" * 20)

    for epoch in range(1, opt.epoch):
        adjust_lr(optimizer, opt.lr, epoch, 0.1, 200)
        train(train_loader, model, optimizer, epoch, opt.test_path)
    
    # plot the eval.png in the training stage
    plot_train(dict_plot, name)
