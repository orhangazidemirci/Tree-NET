import torch
from torch.autograd import Variable
import os
import argparse
from datetime import datetime
from lib.pvt import PolypPVT
from lib.models import call_model
from utils.dataloader import get_loader, test_dataset
from utils.encoder import components_exp
from inference_test import run_benchmark
from utils.utils import clip_gradient, adjust_lr, AvgMeter
import torch.nn.functional as F
import numpy as np
import logging
import torch.nn as nn
from torchvision import transforms
import gc
import matplotlib.pyplot as plt

class EuclideanLoss(nn.Module):
    def __init__(self):
        super(EuclideanLoss, self).__init__()

    def forward(self, predicted, true):
        # Compute Euclidean loss (L2 loss)
        loss = torch.mean((predicted - true) ** 2)
        return loss
    


def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)

    return (wbce + wiou).mean()


def test(model, path, dataset, opt):

    if opt.arch=='pvt':
        pvt=True
    else:
        pvt=False

    image_root = '{}/images/'.format(path)
    gt_root = '{}/masks/'.format(path)
    model.eval()
    num1 = len(os.listdir(gt_root))
    test_loader = test_dataset(image_root, gt_root, opt.trainsize,opt)
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
        if pvt:
            res, res1  = model(image)
            res = F.upsample(res + res1 , size=gt.shape, mode='bilinear', align_corners=False)

        else:
            res = model(image)
            res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)

        
        # res=out_model2(res)
        # res1=out_model2(res1)

        # eval Dice
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        input = res
        target = np.array(gt)
        N = gt.shape
        smooth = 1
        input_flat = np.reshape(input, (-1))
        target_flat = np.reshape(target, (-1))
        intersection = (input_flat * target_flat).sum()
        dice = (2 * intersection.sum() + smooth) / (input.sum() + target.sum() + smooth)
        dice = '{:.4f}'.format(dice)
        dice = float(dice)
        DSC = DSC + dice

    return DSC / num1

transform=transforms.Compose([
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Standardize
])

def train(train_loader, model, optimizer, epoch, test_path, opt):
    model.train()
    global best
    criterion = EuclideanLoss()
    size_rates = [0.75, 1, 1.25] 
    trainsizes= [256,384,512]

    if opt.arch=='pvt':
        pvt=True
    else:
        pvt=False
    loss_P2_record = AvgMeter()
    for i, pack in enumerate(train_loader, start=1):
        for size in trainsizes:
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
            
            if pvt:
                P1, P2= model(images)

      
                loss_P1 = structure_loss(P1, gts)
                loss_P2 = structure_loss(P2, gts)
                loss = loss_P1 + loss_P2             
            else:
                out=model(images)
                loss= criterion(out, gts)
            # if size == 384:
            #     P1 = F.upsample(P1, size=(384, 384), mode='bilinear', align_corners=True)
            #     P2 = F.upsample(P2, size=(384, 384), mode='bilinear', align_corners=True)

            # ---- loss function ----
            # print("input size: ", images.shape)
            # print(P1.shape, gts.shape)



            # ---- backward ----
            loss.backward()
            clip_gradient(optimizer, opt.clip)
            optimizer.step()
            # ---- recording loss ----
            if size == 384:
                if pvt:
                    loss_P2_record.update(loss_P2.data, opt.batchsize)
                else:
                    loss_P2_record.update(loss.data, opt.batchsize)

        # ---- train visualization ----
        if i % 20 == 0 or i == opt.total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                  ' lateral-5: {:0.4f}]'.
                  format(datetime.now(), epoch, opt.epoch, i, opt.total_step,
                         loss_P2_record.show()))
    # save model 
    save_path = (opt.train_save)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    torch.save(model.state_dict(), save_path +str(epoch)+ 'PolypPVT.pth')
    # choose the best model

    global dict_plot
    if opt.dataset == 'CVC-ClinicDB':
        datasets = ['CVC-ClinicDB']
    elif opt.dataset == 'ISAC2018':
        datasets = ['ISAC2018']
 
    if (epoch + 1) % 1 == 0:
        
        for dataset in datasets:
        # for dataset in ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB']:
            dataset_dice = test(model, test_path, dataset, opt)
            logging.info('epoch: {}, dataset: {}, dice: {}'.format(epoch, dataset, dataset_dice))
            print(dataset, ': ', dataset_dice)
            dict_plot['test'].append(dataset_dice)
        meandice = test(model, opt.valid_path, 'valid', opt)
        # dict_plot['test'].append(meandice)
        if meandice > best:
            best = meandice
            torch.save(model.state_dict(), save_path + '_' + opt.dataset + 'best.pth')
            # torch.save(model.state_dict(), save_path +str(epoch)+ 'PolypPVT-best.pth')
            print('##############################################################################best', best)
            logging.info('##############################################################################best:{}'.format(best))


def plot_train(dict_plot=None, name = None):
    color = ['red', 'lawngreen', 'lime', 'gold', 'm', 'plum', 'blue']
    line = ['-', "--"]
    for i in range(len(name)):
        plt.plot(dict_plot[name[i]], label=name[i], color=color[i], linestyle=line[(i + 1) % 2])
        transfuse = {'CVC-300': 0.902, 'CVC-ClinicDB': 0.918, 'Kvasir': 0.918, 'CVC-ColonDB': 0.773,'ETIS-LaribPolypDB': 0.733, 'test':0.83}
        plt.axhline(y=transfuse[name[i]], color=color[i], linestyle='-')
    plt.xlabel("epoch")
    plt.ylabel("dice")
    plt.title('Train')
    plt.legend()
    plt.savefig('eval.png')
    # plt.show()
    
def experiment_prep(opt):
    global best
    gc.collect()
    torch.cuda.empty_cache()
    model_name = opt.arch
    best = 0
    opt.train_save = './model_pth/'+model_name+'/'

    if opt.dataset == 'CVC-ClinicDB':
        opt.train_path = './dataset/train/'
        opt.test_path = './dataset/TestDataset/'
        opt.valid_path= './dataset/valid/CVC-ClinicDB'


    elif opt.dataset == 'ISAC2018':
        opt.train_path = './dataset/ISAC2018/train/'
        opt.test_path = './dataset/ISAC2018/test/'
        opt.valid_path = './dataset/ISAC2018/valid'

    if opt.arch!='pvt':
        opt.encoder_component='Encoder_x4'
        opt.decoder_component='Decoder_x4'

    opt.encoder_path = f'./model_pth/{opt.encoder_component.lower()}_{opt.dataset}.pt'
    opt.decoder_path = f'./model_pth/{opt.decoder_component.lower()}_{opt.dataset}.pt'

    logging.basicConfig(filename=f'train_log{opt.component_selected}{opt.dataset}{opt.arch}{opt.training_mode}.log',
        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
        level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')
        
def experiment(opt):
    experiment_prep(opt)
    model = call_model(opt).cuda() 

    best = 0

    params = model.parameters()

    if opt.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(params, opt.lr, weight_decay=1e-4)
    else:
        optimizer = torch.optim.SGD(params, opt.lr, weight_decay=1e-4, momentum=0.9)

    print(optimizer)
    image_root = '{}/images/'.format(opt.train_path)
    gt_root = '{}/masks/'.format(opt.train_path)

    train_loader = get_loader(opt.train_path+'images/', opt.train_path+'masks/',  batchsize=opt.batchsize, trainsize=opt.trainsize,args=opt,
                              augmentation=opt.augmentation)
    opt.total_step = len(train_loader)

    print("#" * 20, "Start Training", "#" * 20)

    for epoch in range(1, opt.epoch):
        adjust_lr(optimizer, opt.lr, epoch, 0.1, 200)
        train(train_loader, model, optimizer, epoch, opt.test_path, opt)



if __name__ == '__main__':
    dict_plot = {'test':[]}
    name = ['test']
    ##################model_name#############################
    model_name = 'PVT'
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
                        default=8, help='training batch size')

    parser.add_argument('--trainsize', type=int,
                        default=384, help='training dataset size')

    parser.add_argument('--clip', type=float,
                        default=0.5, help='gradient clipping margin')

    parser.add_argument('--decay_rate', type=float,
                        default=0.1, help='decay rate of learning rate')

    parser.add_argument('--decay_epoch', type=int,
                        default=50, help='every n epochs decay learning rate')

    parser.add_argument('--dataset', type=str,
                        default='ISAC2018', help='select the dataset to train - CVC-ClinicDB, ISAC2018')

    parser.add_argument('--train_path', type=str,
                        default='./dataset/train/',
                        help='path to train dataset')

    parser.add_argument('--test_path', type=str,
                        default='./dataset/TestDataset/',
                        help='path to testing the dataset chosen')

    parser.add_argument('--arch', type=str,
                        default='UNET', help='select the model to train - pvt, UNET, NestedUNET')
    parser.add_argument('--encoder_component', type=str, default='Encoder_x4', help='select the component -- Encoder_x4')

    parser.add_argument('--decoder_component', type=str, default='Decoder_x16', help='select the component -- Decoder_x4, Decoder_x8, Decoder_x16')
    parser.add_argument('--component_selected', type=str, default='ORIGINAL', help='select the component -- ORIGINAL, Encoder_x4, Encoder_x16, Decoder_x4, Decoder_x16')

    opt = parser.parse_args()


    opt.training_mode=True
    opt.inference_mode=True
    opt.treenet=False


    opt.bottleneck_size={"Encoder_x4":3,
                "Decoder_x4":18,
                "Decoder_x8":16,
                "Decoder_x16":16
                }   
    
    experiments={'COMPONENTS': ['Encoder_x4', 'Decoder_x4', 'Decoder_x16', 'ORIGINAL'],"BATCH_SIZES": [8,16], "ARCHS": ['pvt', 'UNET', 'NestedUNET'], "INFERENCES": [True, False], "TREENET_MODES": [True, False]}

    if opt.training_mode:
        for batch_size in  experiments["BATCH_SIZES"]:
            opt.batchsize=batch_size
            opt.epoch=3
            for component in experiments["COMPONENTS"]:
                opt.component_selected = component

                if component == "ORIGINAL":

                    opt.inference_mode=True
                    for treenet_mode in experiments["TREENET_MODES"]:
                        opt.treenet=treenet_mode
                        if treenet_mode==False:
                            for arch in experiments["ARCHS"]:
                                if arch=='pvt' and batch_size==8:
                                    o=1
                                elif arch=='UNET' and batch_size==8:  
                                    o=1
                                else:
                                    opt.arch=arch

                                    print(opt.arch,treenet_mode,batch_size)
                                    
                                    experiment(opt)
                
                else:
                    print("Side components")
                    if batch_size==8:
                        o=1
                    else:  
                        print(component,treenet_mode,batch_size)

                        opt.treenet=True
                        opt.batchsize=opt.batchsize*4
                        opt.inference_mode=False
                        experiment_prep(opt)
                        model = call_model(opt).cuda() 


                        
                        train_loader = get_loader(opt.train_path+'images/', opt.train_path+'masks/',  batchsize=opt.batchsize, trainsize=opt.trainsize,args=opt,
                                    augmentation=opt.augmentation)
                        
                        image_root = '{}/images/'.format(opt.valid_path)
                        gt_root = '{}/masks/'.format(opt.valid_path)

                        valid_loader = get_loader(image_root, gt_root,  batchsize=opt.batchsize, trainsize=opt.trainsize,args=opt,
                                    augmentation=False)
                        
                        # valid_loader = test_dataset(image_root, gt_root, opt.trainsize,opt)
                        components_exp(opt, model,train_loader,valid_loader)       

    opt.component_selected='ORIGINAL'
    opt.inference_mode=True
    dataset_paths = {
        'ISAC2018':     './dataset/ISAC2018/test/',
        'CVC-ClinicDB': './dataset/TestDataset/CVC-ClinicDB/',
    }

    

    run_benchmark(opt,dataset_paths)
        
    








    
    # plot the eval.png in the training stage
    # plot_train(dict_plot, name)
