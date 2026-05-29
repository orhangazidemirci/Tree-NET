import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
import random
import torch

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=0.1): # Default std=0.1, but you can override
        """
        Initializes the transform. Stores the mean and std deviation.
        """
        # Ensure std and mean are stored correctly
        self.std = std
        self.mean = mean
        print(f"AddGaussianNoise initialized with mean={self.mean}, std={self.std}") # Debug print

    def __call__(self, tensor):
        """
        Applies the noise addition to the input tensor.
        """
        # Generate noise and add it - core logic identical to the function
        noise = torch.randn(tensor.size(), device=tensor.device) * self.std + self.mean
        noisy_tensor = tensor + noise
        return noisy_tensor

    def __repr__(self):
        # Makes the transform sequence print nicely
        return self.__class__.__name__ + f'(mean={self.mean}, std={self.std})'


class PolypDataset(data.Dataset):
    """
    dataloader for polyp segmentation tasks
    """
    def __init__(self, image_root, gt_root, trainsize, augmentations,args):

        self.encoder_mode=False
        self.decoder_mode=False
        normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                    [0.229, 0.224, 0.225])  # RGB stats
        if args.treenet:
            if "Encoder" in args.component_selected:
                self.encoder_mode=True
            elif "Decoder" in args.component_selected:
                self.decoder_mode=True      
                normalize = transforms.Normalize([0.5], [0.5]) 
         
        self.trainsize = trainsize
        self.augmentations = augmentations
        print(self.augmentations)
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.png') or f.endswith('.jpg')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.filter_files()
        self.size = len(self.images)
        if self.augmentations == 'True':
            print('Using RandomRotation, RandomFlip')
            # self.img_transform = transforms.Compose([
            #     transforms.RandomRotation(90, resample=False, expand=False, center=None, fill=None),
            #     transforms.RandomVerticalFlip(p=0.5),
            #     transforms.RandomHorizontalFlip(p=0.5),
            #     transforms.Resize((self.trainsize, self.trainsize)),
            #     transforms.ToTensor(),
            #     transforms.Normalize([0.485, 0.456, 0.406],
            #                          [0.229, 0.224, 0.225])])
            # self.gt_transform = transforms.Compose([
            #     transforms.RandomRotation(90, resample=False, expand=False, center=None, fill=None),
            #     transforms.RandomVerticalFlip(p=0.5),
            #     transforms.RandomHorizontalFlip(p=0.5),
            #     transforms.Resize((self.trainsize, self.trainsize)),
            #     transforms.ToTensor()])
            noise_std = 0.25 # Adjust this value to control noise intensity

            self.img_transform = transforms.Compose([
                    transforms.RandomRotation(90, resample=False, expand=False, center=None, fill=None),
                    transforms.RandomVerticalFlip(p=0.5),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.Resize((self.trainsize, self.trainsize)),
                    transforms.ColorJitter(contrast=0.5),  # Adjust contrast here
                    transforms.ToTensor(),
                    # AddGaussianNoise(mean=0., std=noise_std), # <--- ADDED NOISE HERE
                    normalize
                ])

            # Ground truth transformations
            self.gt_transform = transforms.Compose([
                transforms.RandomRotation(90, resample=False, expand=False, center=None, fill=None),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Resize((self.trainsize, self.trainsize)),
                transforms.ToTensor()
            ])
            
        else:
            print('no augmentation')
            self.img_transform = transforms.Compose([
                transforms.Resize((self.trainsize, self.trainsize)),
                transforms.ToTensor()
                ,normalize
                ])
            
            self.gt_transform = transforms.Compose([
                transforms.Resize((self.trainsize, self.trainsize)),
                transforms.ToTensor()])
            

    def __getitem__(self, index):
        
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        
        seed = np.random.randint(2147483647) # make a seed with numpy generator 
        random.seed(seed) # apply this seed to img tranfsorms
        torch.manual_seed(seed) # needed for torchvision 0.7
        if self.img_transform is not None:
            image = self.img_transform(image)
            
        random.seed(seed) # apply this seed to img tranfsorms
        torch.manual_seed(seed) # needed for torchvision 0.7
        if self.gt_transform is not None:
            gt = self.gt_transform(gt)
        return image, gt

    # def filter_files(self):
    #     assert len(self.images) == len(self.gts)
    #     images = []
    #     gts = []
    #     for img_path, gt_path in zip(self.images, self.gts):
    #         img = Image.open(img_path)
    #         gt = Image.open(gt_path)
    #         if img.size == gt.size:
    #             images.append(img_path)
    #             gts.append(gt_path)
    #     self.images = images
    #     self.gts = gts

    def filter_files(self):
        img_dict = {os.path.splitext(os.path.basename(p))[0]: p for p in self.images}
        gt_dict  = {os.path.splitext(os.path.basename(p))[0]: p for p in self.gts}
        common   = sorted(img_dict.keys() & gt_dict.keys())
        self.images = [img_dict[k] for k in common]
        self.gts    = [gt_dict[k]  for k in common]
        
    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            if self.decoder_mode:
                return img.convert('L')
            return img.convert('RGB')
            
            # 

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            # return img.convert('1')
            if self.encoder_mode:
                return img.convert('RGB')
            return img.convert('L')

            # return img.convert('RGB')

    def resize(self, img, gt):
        assert img.size == gt.size
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST)
        else:
            return img, gt

    def __len__(self):
        return self.size


def get_loader(image_root, gt_root, batchsize, trainsize, args, shuffle=True, num_workers=4, pin_memory=True, augmentation=False):

    dataset = PolypDataset(image_root, gt_root, trainsize, augmentation, args)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader


class test_dataset:
    def __init__(self, image_root, gt_root, testsize, args):

        self.encoder_mode=False
        self.decoder_mode=False
        normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                    [0.229, 0.224, 0.225])  # RGB stats
        if args.treenet:
            if "Encoder" in args.component_selected:
                self.encoder_mode=True
            elif "Decoder" in args.component_selected:
                self.decoder_mode=True      
                normalize = transforms.Normalize([0.5], [0.5]) 
                  

        self.testsize = testsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.tif') or f.endswith('.png') or f.endswith('.jpg')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            normalize
            ])
        self.gt_transform = transforms.ToTensor()
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        image = self.transform(image).unsqueeze(0)
        gt = self.binary_loader(self.gts[self.index])
        name = self.images[self.index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        self.index += 1
        return image, gt, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            if self.decoder_mode:
                return img.convert('L')
            return img.convert('RGB')
    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            if self.encoder_mode:
                return img.convert('RGB')
            return img.convert('L')



