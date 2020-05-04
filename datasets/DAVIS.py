import os
import os.path as osp
import numpy as np
from PIL import Image
import albumentations as A
import torch
import torchvision
from torch.utils import data

import glob

import albumentations as A

class DAVIS(data.Dataset):
    # for multi object, do shuffling

    def __init__(self, root, type = 'train', height = 480, width = 480,imset='2017/train.txt', resolution='480p', single_object=False):
        self.root = root
        self.mask_dir = os.path.join(root, 'Annotations', resolution)
        self.mask480_dir = os.path.join(root, 'Annotations', '480p')
        self.image_dir = os.path.join(root, 'JPEGImages', resolution)
        _imset_dir = os.path.join(root, 'ImageSets')
        _imset_f = os.path.join(_imset_dir, imset)
        
        self.videos = []
        self.num_frames = {}
        self.num_objects = {}
        self.shape = {}
        self.size_480p = {}
        self.type = type 
        with open(os.path.join(_imset_f), "r") as lines:
            for line in lines:
                _video = line.rstrip('\n')
                self.videos.append(_video)
                #print(self.videos[-1])
                self.num_frames[_video] = len(glob.glob(os.path.join(self.image_dir, _video, '*.jpg')))
                _mask = np.array(Image.open(os.path.join(self.mask_dir, _video, '00000.png')).convert("P"))
                self.num_objects[_video] = np.max(_mask)
                self.shape[_video] = np.shape(_mask)
                _mask480 = np.array(Image.open(os.path.join(self.mask480_dir, _video, '00000.png')).convert("P"))
                self.size_480p[_video] = np.shape(_mask480)

        self.K = 11
        self.single_object = single_object

        self.height = height 
        self.width = width 
        self.augmentation = self.get_training_augmentation() if type == 'train' else self.get_validation_augmentation()
    def __len__(self):
        return len(self.videos)


    def To_onehot(self, mask):
        M = np.zeros((self.K, mask.shape[0], mask.shape[1]), dtype=np.uint8)
        for k in range(self.K):
            M[k] = (mask == k).astype(np.uint8)
        return M
    
    def All_to_onehot(self, masks):
        Ms = np.zeros((self.K, masks.shape[0], masks.shape[1], masks.shape[2]), dtype=np.uint8)
        for n in range(masks.shape[0]):
            Ms[:,n] = self.To_onehot(masks[n])
        return Ms

    def __getitem__(self, index):
        #print(index)
        video = self.videos[index]
        info = {}
        info['name'] = video
        info['num_frames'] = self.num_frames[video]
        info['size_480p'] = self.size_480p[video]

        if self.type == 'train':
            n = 3
            N_frames = np.empty((n, self.height, self.width, 3), dtype=np.float32)
            N_masks =  np.empty((n, self.height, self.width), dtype=np.uint8)
            id = [-1]
            for i in range(n):
                f = int(torch.randint(id[i] + 1, self.num_frames[video] - 3 + i + 1, (1,)))
                id.append(f)
                img_file = os.path.join(self.image_dir, video, '{:05d}.jpg'.format(f))
                image = np.array(Image.open(img_file).convert('RGB'))/255.
                try:
                    mask_file = os.path.join(self.mask_dir, video, '{:05d}.png'.format(f))  
                    mask = np.array(Image.open(mask_file).convert('P'), dtype=np.uint8)
                except:
                    # print('a')
                    mask = np.ones((self.height, self.width), dtype = np.uint8) * 255
                sample = self.augmentation(image = image, mask = mask)
                N_frames[i] = sample['image']
                N_masks[i] = sample['mask']  
                
        else:
            N_frames = np.empty((self.num_frames[video],)+self.shape[video]+(3,), dtype=np.float32)
            N_masks = np.empty((self.num_frames[video],)+self.shape[video], dtype=np.uint8)
            for f in range(self.num_frames[video]):
                img_file = os.path.join(self.image_dir, video, '{:05d}.jpg'.format(f))
                N_frames[f] = np.array(Image.open(img_file).convert('RGB'))/255.
                try:
                    mask_file = os.path.join(self.mask_dir, video, '{:05d}.png'.format(f))  
                    N_masks[f] = np.array(Image.open(mask_file).convert('P'), dtype=np.uint8)
                except:
                    # print('a')
                    N_masks[f] = 255
        
        #print(N_frames.shape, N_masks.shape)
        Fs = torch.from_numpy(np.transpose(N_frames.copy(), (3, 0, 1, 2)).copy()).float()
        if self.single_object:
            N_masks = (N_masks > 0.5).astype(np.uint8) * (N_masks < 255).astype(np.uint8)
            Ms = torch.from_numpy(self.All_to_onehot(N_masks).copy()).float()
            num_objects = torch.LongTensor([int(1)])
            return Fs, Ms, num_objects, info
        else:
            Ms = torch.from_numpy(self.All_to_onehot(N_masks).copy()).float()
            num_objects = torch.LongTensor([int(self.num_objects[video])])
            return Fs, Ms, num_objects, info


    def round_clip_0_1(self,x, **kwargs):
        return x.round().clip(0, 1)

    # define heavy augmentations
    def get_training_augmentation(self):
        train_transform = [

            #A.HorizontalFlip(p=0.5),

            #A.ShiftScaleRotate(scale_limit=0.5, rotate_limit=20,p = 1, border_mode = 0),

            A.PadIfNeeded(min_height=480, min_width=1152, always_apply=True,border_mode = 0),
            A.RandomCrop(height=self.height, width=self.width, always_apply=True),

            A.IAAAdditiveGaussianNoise(p=0.2),
            A.IAAPerspective(p=0.5),

            A.Lambda(mask=self.round_clip_0_1)
        ]
        return A.Compose(train_transform)

    def get_validation_augmentation(self):
        """Add paddings to make image shape divisible by 32"""
        test_transform = [
            A.PadIfNeeded(self.height, self.width)
        ]
        return A.Compose(test_transform)

    def get_preprocessing(self,preprocessing_fn):
        """Construct preprocessing transform
        
        Args:
            preprocessing_fn (callbale): data normalization function 
                (can be specific for each pretrained neural network)
        Return:
            transform: albumentations.Compose
        
        """
        
        _transform = [
            A.Lambda(image=preprocessing_fn),
        ]
        return A.Compose(_transform)



if __name__ == '__main__':
    pass
    # dataset = DAVIS(root='../Data/DAVIS',
    #         type='train',
    #         imset='2017/train.txt',
    #         resolution='480p')
    # import matplotlib.pyplot as plt 
    # for i in range(len(dataset)):
    #     Fs, Ms, no, info = dataset[i]
    #     Fs = torch.transpose(Fs, 0, 1)
    #     Fs = torch.transpose(Fs, 1, 2)
    #     Fs = torch.transpose(Fs, 2, 3)
    #     Ms = np.argmax(Ms, axis = 0)
    #     for i in range(3):
    #         plt.subplot(3, 2, i * 2 + 1)
    #         plt.imshow(np.asarray(Fs[i]))
    #         plt.subplot(3, 2, i * 2 + 2)
    #         plt.imshow(np.asarray(Ms[i]))
    #     plt.show() 
    #     #plt.close()