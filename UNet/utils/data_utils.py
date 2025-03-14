import math
import os

import numpy as np
import torch

from monai import data, transforms
from monai.data import load_decathlon_datalist

import pickle
import torch.nn.functional as F
from monai.transforms.transform import MapTransform
from monai.transforms import Compose
from catalyst.data.sampler import DistributedSamplerWrapper

rician_std = 0.1 # 0.1 for 10% noise

class Sampler(torch.utils.data.Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, make_even=True):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        self.shuffle = shuffle
        self.make_even = make_even
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        indices = list(range(len(self.dataset)))
        self.valid_length = len(indices[self.rank : self.total_size : self.num_replicas])

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))
        if self.make_even:
            if len(indices) < self.total_size:
                if self.total_size - len(indices) < len(indices):
                    indices += indices[: (self.total_size - len(indices))]
                else:
                    extra_ids = np.random.randint(low=0, high=len(indices), size=self.total_size - len(indices))
                    indices += [indices[ids] for ids in extra_ids]
            assert len(indices) == self.total_size
        indices = indices[self.rank : self.total_size : self.num_replicas]
        self.num_samples = len(indices)
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch

def get_shm(l,path):
    for i in range(len(l)):
        if os.sep in l[i]:
            ptid,ptid2,f = l[i].split(os.sep)[-3:]
        elif '/' in l[i]:
            ptid,ptid2,f = l[i].split('/')[-3:]
        elif '\\' in l[i]:
            ptid,ptid2,f = l[i].split('\\')[-3:]
        else:
            ValueError('Path error for case {}'.format(l[i]))
        if ptid2.startswith('AIPR'):
            ptid=ptid2
        elif not ptid.startswith('AIPR'):
            ValueError('PTID error for case {}'.format(l[i]))

        l[i] = os.path.join(path,ptid,f)

    return l

def get_loader(args):
    # TRAIN DATALOADER STARTS HERE
    cache_dir_suffix = None
    train_transform_list = [
        transforms.LoadImaged(keys=["axt2", "highb", "adc", "wg","pz","tz", "label"]),
        transforms.EnsureTyped(keys=["axt2", "highb", "adc", "wg","pz","tz", "label", "data"],data_type='tensor'),
        transforms.EnsureChannelFirstd(keys=["axt2", "highb", "adc", "wg","pz","tz", "label"],channel_dim='no_channel'),
        transforms.Orientationd(keys=["axt2", "highb", "adc", "wg","pz","tz", "label"], axcodes="RAS"),
        transforms.Spacingd(
            keys=["axt2", "highb", "adc", "wg","pz","tz", "label"], pixdim=(args.space_x, args.space_y, args.space_z), mode=("bilinear", "bilinear", "bilinear", "nearest","nearest","nearest","nearest")
        ),
        transforms.ScaleIntensityRangePercentilesd(
                keys=["axt2","highb","adc"], lower=args.min_percentile, upper=args.max_percentile, b_min=0.0, b_max=1.0, clip=True, channel_wise=True
        ),
        
        transforms.RandRicianNoised(keys=["axt2"], prob=args.randnoise_prob, mean=0, std=rician_std), # Point of no return re: randomness
        transforms.RandRicianNoised(keys=["highb","adc"], prob=args.randnoise_prob, mean=0, std=rician_std),
        transforms.RandScaleIntensityd(keys=["axt2","highb","adc"], factors=0.1, prob=args.randscale_prob, channel_wise=True),
        transforms.RandShiftIntensityd(keys=["axt2","highb","adc"], offsets=0.1, prob=args.randshift_prob, channel_wise=True),

        transforms.ThresholdIntensityd(keys=["axt2","highb","adc"], threshold=1.0, above=False, cval=1.0), # Clip any values above 1.0 to 1.0
        transforms.ThresholdIntensityd(keys=["axt2","highb","adc"], threshold=0.0, above=True, cval=0.0), # Clip any values below 0.0 to 0.0

    ]
    train_transform_list += [
        transforms.RandFlipd(keys=["axt2", "highb", "adc", "wg", "label"], prob=args.randflip_prob, spatial_axis=0),
        transforms.RandRotated(keys=["axt2", "highb", "adc", "wg", "label"], range_z=15*3.14/180, prob=args.randrotate_prob, mode=('bilinear','bilinear','bilinear','nearest','nearest')),
        transforms.RandAffined(keys=["axt2", "highb", "adc", "wg", "label"], prob=args.randaffine_prob, translate_range=(25,25,2), padding_mode='zeros', mode=('bilinear','bilinear','bilinear','nearest','nearest')),
        transforms.RandZoomd(keys=["axt2", "highb", "adc", "wg", "label"], prob=args.randzoom_prob, min_zoom=0.9, max_zoom=1.1, mode=('bilinear','bilinear','bilinear','nearest','nearest')),
    
        transforms.MaskIntensityd(keys=["axt2", "highb", "adc", "pz","tz"], mask_key="wg"),
        
        transforms.ConcatItemsd(keys=["axt2", "highb", "adc"], name="image"),
        transforms.EnsureTyped(keys=["image", "wg", "label"],data_type='tensor'),
    ]
    cache_dir_suffix = "WG"

    train_transform = transforms.Compose(train_transform_list)
    val_transform_list = [
        transforms.LoadImaged(keys=["axt2", "highb", "adc", "wg","pz","tz", "label"]),
        transforms.EnsureTyped(keys=["axt2", "highb", "adc", "wg","pz","tz", "label", "data"],data_type='tensor'),
        transforms.EnsureChannelFirstd(keys=["axt2", "highb", "adc", "wg","pz","tz", "label"],channel_dim='no_channel'),
        transforms.Orientationd(keys=["axt2", "highb", "adc", "wg","pz","tz", "label"], axcodes="RAS"),
        transforms.Spacingd(
            keys=["axt2", "highb", "adc", "wg","pz","tz", "label"], pixdim=(args.space_x, args.space_y, args.space_z), mode=("bilinear", "bilinear", "bilinear", "nearest","nearest","nearest","nearest")
        ),
        transforms.ScaleIntensityRangePercentilesd(
                    keys=["axt2","highb","adc"], lower=args.min_percentile, upper=args.max_percentile, b_min=0.0, b_max=1.0, clip=True, channel_wise=True
            ),
    ]
    val_transform_list += [
        transforms.MaskIntensityd(keys=["axt2", "highb", "adc"], mask_key="wg"),
        transforms.ConcatItemsd(keys=["axt2", "highb", "adc"], name="image"),
        transforms.EnsureTyped(keys=["image", "wg", "label"],data_type='tensor'),
    ]
        
    val_transform = transforms.Compose(val_transform_list)
    
    cache_last = args.cache_add if hasattr(args,"cache_add") and args.cache_add else "cache"
    cache_dir = os.sep.join([args.cache_dir,cache_last]) if args.cache_dir is not None else None
    picklename = f'train_set.pickle'
    with open('./dataset/'+picklename,'rb') as f:
        t2_l3,b16mc_l3,newadc_l3,wg_gt_l3,pz_l3,tz_l3,pc_t_gt_l3,data_l3 = pickle.load(f)

    t2_l =      list(t2_l3)
    b16mc_l =   list(b16mc_l3)
    newadc_l =  list(newadc_l3)
    pc_t_gt_l = list(pc_t_gt_l3)
    wg_gt_l =   list(wg_gt_l3)
    pz_gt_l =   list(pz_l3)
    tz_gt_l =   list(tz_l3)
    data_l =    list(data_l3)

    t2_l = get_shm(t2_l,args.data_dir)
    b16mc_l = get_shm(b16mc_l,args.data_dir)
    newadc_l = get_shm(newadc_l,args.data_dir)
    wg_gt_l = get_shm(wg_gt_l,args.data_dir)
    pz_gt_l = get_shm(pz_gt_l,args.data_dir)
    tz_gt_l = get_shm(tz_gt_l,args.data_dir)
    pc_t_gt_l = get_shm(pc_t_gt_l,args.data_dir)
    
    
    length = len(t2_l)
    if len(t2_l) != len(b16mc_l) or len(t2_l) != len(newadc_l) or len(t2_l) != len(wg_gt_l) \
        or len(t2_l) != len(pc_t_gt_l) or len(t2_l) != len(data_l) or len(t2_l) != len(pz_gt_l) \
            or len(t2_l) != len(tz_gt_l):
        raise ValueError('Length mismatch')


    train_datafiles = [{'axt2':t2_l[i],'highb':b16mc_l[i],'adc':newadc_l[i],'wg':wg_gt_l[i],\
                        'pz':pz_gt_l[i],'tz':tz_gt_l[i], 'label':pc_t_gt_l[i],'data':data_l[i],\
                        'clinical':data_l[i][3:6]} for i in range(length)]
    if args.test:
        print("Pulling data from test set!")
        picklename = f'test_set.pickle'
        with open('./dataset/'+picklename,'rb') as f:
            t2_l3,b16mc_l3,newadc_l3,wg_gt_l3,pz_l3,tz_l3,pc_t_gt_l3,data_l3 = pickle.load(f)
    else:
        picklename = f'val_set.pickle'
        with open('./dataset/'+picklename,'rb') as f:
            t2_l3,b16mc_l3,newadc_l3,wg_gt_l3,pz_l3,tz_l3,pc_t_gt_l3,data_l3 = pickle.load(f)

    t2_l_m = list(t2_l3)
    b16mc_l_m = list(b16mc_l3)
    newadc_l_m = list(newadc_l3)
    wg_gt_l_m = list(wg_gt_l3)
    pz_gt_l_m = list(pz_l3)
    tz_gt_l_m = list(tz_l3)
    pc_t_gt_l_m = list(pc_t_gt_l3)
    data_l_m = list(data_l3)
            
    t2_l_m = get_shm(t2_l_m,args.data_dir)
    b16mc_l_m = get_shm(b16mc_l_m,args.data_dir)
    newadc_l_m = get_shm(newadc_l_m,args.data_dir)
    wg_gt_l_m = get_shm(wg_gt_l_m,args.data_dir)
    pz_gt_l_m = get_shm(pz_gt_l_m,args.data_dir)
    tz_gt_l_m = get_shm(tz_gt_l_m,args.data_dir)
    pc_t_gt_l_m = get_shm(pc_t_gt_l_m,args.data_dir)

    length = len(t2_l_m)
    if len(t2_l_m) != len(b16mc_l_m) or len(t2_l_m) != len(newadc_l_m) or len(t2_l_m) != len(wg_gt_l_m) \
        or len(t2_l_m) != len(pc_t_gt_l_m) or len(t2_l_m) != len(data_l_m) or len(t2_l_m) != len(pz_gt_l_m) \
        or len(t2_l_m) != len(tz_gt_l_m):
        raise ValueError('Length mismatch')

    val_mixed_datafiles = [{'axt2':t2_l_m[i],'highb':b16mc_l_m[i],'adc':newadc_l_m[i],'wg':wg_gt_l_m[i],\
                            'pz':pz_gt_l_m[i],'tz':tz_gt_l_m[i], 'label':pc_t_gt_l_m[i],'data':data_l_m[i],\
                            'clinical':data_l_m[i][3:6]} for i in range(length)]

    if not args.nocache and cache_dir is not None:
        train_ds = data.PersistentDataset(data=train_datafiles, transform=train_transform, cache_dir=cache_dir)
    else:
        train_ds = data.Dataset(data=train_datafiles, transform=train_transform)
    train_sampler = Sampler(train_ds, num_replicas=args.world_size, rank=args.rank) if args.distributed else None
    shuffle = (train_sampler is None)
    batch_size = args.batch_size
    train_loader = data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=True if args.workers > 0 else False,
        prefetch_factor=args.prefetch_factor,
        sampler = train_sampler,
    )
    
    if args.test:
        val_cache_dir = cache_dir + cache_dir_suffix + "_test" if cache_dir is not None else None
    else:
        val_cache_dir = cache_dir + cache_dir_suffix + "_val" if cache_dir is not None else None
    if not args.nocache and val_cache_dir is not None:
        val_ds = data.PersistentDataset(data=val_mixed_datafiles, transform=val_transform, cache_dir=val_cache_dir)
    else:
        val_ds = data.Dataset(data=val_mixed_datafiles, transform=val_transform)
    val_sampler = Sampler(val_ds, shuffle=False, num_replicas=args.world_size, rank=args.rank) if args.distributed else None
    val_loader = data.DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False if not args.distributed else None,
        num_workers=args.valworkers,
        pin_memory=True,
        persistent_workers=True if args.workers > 0 else False,
        sampler=val_sampler,
    )
    
    if args.test:
        loader = [None, None, val_loader]
    else:
        loader = [train_loader, val_loader]

    return loader

if __name__ == '__main__':
    raise NotImplementedError("This is a utility module and cannot be run directly")