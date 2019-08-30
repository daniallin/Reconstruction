import os
import pandas as pd
from torch.utils.data import DataLoader

from dataload.nyuv2_dataset import NYUv2
from dataload.vkitti_loader import VirtualKITTI


def data_loader(args, **kwargs):
    if args.dataset == 'nyu':
        train_set = NYUv2(args.data_path)
        val_set = NYUv2(args.data_path, train=False)
        print('Train_size: {}. Validation size: {}'.format(len(train_set), len(val_set)))

        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        return train_loader, val_loader
    elif args.dataset == 'vkitti':
        train_df, val_df = list(), list()
        pathes = os.listdir(args.vkitti_datainfo)
        pathes.sort()
        for i, p in enumerate(pathes):
            data_path = os.path.join(args.vkitti_datainfo, p)
            df = pd.read_pickle(data_path)
            if i == 1 or i == 2:
                val_df.append(df)
            else:
                train_df.append(df)
        train_set = VirtualKITTI(pd.concat(train_df, ignore_index=True), args)
        val_set = VirtualKITTI(pd.concat(val_df, ignore_index=True), args)
        print('Train_size: {}. Validation size: {}'.format(len(train_set), len(val_set)))
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        return train_loader, val_loader

    else:
        raise NotImplementedError

