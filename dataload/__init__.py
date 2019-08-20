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
        for i, p in enumerate(list(os.listdir(args.vkitti_path)).sort()):
            data_path = os.path.join(args.vkitti_path, p)
            df = pd.read_pickle(data_path)
            if i < 3:
                train_df.append(df)
            else:
                val_df.append(df)
        train_loader = DataLoader(pd.concat(train_df), args)
        val_loader = DataLoader(pd.concat(val_df), args)
        return train_loader, val_loader

    else:
        raise NotImplementedError

