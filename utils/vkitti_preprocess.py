import os
import glob
import pandas as pd
import numpy as np
from utils.params import set_params

WORLD_LIST = ['0001', '0002', '0006', '0018', '0020']


class DataPath():
    def __init__(self, path, version, worlds, variation='clone'):
        self.path = path
        self.version = version
        self.worlds = worlds
        self.variation = variation

    def camera_pose(self):
        data_path = self.path + self.version + '_extrinsicsgt/'
        pose_path = list()
        for i, world in enumerate(self.worlds):
            pose_path.append(data_path + world + '_' + self.variation + '.txt')

        return pose_path

    def rgb_path(self):
        data_path = self.path + self.version + '_rgb/'
        img_path = list()
        for i, world in enumerate(self.worlds):
            img_path.append(data_path + world + '/' + self.variation)
        return img_path

    def semantic_path(self):
        data_path = self.path + self.version + '_scenegt/'
        img_path = list()
        label_path = list()
        for i, world in enumerate(self.worlds):
            img_path.append(data_path + world + '/' + self.variation)
            label_path.append(data_path + world + '_' + self.variation + '_scenegt_rgb_encoding.txt')
        return img_path, label_path

    def depth_path(self):
        data_path = self.path + self.version + '_depthgt/'
        img_path = list()
        for i, world in enumerate(self.worlds):
            img_path.append(data_path + world + '/' + self.variation)
        return img_path


class DataProcess():
    def __init__(self, rgb_path, semantic_path, depth_path, camera_path, args):
        self.rp = rgb_path
        self.sp = semantic_path
        self.dp = depth_path
        self.cp = camera_path
        self.args = args

    def get_csv(self):
        seq_len = self.args.seq_len
        if self.args.sample_times > 1:
            sample_interval = int(np.ceil(seq_len / self.args.sample_times))
            start_frames = list(range(0, seq_len, sample_interval))
            print('Sample start from frame {}'.format(start_frames))
        else:
            start_frames = [0]

        for i in range(len(self.rp)):
            df_seq_len, df_img, df_sem_img = list(), list(), list()
            df_sem_label, df_depth_img, df_pose = list(), list(), list()

            imgs = glob.glob('{}/*.png'.format(self.rp[i]))
            imgs.sort()
            sem_imgs = glob.glob('{}/*.png'.format(self.sp[0][i]))
            sem_imgs.sort()
            sem_labels = pd.read_csv(self.sp[1][i], sep=' ', index_col=False)
            depth_imgs = glob.glob('{}/*.png'.format(self.dp[i]))
            depth_imgs.sort()
            poses = pd.read_csv(self.cp[i], sep=' ', index_col=False)

            for st in start_frames:
                num_frames = len(poses)
                jump = seq_len - self.args.overlap
                residual = num_frames % seq_len
                if residual != 0:
                    num_frames -= residual
                print(seq_len, jump, num_frames, st)
                img_slt = [imgs[i:i+seq_len] for i in range(st, num_frames, jump)]
                sem_img_slt = [sem_imgs[i:i+seq_len] for i in range(st, num_frames, jump)]
                sem_label_slt = [sem_labels[i:i+seq_len] for i in range(st, num_frames, jump)]
                depth_img_slt = [depth_imgs[i:i+seq_len] for i in range(st, num_frames, jump)]
                pose_slt = [poses[i:i+seq_len] for i in range(st, num_frames, jump)]

                df_seq_len += [len(xs) for xs in img_slt]
                df_img += img_slt
                df_sem_img += sem_img_slt
                df_sem_label += sem_label_slt
                df_depth_img += depth_img_slt
                df_pose += pose_slt
            data = {'seq_len': df_seq_len, 'semantic_img': df_sem_img,
                    'semantic_label': df_sem_label, 'depth_img': df_depth_img,
                    'camera_pose': df_pose}
            df = pd.DataFrame(data, columns=['seq_len', 'semantic_img', 'semantic_label',
                                             'depth_img', 'camera_pose'])
            save_name = self.cp[i].split('/')[-1].split('.')[0] + '_'\
                        + str(args.seq_len) + '.pickle'
            # print(save_name)
            if not os.path.exists('processed_data'):
                os.mkdir('processed_data')
            df.to_pickle(os.path.join('processed_data', save_name))


if __name__ == '__main__':
    path_obtainer = DataPath(path='E:/Datasets/vkitti/', version='vkitti_1.3.1', worlds=WORLD_LIST)
    rgb = path_obtainer.rgb_path()
    semantic = path_obtainer.semantic_path()
    depth = path_obtainer.depth_path()
    camera_pose = path_obtainer.camera_pose()

    args = set_params()
    data_process = DataProcess(rgb, semantic, depth, camera_pose, args)
    data_process.get_csv()
