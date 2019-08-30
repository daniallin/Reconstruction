import os
import glob
import math
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
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
        if self.args.sample_times > 1:
            sample_interval = int(np.ceil(self.args.seq_len / self.args.sample_times))
            start_frames = list(range(0, self.args.seq_len, sample_interval))
            print('Sample start from frame {}'.format(start_frames))
        else:
            start_frames = [0]

        for i in range(len(self.rp)):
            df_seq_len, df_img, df_sem_img = list(), list(), list()
            df_depth_img, df_pose = list(), list()
            df_pose_6d = list()

            imgs = glob.glob('{}/*.png'.format(self.rp[i]))
            imgs.sort()
            sem_imgs = glob.glob('{}/*.png'.format(self.sp[0][i]))
            sem_imgs.sort()
            sem_labels = pd.read_csv(self.sp[1][i], sep=' ', index_col=False)
            depth_imgs = glob.glob('{}/*.png'.format(self.dp[i]))
            depth_imgs.sort()
            poses = pd.read_csv(self.cp[i], sep=' ', index_col=0)

            for st in start_frames:
                seq_len = np.random.randint(self.args.seq_len-1, self.args.seq_len+2)
                num_frames = len(poses) - st
                jump = seq_len - self.args.overlap
                residual = num_frames % jump
                if residual != 0:
                    num_frames -= residual
                print(seq_len, jump, num_frames, st)
                img_slt = [imgs[i:i+seq_len] for i in range(st, num_frames, jump)]
                sem_img_slt = [sem_imgs[i:i+seq_len] for i in range(st, num_frames, jump)]
                depth_img_slt = [depth_imgs[i:i+seq_len] for i in range(st, num_frames, jump)]
                pose_slt = [poses[i:i+seq_len] for i in range(st, num_frames, jump)]
                pose_slt_6d = [self.get_6d_poses(poses[i:i+seq_len]) for i in range(st, num_frames, jump)]

                df_seq_len += [len(xs) for xs in img_slt]
                df_img += img_slt
                df_sem_img += sem_img_slt
                df_depth_img += depth_img_slt
                df_pose += pose_slt
                df_pose_6d += pose_slt_6d
            data = {'seq_len': df_seq_len, 'image_path': df_img, 'semantic_img': df_sem_img,
                    'depth_img': df_depth_img, 'camera_pose': df_pose, 'camera_pose_6d': df_pose_6d}
            df = pd.DataFrame(data, columns=['seq_len', 'image_path', 'semantic_img',
                                             'depth_img', 'camera_pose', 'camera_pose_6d'])
            df = pd.concat([df, sem_labels], axis=1)
            save_name = self.cp[i].split('/')[-1].split('.')[0] + '_'\
                        + str(args.seq_len) + '.pickle'
            # print(save_name)
            if not os.path.exists('../datainfo/vkitti'):
                os.mkdir('../datainfo/vkitti')
            df.to_pickle(os.path.join('../datainfo/vkitti', save_name))

    def get_6d_poses(self, ps):
        """ For 6dof pose representation """
        poses = list()
        for p in ps.iterrows():
            frame_id = int(p[0])
            SE3 = np.array(p[1:]).reshape((4, 4))
            tran = np.array([SE3[0][3], SE3[1][3], SE3[2][3]])
            angles = self.rotation_matrix_to_euler_angles(SE3[:3, :3])
            pose = np.concatenate((np.array([frame_id]), angles, tran))
            poses.append(pose)

        return poses  # rpyxyz

    def rotation_matrix_to_euler_angles(self, R):
        """ calculates rotation matrix to euler angles
            referred from https://www.learnopencv.com/rotation-matrix-to-euler-angles
        """
        assert (self.is_rotation_matrix(R))
        sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        singular = sy < 1e-6

        if not singular:
            x = math.atan2(R[2, 1], R[2, 2])
            y = math.atan2(-R[2, 0], sy)
            z = math.atan2(R[1, 0], R[0, 0])
        else:
            x = math.atan2(-R[1, 2], R[1, 1])
            y = math.atan2(-R[2, 0], sy)
            z = 0

        return np.array([x, y, z])

    def is_rotation_matrix(self, R):
        """ Checks if a matrix is a valid rotation matrix
            referred from https://www.learnopencv.com/rotation-matrix-to-euler-angles/
        """
        Rt = np.transpose(R)
        shouldBeIdentity = np.dot(Rt, R)
        I = np.identity(3, dtype=R.dtype)
        n = np.linalg.norm(I - shouldBeIdentity)
        return n < 1e-6


def calculate_rgb_mean_std(img_path_list, minus_point_5=False):
    n_images = len(img_path_list)
    cnt_pixels = 0
    print('Numbers of frames in training dataset: {}'.format(n_images))
    mean_np = [0, 0, 0]
    mean_tensor = [0, 0, 0]
    to_tensor = transforms.ToTensor()

    image_sequence = []
    for idx, img_path in enumerate(img_path_list):
        print('{} / {}'.format(idx, n_images), end='\r')
        img_as_img = Image.open(img_path)
        img_as_tensor = to_tensor(img_as_img)
        if minus_point_5:
            img_as_tensor = img_as_tensor - 0.5
        img_as_np = np.array(img_as_img)
        img_as_np = np.rollaxis(img_as_np, 2, 0)
        cnt_pixels += img_as_np.shape[1]*img_as_np.shape[2]
        for c in range(3):
            mean_tensor[c] += float(torch.sum(img_as_tensor[c]))
            mean_np[c] += float(np.sum(img_as_np[c]))
    mean_tensor =  [v / cnt_pixels for v in mean_tensor]
    mean_np = [v / cnt_pixels for v in mean_np]
    print('mean_tensor = ', mean_tensor)
    print('mean_np = ', mean_np)

    std_tensor = [0, 0, 0]
    std_np = [0, 0, 0]
    for idx, img_path in enumerate(img_path_list):
        print('{} / {}'.format(idx, n_images), end='\r')
        img_as_img = Image.open(img_path)
        img_as_tensor = to_tensor(img_as_img)
        if minus_point_5:
            img_as_tensor = img_as_tensor - 0.5
        img_as_np = np.array(img_as_img)
        img_as_np = np.rollaxis(img_as_np, 2, 0)
        for c in range(3):
            tmp = (img_as_tensor[c] - mean_tensor[c])**2
            std_tensor[c] += float(torch.sum(tmp))
            tmp = (img_as_np[c] - mean_np[c])**2
            std_np[c] += float(np.sum(tmp))
    std_tensor = [math.sqrt(v / cnt_pixels) for v in std_tensor]
    std_np = [math.sqrt(v / cnt_pixels) for v in std_np]
    print('std_tensor = ', std_tensor)
    print('std_np = ', std_np)


if __name__ == '__main__':
    # must set the path as your virtual kitti dataset path
    path_obtainer = DataPath(path='E:/Datasets/vkitti/', version='vkitti_1.3.1', worlds=WORLD_LIST)
    rgb = path_obtainer.rgb_path()
    semantic = path_obtainer.semantic_path()
    depth = path_obtainer.depth_path()
    camera_pose = path_obtainer.camera_pose()

    args = set_params()
    data_process = DataProcess(rgb, semantic, depth, camera_pose, args)
    data_process.get_csv()

    img_path_list = []
    for p in rgb:
        img_path_list.extend(glob.glob(p + '/*.png'))
    calculate_rgb_mean_std(img_path_list, minus_point_5=args.minus_point_5)
