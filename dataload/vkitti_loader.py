import os
import math
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class VirtualKITTI(Dataset):
    def __init__(self, dataframe, args):
        self.df = dataframe
        self.seq_len_list = list(self.df.seq_len)
        self.imgs = np.asarray(self.df.image_path)
        self.depth_imgs = self.df.depth_img
        self.semantic_imgs = self.df.semantic_img
        self.semantic_label = self.df.semantic_label
        self.camera_pose = self.df.camera_pose
        self.camera_pose_6d = self.df.camera_pose_6d

        # Transforms
        transform_ops = []
        if args.resize_mode == 'crop':
            transform_ops.append(transforms.CenterCrop((args.crop_size[0], args.crop_size[1])))
        elif args.resize_mode == 'rescale':
            transform_ops.append(transforms.Resize((args.crop_size[0], args.crop_size[1])))
        transform_ops.append(transforms.ToTensor())
        self.transformer = transforms.Compose(transform_ops)
        self.minus_point_5 = args.minus_point_5
        self.normalizer = transforms.Normalize(mean=args.img_mean, std=args.img_std)
        self.gray_normalizer = transforms.Normalize(mean=[0.5], std=[0.5])

        self.data_len = len(self.df)

    def __getitem__(self, index):
        # get RGB images
        img_seq = []
        for img_path in self.imgs[index]:
            img = Image.open(img_path)
            img = self.transformer(img)
            if self.minus_point_5:
                img = img - 0.5  # from [0, 1] -> [-0.5, 0.5]
            img = self.normalizer(img)
            img = img.unsqueeze(0)
            img_seq.append(img)
        img_seq = torch.cat(img_seq, 0)

        depth_img_seq = []
        for img_path in self.depth_imgs[index]:
            img = Image.open(img_path)
            img = self.transformer(img)
            # if self.minus_point_5:
            #     img = img - 0.5  # from [0, 1] -> [-0.5, 0.5]
            # img = self.gray_normalizer(img)
            img = img.unsqueeze(0)
            depth_img_seq.append(img)
        depth_img_seq = torch.cat(depth_img_seq, 0)

        sem_img_seq = []
        for img_path in self.semantic_imgs[index]:
            img = Image.open(img_path)
            img = self.transformer(img)
            # if self.minus_point_5:
            #     img = img - 0.5  # from [0, 1] -> [-0.5, 0.5]
            # img = self.normalizer(img)
            img = img.unsqueeze(0)
            sem_img_seq.append(img)
        sem_img_seq = torch.cat(sem_img_seq, 0)

        # camera poses
        # the first number in camera poses is the frame id.
        poses_6d = np.hsplit(np.asarray(self.camera_pose_6d[index]), np.array([1]))[-1]
        raw_poses = np.asarray(self.camera_pose[index])

        # opposite rotation of the first frame
        first_R = raw_poses[0].reshape((4, 4))[:3, :3].T

        # get relative pose w.r.t. the first frame in the sequence
        poses_6d[1:] = poses_6d[1:] - poses_6d[0]

        # rotate the sequence relative to the first frame
        for pose_6d in poses_6d[1:]:
            location = first_R.dot(pose_6d[3:])
            pose_6d[3:] = location[:]

        # get relative pose w.r.t. previous frame
        poses_6d[2:] = poses_6d[2:] - poses_6d[1:-1]

        # consider cases when rotation angles over Y axis go through PI -PI discontinuity
        for pose_6d in poses_6d[1:]:
            pose_6d[0] = self.normalize_angle_delta(pose_6d[0])

        camera_pose_seq = poses_6d

        return img_seq, depth_img_seq, sem_img_seq, camera_pose_seq

    def __len__(self):
        return self.data_len

    def p_to_se3(self, p):
        SE3 = np.array([
            [p[0], p[1], p[2], p[3]],
            [p[4], p[5], p[6], p[7]],
            [p[8], p[9], p[10], p[11]],
            [0, 0, 0, 1]
        ])
        return SE3

    def get_ground_6d_poses(self, p, p2):
        """ For 6dof pose representaion """
        SE1 = self.p_to_se3(p)
        SE2 = self.p_to_se3(p2)

        SE12 = np.matmul(np.linalg.inv(SE1), SE2)

        pos = np.array([SE12[0][3], SE12[1][3], SE12[2][3]])
        angles = self.rotation_matrix_to_euler_angles(SE12[:3, :3])
        return np.concatenate((angles, pos))    # rpyxyz

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

    def normalize_angle_delta(self, angle):
        if (angle > np.pi):
            angle = angle - 2 * np.pi
        elif (angle < -np.pi):
            angle = 2 * np.pi + angle
        return angle


# def get_ground_6d_poses_quat(p, p2):
#     """ For 6dof pose representaion """
#     SE1 = p_to_se3(p)
#     SE2 = p_to_se3(p2)
#
#     SE12 = np.matmul(np.linalg.inv(SE1), SE2)
#
#     pos = np.array([SE12[0][3], SE12[1][3], SE12[2][3]])
#     quat = rotation_matrix_to_quaternion(SE12[:3, :3])
#     return np.concatenate((quat, pos))    # qxyz
#
#
# def rotation_matrix_to_quaternion(R):
#     assert (is_rotation_matrix(R))
#
#     qw = np.sqrt(1 + np.sum(np.diag(R))) / 2.0
#     qx = (R[2, 1] - R[1, 2]) / (4 * qw)
#     qy = (R[0, 2] - R[2, 0]) / (4 * qw)
#     qz = (R[1, 0] - R[0, 1]) / (4 * qw)
#
#     return np.array([qw, qx, qy, qz])


