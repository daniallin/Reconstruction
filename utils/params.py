import argparse


def set_params():
    parser = argparse.ArgumentParser(description='Depth Estimation using Monocular Images')
    parser.add_argument('--dataset', type=str, default='vkitti', choices=['nyu', 'vkitti'])
    parser.add_argument('--nyu_path', type=str, default='dataset/nyuv2')
    parser.add_argument('--vkitti_path', type=str, default='dataset/vkitti/')
    parser.add_argument('--vkitti_datainfo', type=str, default='datainfo/vkitti/')
    parser.add_argument('--class_num', type=int, default=14,  help='the number of classes in segmantic')
    parser.add_argument('--weight', default='equal', type=str, help='multi-task weighting: equal, uncert, dwa')
    parser.add_argument('--temp', default=2.0, type=float, help='temperature for DWA (must be positive)')
    parser.add_argument('--img_mean', default=(0.4695473891639639, 0.5105454725388299, 0.4300407379334988), type=tuple)
    parser.add_argument('--img_std', default=(0.288617005264684, 0.2911478421011405, 0.2964820763906757), type=tuple)
    parser.add_argument('--minus_point_5', default=False, type=bool)
    parser.add_argument('--crop_size', type=float, default=(320, 480))
    parser.add_argument('--resize_mode', default='crop', type=str)

    # train
    parser.add_argument('--use_cuda', type=bool, default=True)
    parser.add_argument('--gpu_ids', type=str, default=['0'], help='IDs of GPUs to use')
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--sep_optim', type=bool, default=True)
    parser.add_argument('--depth_lr', type=float, default=0.0001)
    parser.add_argument('--seg_lr', type=float, default=0.002)
    parser.add_argument('--vo_lr', type=float, default=0.0005)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--seq_len', type=int, default=6)
    parser.add_argument('--overlap', type=int, default=1, help='number of overlap when select data')
    parser.add_argument('--sample_times', type=int, default=3, help='sampling times for the dataset')
    parser.add_argument('--sync_bn', type=bool, default=False, help='whether using sync batch normalization')
    parser.add_argument('--rnn_hidden_size', type=int, default=1000)
    parser.add_argument('--rnn_dropout_between', type=float, default=0.)
    parser.add_argument('--rnn_dropout_out', type=float, default=0.5)
    parser.add_argument('--output_scale', type=int, default=16, help='output scale of encoder')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=1e-3, help='w-decay (default: 5e-4)')
    parser.add_argument('--min_depth', type=float, default=10.0, help='Minimum of input depths')
    parser.add_argument('--max_depth', type=float, default=1000.0, help='Maximum of input depths')

    # model
    parser.add_argument('--freeze_bn', type=bool, default=False)
    parser.add_argument('--task_num', type=int, default=3, help='task number train in this model')
    parser.add_argument('--backbone', type=str, default='resnext', choices=['resnext', 'sknet'], help='encoder model')
    parser.add_argument('--model_name', type=str, default='reconstruct_mtn', choices=['mtan', 'reconstruct_mtn'])
    parser.add_argument('--pretrained_net', type=str, default=None)
    parser.add_argument('--use_pretrain', type=bool, default=True, help='whether using pretrained encoder network')
    parser.add_argument('--resume', type=str, default='train_results/2019-09-04-21-36-vkitti/checkpoint.pth', help='Start training from an existing model.')
    parser.add_argument('--save_path', type=str, default='train_results/')

    args = parser.parse_args()
    return args



# Numbers of frames in training dataset: 2126
# mean_tensor =  [0.4695473891639639, 0.5105454725388299, 0.4300407379334988]
# mean_np =  [119.73433290361544, 130.1889760160859, 109.6601286780393]
# std_tensor =  [0.288617005264684, 0.2911478421011405, 0.2964820763906757]
# std_np =  [73.59734829470447, 74.24285335838441, 75.60289938939736]


