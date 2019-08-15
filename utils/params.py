import argparse


def set_params():
    parser = argparse.ArgumentParser(description='Depth Estimation using Monocular Images')
    parser.add_argument('--dataset', type=str, default='nyu', choices=['nyu', 'kitti'])
    parser.add_argument('--data_path', type=str, default='data/nyuv2')
    parser.add_argument('--class_num', type=int, default=10,  help='the number of classes in segmantic')
    parser.add_argument('--weight', default='equal', type=str, help='multi-task weighting: equal, uncert, dwa')
    parser.add_argument('--temp', default=2.0, type=float, help='temperature for DWA (must be positive)')

    # train
    parser.add_argument('--use_cuda', type=bool, default=True)
    parser.add_argument('--gpu_ids', type=str, default=['0'], help='IDs of GPUs to use')
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--seq_len', type=int, default=6)
    parser.add_argument('--overlap', type=int, default=1, help='number of overlap when select data')
    parser.add_argument('--sample_times', type=int, default=1, help='sampling times for the dataset')
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
    parser.add_argument('--crop_size', type=float, default=(320, 960), help='Maximum of input depths')

    # model
    parser.add_argument('--task_num', type=int, default=3, help='task number train in this model')
    parser.add_argument('--backbone', type=str, default='resnext', choices=['resnext', 'sknet'], help='encoder model')
    parser.add_argument('--model_name', type=str, default='mtan', choices=['mtan'])
    parser.add_argument('--pretrained_net', type=str, default=None)
    parser.add_argument('--use_pretrain', type=bool, default=False)
    parser.add_argument('--resume', type=str, default=None, help='Start training from an existing model.')
    parser.add_argument('--save_path', type=str, default='train_results/')

    args = parser.parse_args()
    return args






