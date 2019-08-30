import os
import time
from tqdm import tqdm

from utils.params import set_params
from utils.helper import set_random_seed, AverageMeter
from utils.keeper import Keeper
from utils.loss import *
from models import build_model
from dataload import data_loader


def main(args):
    model = build_model(args.model_name, args)
    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

    log.info('LOSS FORMAT: SEMANTIC_LOSS MEAN_IOU PIX_ACC | DEPTH_LOSS ABS_ERR REL_ERR'
             ' | NORMAL_LOSS MEAN MED <11.25 <22.5 <30\n')

    log.info('loading data...\n')
    train_loader, val_loader = data_loader(args)
    train_bts, val_bts = len(train_loader), len(val_loader)
    log.info('train batch number: {0}; validation batch number: {1}'.format(train_bts, val_bts))

    # Whether using checkpoint
    if args.resume is not None:
        if not os.path.exists(args.resume):
            raise RuntimeError("=> no checkpoint found")
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['state_dict'])

        optimizer.load_state_dict(checkpoint['optimizer'])
        chk_loss = checkpoint['chk_loss']
        best_loss = checkpoint['best_loss']
        args.start_epoch = checkpoint['epoch'] + 1
    else:
        best_loss = np.inf

    # whether using pretrained model
    if args.pretrained_net is not None and args.resume is None:
        pretrained_w = torch.load(args.pretrained_net)
        model_dict = model.state_dict()
        pretrained_dict = {k: torch.from_numpy(v) for k, v in pretrained_w.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    model = model.cuda() if args.use_cuda else model

    # -------------------- training -------------------- #
    alpha_weight = np.ones([3, args.epochs])
    T = args.temp
    for epoch in range(args.epochs):
        e_time = time.time()
        log.info('training: epoch {}/{} \n'.format(epoch+1, args.epochs))

        model.train()
        cost = np.zeros(16, dtype=np.float32)
        avg_cost = np.zeros(16, dtype=np.float32)

        # apply Dynamic Weight Average
        if args.weight == 'dwa':
            if epoch == 0 or epoch == 1:
                alpha_weight[:, epoch] = 1.0
            else:
                w_1 = avg_cost[epoch - 1, 0] / avg_cost[epoch - 2, 0]
                w_2 = avg_cost[epoch - 1, 3] / avg_cost[epoch - 2, 3]
                w_3 = avg_cost[epoch - 1, 6] / avg_cost[epoch - 2, 6]
                alpha_weight[0, epoch] = 3 * np.exp(w_1 / T) / (np.exp(w_1 / T) + np.exp(w_2 / T) + np.exp(w_3 / T))
                alpha_weight[1, epoch] = 3 * np.exp(w_2 / T) / (np.exp(w_1 / T) + np.exp(w_2 / T) + np.exp(w_3 / T))
                alpha_weight[2, epoch] = 3 * np.exp(w_3 / T) / (np.exp(w_1 / T) + np.exp(w_2 / T) + np.exp(w_3 / T))

        for k, (train_img, train_depth, train_sem, train_pose) in enumerate(tqdm(train_loader)):
            train_sem = train_sem.type(torch.LongTensor)
            train_depth = train_depth.type(torch.FloatTensor)
            batch_size = train_img.size(0)
            seq_len = train_img.size(1)
            # print('train depth size is {}'.format(train_depth.size()))
            # print('train semantic size is {}'.format(train_sem.size()))
            train_depth = train_depth.view(batch_size * seq_len, train_depth.size(2), train_depth.size(3), train_depth.size(4))
            train_sem = train_sem.view(batch_size * seq_len, train_sem.size(3), train_sem.size(4))
            if args.use_cuda:
                train_img, train_depth, train_sem, train_pose = \
                    train_img.cuda(), train_depth.cuda(), train_sem.cuda(), train_pose.cuda()

            optimizer.zero_grad()

            train_preds, logsigma = model(train_img)

            train_losses = get_mtn_loss(train_preds, (train_depth, train_sem, train_pose))

            if args.weight == 'equal' or args.weight == 'dwa':
                train_loss = torch.mean(sum(alpha_weight[i, epoch] * train_losses[i] for i in range(3)))
            else:
                train_loss = sum(1 / (2 * torch.exp(logsigma[i])) * train_losses[i] + logsigma[i] / 2 for i in range(3))

            train_loss.backward()
            optimizer.step()

            log.info('train loss of batch/epoch {}/{} is {}'.format(epoch, k, train_loss))
            cost[0] = train_losses[0].item()
            cost[1], cost[2] = depth_error(train_preds[0], train_depth)

            cost[3] = train_losses[1].item()
            cost[4] = get_miou(train_preds[1], train_sem, class_num=args.class_num).item()
            cost[5] = get_iou(train_preds[1], train_sem).item()
            cost[6] = train_losses[2].item()
            cost[7] = train_loss
            avg_cost[:8] += cost[:8] / train_bts

        # evaluating test data
        model.eval()
        with torch.no_grad():
            for k, (val_img, val_depth, val_sem, val_pose) in enumerate(tqdm(val_loader)):
                val_sem = val_sem.type(torch.LongTensor)
                val_depth = val_depth.type(torch.FloatTensor)
                batch_size = val_img.size(0)
                seq_len = val_img.size(1)
                # print('train depth size is {}'.format(train_depth.size()))
                # print('train semantic size is {}'.format(train_sem.size()))
                val_depth = val_depth.view(batch_size * seq_len, val_depth.size(2), val_depth.size(3), val_depth.size(4))
                val_sem = val_sem.view(batch_size * seq_len, val_sem.size(3), val_sem.size(4))
                if args.use_cuda:
                    val_img, val_depth, val_sem, val_pose = \
                        val_img.cuda(), val_depth.cuda(), val_sem.cuda(), val_pose.cuda()

                val_preds, val_logsigma = model(val_img)
                val_losses = get_mtn_loss(val_preds, (val_depth, val_sem, val_pose))

                if args.weight == 'equal' or args.weight == 'dwa':
                    val_loss = torch.mean(sum(alpha_weight[i, epoch] * val_losses[i] for i in range(3)))
                else:
                    val_loss = sum(
                        1 / (2 * torch.exp(val_logsigma[i])) * val_losses[i] + val_logsigma[i] / 2 for i in range(3))

                cost[8] = val_losses[0].item()
                cost[9], cost[10] = depth_error(val_preds[0], val_depth)

                cost[11] = val_losses[1].item()
                cost[12] = get_miou(val_preds[1], val_sem, class_num=args.class_num).item()
                cost[13] = get_iou(val_preds[1], val_sem).item()
                cost[14] = val_losses[2].item()
                cost[15] = val_loss
                avg_cost[8:] += cost[8:] / val_bts

        print(
            'Epoch: {:04d} | TRAIN: {:.4f} {:.4f} {:.4f} | {:.4f} {:.4f} {:.4f} | {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}'
            'TEST: {:.4f} {:.4f} {:.4f} | {:.4f} {:.4f} {:.4f} | {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} '
            .format(epoch, *avg_cost[epoch, :]))
        keeper.save_loss(avg_cost.cpu().numpy(), 'losses.csv')

        if avg_cost[-1] < best_loss:
            best_loss = avg_cost[-1]
            keeper.save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_loss': best_loss,
            }, 'best_model.pth')

        keeper.save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'chk_loss': avg_cost[12],
            'best_loss': best_loss,
        })

        log.info('training time of epoch {}/{} is {} \n'.format(epoch + 1, args.epochs, time.time() - e_time))


if __name__ == '__main__':
    set_random_seed()
    args = set_params()

    keeper = Keeper(args)
    log = keeper.setup_logger()
    log.info('Welcome to summoner\'s rift')

    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(e) for e in args.gpu_ids)
    args.use_cuda = torch.cuda.is_available()

    print(args)
    keeper.save_experiment_config()

    start_time = time.time()

    log.info("Thirty seconds until minion spawn!")

    main(args)

    log.info('Victory! Total game time is: {}'.format(time.time()-start_time))

