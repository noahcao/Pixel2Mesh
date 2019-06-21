import pickle

import tensorflow as tf

from p2m.config import update_config
from p2m.dataset import DataFetcher
from p2m.models.gcn import GCN
import numpy as np

from p2m.utils import construct_feed_dict

import argparse
import os
import pprint


def parse_args():
    parser = argparse.ArgumentParser(description='Pixel2Mesh Train Entrypoint')
    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)

    args, rest = parser.parse_known_args()
    update_config(args.cfg)

    # training
    parser.add_argument('--frequent', help='frequency of logging', default=config.PRINT_FREQ, type=int)
    parser.add_argument('--batch-size', help='batch size', type=int)
    parser.add_argument("--checkpoint", help="checkpoint file", type=str)

    args = parser.parse_args()

    return args


def reset_config(config, args):
    if args.batch_size:
        config.TRAIN.BATCH_SIZE = config.TEST.BATCH_SIZE = args.batch_size
    if args.frequent:
        config.PRINT_FREQ = args.frequent


def main():
    args = parse_args()
    reset_config(config, args)

    logger, final_output_dir, tb_log_dir = create_logger(config, 'train')

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED

    model = sppe.networks.get_pose_net(config, is_train=True)

    if args.checkpoint:
        print("=> Loading checkpoint:", args.checkpoint)
        checkpoint = torch.load(args.checkpoint)
        config.TRAIN.BEGIN_EPOCH = checkpoint["epoch"]
        model.load_state_dict(checkpoint["state_dict"])
    else:
        checkpoint = None

    # copy model file
    # TODO: copy current files

    # workaround: log_dir -> logdir
    if 'log_dir' in SummaryWriter.__init__.__code__.co_varnames:
        writer = SummaryWriter(log_dir=tb_log_dir)
    elif 'logdir' in SummaryWriter.__init__.__code__.co_varnames:
        writer = SummaryWriter(logdir=tb_log_dir)
    else:
        raise NotImplementedError

    writer_dict = {
        'writer': writer,
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    # dump_input = torch.rand((config.TRAIN.BATCH_SIZE,
    #                          3,
    #                          config.MODEL.IMAGE_SIZE[1],
    #                          config.MODEL.IMAGE_SIZE[0]))
    # writer_dict['writer'].add_graph(model, (dump_input,), verbose=False)

    gpus = [int(i) for i in config.GPUS.split(',')]
    model = torch.nn.DataParallel(model, device_ids=gpus).cuda()

    # define loss function (criterion) and optimizer
    if config.LOSS.TARGET_TYPE == "gaussian":
        criterion = JointsMSELoss(use_target_weight=config.LOSS.USE_TARGET_WEIGHT).cuda()
    elif config.LOSS.TARGET_TYPE == "argmax":
        criterion = SoftArgMaxLoss(use_target_weight=config.LOSS.USE_TARGET_WEIGHT).cuda()

    optimizer = get_optimizer(config, model)

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, config.TRAIN.LR_STEP, config.TRAIN.LR_FACTOR
    )

    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])

    # Data loading code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    if config.DATASET.DATASET == "mpii":
        config_dataset_class = dataset.MPIIDataset
    else:
        raise NotImplementedError

    train_dataset = config_dataset_class(
        config,
        config.DATASET.ROOT,
        config.DATASET.TRAIN_SET,
        True,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )
    valid_dataset = config_dataset_class(
        config,
        config.DATASET.ROOT,
        config.DATASET.TEST_SET,
        False,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.TRAIN.BATCH_SIZE * len(gpus),
        shuffle=config.TRAIN.SHUFFLE,
        num_workers=config.WORKERS,
        pin_memory=True
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config.TEST.BATCH_SIZE * len(gpus),
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True
    )

    best_perf = 0.0
    for epoch in range(config.TRAIN.BEGIN_EPOCH, config.TRAIN.END_EPOCH):
        lr_scheduler.step()

        # train for one epoch
        train(config, train_loader, model, criterion, optimizer, epoch,
              final_output_dir, tb_log_dir, writer_dict)

        # evaluate on validation set
        perf_indicator = validate(config, valid_loader, valid_dataset, model,
                                  criterion, final_output_dir, tb_log_dir,
                                  writer_dict)

        if perf_indicator > best_perf:
            best_perf = perf_indicator
            best_model = True
        else:
            best_model = False

        logger.info('=> saving checkpoint to {}'.format(final_output_dir))
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.module.state_dict(),
            'perf': perf_indicator,
            'optimizer': optimizer.state_dict(),
        }, best_model, final_output_dir)

    final_model_state_file = os.path.join(final_output_dir,
                                          'final_state.pth.tar')
    logger.info('saving final model state to {}'.format(final_model_state_file))
    torch.save(model.module.state_dict(), final_model_state_file)
    writer_dict['writer'].close()


if __name__ == '__main__':
    main()

if __name__ == "__main__":
    seed = 1024
    np.random.seed(seed)
    tf.set_random_seed(seed)

    config =

    # Define placeholders(dict) and model
    num_blocks = 3
    num_supports = 2
    placeholders = {
        'features': tf.placeholder(tf.float32, shape=(None, 3)),
        'img_inp': tf.placeholder(tf.float32, shape=(224, 224, 3)),
        'labels': tf.placeholder(tf.float32, shape=(None, 6)),
        'support1': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
        'support2': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
        'support3': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
        'faces': [tf.placeholder(tf.int32, shape=(None, 4)) for _ in range(num_blocks)],  # for face loss, not used.
        'edges': [tf.placeholder(tf.int32, shape=(None, 2)) for _ in range(num_blocks)],
        'lape_idx': [tf.placeholder(tf.int32, shape=(None, 10)) for _ in range(num_blocks)],  # for laplace term
        'pool_idx': [tf.placeholder(tf.int32, shape=(None, 2)) for _ in range(num_blocks - 1)]  # for unpooling
    }
    model = GCN(placeholders, logging=True)

    # Load data, initialize session
    data = DataFetcher(FLAGS.data_list)
    data.setDaemon(True)  ####
    data.start()
    config = tf.ConfigProto()
    # config.gpu_options.allow_growth=True
    config.allow_soft_placement = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    # model.load(sess)

    # Train graph model
    train_loss = open('record_train_loss.txt', 'a')
    train_loss.write('Start training, lr =  %f\n' % (FLAGS.learning_rate))
    with open('Data/ellipsoid/info_ellipsoid.dat', 'rb') as info_ellipsoid:
        pkl = pickle.load(info_ellipsoid)
    feed_dict = construct_feed_dict(pkl, placeholders)

    train_number = data.number
    for epoch in range(FLAGS.epochs):
        all_loss = np.zeros(train_number, dtype='float32')
        print("Epoch %d, expected total iters = %d" % (epoch + 1, train_number))
        for iters in range(train_number):
            # Fetch training data
            img_inp, y_train, data_id = data.fetch()
            feed_dict.update({placeholders['img_inp']: img_inp})
            feed_dict.update({placeholders['labels']: y_train})

            # Training step
            _, dists, out1, out2, out3 = sess.run(
                [model.opt_op, model.loss, model.output1, model.output2, model.output3],
                feed_dict=feed_dict)
            all_loss[iters] = dists
            mean_loss = np.mean(all_loss[np.where(all_loss)])
            if (iters + 1) % 128 == 0:
                print('Epoch %d, Iteration %d' % (epoch + 1, iters + 1))
                print('Mean loss = %f, iter loss = %f, %d' % (mean_loss, dists, data.queue.qsize()))
                sys.stdout.flush()
        # Save model
        model.save(sess)
        train_loss.write('Epoch %d, loss %f\n' % (epoch + 1, mean_loss))
        train_loss.flush()

    data.shutdown()
    print('Training Finished!')
