import argparse
import ast
import logging
import os

import numpy as np
import paddle.fluid as fluid

from config import parse_config, merge_configs, print_configs
from model import ECO
from reader import KineticsReader

logging.root.handlers = []
FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
logging.basicConfig(filename='logger.log', level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser("Paddle Video train script")
    parser.add_argument(
        '--model_name',
        type=str,
        default='tsn',
        help='name of model to train.')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/tsn.txt',
        help='path to config file of model')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=None,
        help='training batch size. None to use config file setting.')
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=None,
        help='learning rate use for training. None to use config file setting.')
    parser.add_argument(
        '--pretrain',
        type=str,
        default=True,
        help='path to pretrain weights. None to use default weights path in  ~/.paddle/weights.'
    )
    parser.add_argument(
        '--use_gpu',
        type=ast.literal_eval,
        default=True,
        help='default use gpu.')
    parser.add_argument(
        '--epoch',
        type=int,
        default=100,
        help='epoch number, 0 for read from config file')
    parser.add_argument(
        '--save_dir',
        type=str,
        default='checkpoints_models',
        help='directory name to save train snapshoot')
    parser.add_argument(
        '--gpu_num',
        type=ast.literal_eval,
        default=1,
        help='default gpu num.')
    parser.add_argument(
        '--gd',
        type=int,
        default=50,
        help='clip gradient')
    parser.add_argument(
        '--selected_gpus',
        type=str,
        default='0,1',
        help='selected_gpus')
    parser.add_argument(
        '--num_saturate',
        type=int,
        default='5',
        help='num_saturate')
    parser.add_argument(
        '--eval_freq',
        type=int,
        default='1',
        help='eval_freq')
    parser.add_argument("--use_paddlecloud", type=bool, default=False)
    parser.add_argument("--cluster_node_ips", type=str, default="10.163.143.6")
    parser.add_argument("--node_ip", type=str, default="10.163.143.6", help='The?current?node?ip.')
    parser.add_argument(
        "--started_port", type=int, default=0, help="The?trainer's?started?port?on?a?single?node")
    parser.add_argument("--log_level", type=int, default=20, help='Logging?level,?default?is?logging.INFO')
    parser.add_argument("--log_dir", type=str, default="/home/w00445241/project_set/project_paddle/Eco/TSN/log_dir")
    parser.add_argument("--distributed", type=str, default=True)
    parser.add_argument("--dataset_base_path", type=str, default="/home/w00445241/project_set/project_paddle/UCF-101/")
    parser.add_argument("--output_base_path", type=str,
                        default="/home/w00445241/project_set/project_paddle/Eco/TSN/checkpoints_models/ECO_distributed_1/")
    args = parser.parse_args()
    return args


def train(args, distributed):
    #===================== GPU CONF =====================#
    if distributed:
        # if run on parallel mode
        place = fluid.CUDAPlace(fluid.dygraph.parallel.Env().dev_id)
    else:
        # if run on single GPU mode, and select gpu number.
        args.use_gpu = True
        place = fluid.CUDAPlace(args.gpu_num) if args.use_gpu else fluid.CPUPlace()
    # ===================== Dygraph Mode =====================#
    with fluid.dygraph.guard(place):
        # leverage from TSN training script
        config = parse_config(args.config)
        train_config = merge_configs(config, 'train', vars(args))
        val_config = merge_configs(config, 'valid', vars(args))
        print_configs(train_config, 'Train')

        # ===================== Init ECO =====================#
        train_model = ECO.ECO(num_classes=train_config['MODEL']['num_classes'],
                              num_segments=train_config['MODEL']['seg_num'])
        if distributed:
            strategy = fluid.dygraph.parallel.prepare_context()
            train_model = fluid.dygraph.parallel.DataParallel(train_model, strategy)

        # trick 1: use clip gradient method to avoid gradient explosion
        if args.gd is not None:
            clip = fluid.clip.GradientClipByGlobalNorm(clip_norm=args.gd)
            print('clip:', clip)

        # ===================== Init Optimizer =====================#
        # optimizer config: use momentum, nesterov, weight decay, lr decay
        learning_rate = 0.001
        opt = fluid.optimizer.Momentum(learning_rate, 0.9,
                                       parameter_list=train_model.parameters(),
                                       use_nesterov=True,
                                       regularization=fluid.regularizer.L2Decay(regularization_coeff=5e-4),
                                       grad_clip=clip)
        # trick 2: Freezing BatchNorm2D except the first one.
        # trick 3: make all weight layer lr mult as 1, bias lr mult as 2.
        get_optim_policies(opt)
        print('get_optim_policies:--batch_norm_0.w_0', opt._parameter_list[2].optimize_attr,opt._parameter_list[2].stop_gradient)
        print('get_optim_policies:--batch_norm_0.b_0', opt._parameter_list[3].optimize_attr,opt._parameter_list[2].stop_gradient)

        # ===================== Use Pretrained Model =====================#
        # use pretrained model: ECO_Full_rgb_model_Kinetics.pth 2.tar(download from MZO git)
        # then transform it from torch to paddle weight except fc layer.
        if args.pretrain:
            model, _ = fluid.dygraph.load_dygraph(args.save_dir + '/ECO_FULL_RGB_seg16')
            # also tried using pretrained model on torch, 32F-92.9%,16F-91.8% precision trained on torch
            # model, _ = fluid.dygraph.load_dygraph(args.save_dir + '/eco_91.81_model_best')
            train_model.load_dict(model)

        # build model
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)

        # ===================== Init Data Reader =====================#
        # leverage from TSN training script
        train_config.TRAIN.batch_size = train_config.TRAIN.batch_size
        train_reader = KineticsReader('ECO', 'train', train_config).create_reader()
        print('train_reader', train_reader)
        val_reader = KineticsReader('ECO', 'valid', val_config).create_reader()
        if distributed:
            train_reader = fluid.contrib.reader.distributed_batch_reader(train_reader)

        # ===================== Init Trick Params =====================#
        epochs = args.epoch or train_model.epoch_num()
        loss_summ = 0
        saturate_cnt = 0
        exp_num = 0
        best_prec1 = 0

        for i in range(epochs):
            train_model.train()
            # trick 4: Saturate lr decay: different from lr piecewise decay or others
            # calculate prec every epoch, if prec1 does not rise for 5 times(named model saturated), then use decay lr.
            if saturate_cnt == args.num_saturate:
                exp_num = exp_num + 1
                saturate_cnt = 0
                decay = 0.1 ** (exp_num)
                learning_rate = learning_rate * decay
                opt = fluid.optimizer.Momentum(learning_rate, 0.9,
                                               parameter_list=train_model.parameters(),
                                               use_nesterov=True,
                                               regularization=fluid.regularizer.L2Decay(regularization_coeff=5e-4),
                                               grad_clip=clip)
                print('get_optim_policies:--batch_norm_0.w_0', opt._parameter_list[2].optimize_attr,
                      opt._parameter_list[2].stop_gradient)
                print('get_optim_policies:--batch_norm_0.b_0', opt._parameter_list[3].optimize_attr,
                      opt._parameter_list[2].stop_gradient)
                print("- Learning rate decreases by a factor of '{}'".format(10 ** (exp_num)))
            
            for batch_id, data in enumerate(train_reader()):
                lr = opt.current_step_lr()
                print('lr:', lr)  # check lr every batch ids
                dy_x_data = np.array([x[0] for x in data]).astype('float32')
                y_data = np.array([[x[1]] for x in data]).astype('int64')

                img = fluid.dygraph.to_variable(dy_x_data)
                label = fluid.dygraph.to_variable(y_data)
                label.stop_gradient = True

                out, acc = train_model(img, label)
                loss = fluid.layers.cross_entropy(out, label)
                avg_loss = fluid.layers.mean(loss)
                loss_summ += avg_loss
                if distributed:
                    avg_loss = train_model.scale_loss(avg_loss)
                avg_loss.backward()
                if distributed:
                    train_model.apply_collective_grads()

                if (batch_id + 1) % 4 == 0:
                    # trick 5: scale down gradients when iter size is functioning every 4 batches
                    opt.minimize(loss_summ)
                    opt.clear_gradients()
                    loss_summ = 0

                if batch_id % 1 == 0:
                    logger.info(
                        "Loss at epoch {} step {}: {}, acc: {}".format(i, batch_id, avg_loss.numpy(), acc.numpy()))
                    print("Loss at epoch {} step {}: {}, acc: {}".format(i, batch_id, avg_loss.numpy(), acc.numpy()))

            if (i + 1) % args.eval_freq == 0 or i == args.epochs - 1:
                train_model.eval()
                acc_list = []
                false_class = []

                for batch_id, data in enumerate(val_reader()):
                    dy_x_data = np.array([x[0] for x in data]).astype('float32')
                    y_data = np.array([[x[1]] for x in data]).astype('int64')

                    img = fluid.dygraph.to_variable(dy_x_data)
                    label = fluid.dygraph.to_variable(y_data)
                    label.stop_gradient = True

                    out, acc = train_model(img, label)
                    if acc.numpy()[0] != 1:
                        false_class.append(label.numpy()[0][0])
                    acc_list.append(acc.numpy()[0])
                    print(batch_id, 'acc:', np.mean(acc_list))
                    if len(false_class) == 0:
                        continue
                print("validate set acc:{}".format(np.mean(acc_list)))
                prec1 = np.mean(acc_list)
                # remember best prec@1 and save checkpoint
                is_best = prec1 > best_prec1
                if is_best:
                    saturate_cnt = 0
                    fluid.dygraph.save_dygraph(train_model.state_dict(),
                                               args.save_dir + '/ECO_FULL_1/' + str(i) + '_best_' + str(prec1))
                else:
                    saturate_cnt = saturate_cnt + 1

                print("- Validation Prec@1 saturates for {} epochs.".format(saturate_cnt), best_prec1)
                best_prec1 = max(prec1, best_prec1)

        logger.info("Final loss: {}".format(avg_loss.numpy()))
        print("Final loss: {}".format(avg_loss.numpy()))

def get_optim_policies(opt):
    # num = 0
    for i, param in enumerate(opt._parameter_list):
        if 'batch_norm' in param.name: # bn not particepate in training process
            if 'batch_norm_0.w_0' in param.name: #except the first res 3d part
                param.optimize_attr['learning_rate'] = 1
                # param.stop_gradient = False
                continue
            if 'batch_norm_0.b_0' in param.name:
                param.optimize_attr['learning_rate'] = 2
                # param.stop_gradient = False
                continue
            else:
                if i >= 414 and i <= 484:
                    continue
                else:
                    param.stop_gradient = True
            continue
        if 'w' in param.name:
            param.optimize_attr['learning_rate'] = 1
        if 'b' in param.name:
            param.optimize_attr['learning_rate'] = 2
    print('freeze bn2d')


if __name__ == "__main__":
    import paddle.distributed.launch as launch

    args = parse_args()

    distributed = False
    if distributed:
        import paddle.distributed.launch as launch

        args.training_script = "train.py"
        args.training_script_args = ["--distributed", "--dataset_base_path",
                                     "/home/w00445241/project_set/project_paddle/UCF-101/", "--output_base_path",
                                     "/home/w00445241/project_set/project_paddle/Eco/TSN/checkpoints_models/ECO_distributed_1/"]
        print(args.log_dir, args.training_script, args.training_script_args)
        launch.launch(args)

    logger.info(args)

    train(args, distributed)
