from _collections import OrderedDict

import numpy
import paddle.fluid as fluid
import torch

from model import ECO

# Model Transform script, transform from torch to paddle.
path = 'checkpoints_models/ECO_Full_rgb_model_Kinetics.pth 2.tar'
save_path = 'checkpoints_models/ECO_FULL_RGB_seg16'
torch_weight = torch.load(path, map_location=torch.device('cpu'))
torch_weight = torch_weight['state_dict']
print('loaded')

num = 0
for torch_key in torch_weight:
    if 'bn.num_batches_tracked' in torch_key:
        num += 1
    print(torch_key)
    print(num)

with fluid.dygraph.guard():
    paddle_model = ECO.ECO(num_classes=101, num_segments=16)
    paddle_weight = paddle_model.state_dict()
    new_weight_dict = OrderedDict()
    matched_bn_var = 0
    matched_bn_mean = 0
    matched_fc = 0
    matched_base = 0
    matched_linear = 0
    for paddle_key in paddle_weight.keys():
        print('paddle:', paddle_key)
        if len(paddle_key.split('.')) == 3:  # sub module
            torch_key = 'module.base_model.' + paddle_key.split('.')[1] + '.' + paddle_key.split('.')[2]
            name = 'inception'
        elif len(paddle_key.split('.')) == 4:
            torch_key = 'module.base_model.' + paddle_key.split('.')[2] + '.' + paddle_key.split('.')[3]
            name = '3d'
        else:  # main module
            torch_key = 'module.base_model.' + paddle_key.split('.')[0]
            name = 'ECO'
        print('paddle:', paddle_key.split('.'), len(paddle_key.split('.')))
        if not 'linear' in paddle_key or not 'fc' in paddle_key:  # fc need to Transpose
            # NOT FC PART, bn,var,mean
            if '_bn._mean' in paddle_key:
                if name == '3d':
                    key_name = paddle_key.split('.')[2]
                elif name == 'inception':
                    key_name = paddle_key.split('.')[1]
                else:
                    key_name = paddle_key.split('.')[0]
                torch_key = 'module.base_model.' + key_name + '.running_mean'
                print('_bn._mean:\n', paddle_key, '\n', torch_key)
                new_weight_dict[paddle_key] = torch_weight[torch_key].detach().numpy().astype(numpy.float32)
                matched_bn_mean += 1
            elif '_bn._variance' in paddle_key:
                if name == '3d':
                    key_name = paddle_key.split('.')[2]
                elif name == 'inception':
                    key_name = paddle_key.split('.')[1]
                else:
                    key_name = paddle_key.split('.')[0]
                torch_key = 'module.base_model.' + key_name + '.running_var'
                print('_bn._variance:\n', paddle_key, '\n', torch_key)
                new_weight_dict[paddle_key] = torch_weight[torch_key].detach().numpy().astype(numpy.float32)
                matched_bn_var += 1

            else:
                print('not linear:\n', paddle_key, '\n', torch_key)
                if torch_key == 'module.base_model.fc_final':
                    new_weight_dict[paddle_key] = paddle_weight[paddle_key]
                    matched_fc += 1
                    print('matched_fc_final:', paddle_key)

                    continue
                elif torch_key == 'module.base_model.fc_0':
                    new_weight_dict[paddle_key] = paddle_weight[paddle_key]
                    matched_fc += 1
                    print('matched_fc0:', paddle_key)
                    continue
                else:
                    new_weight_dict[paddle_key] = torch_weight[torch_key].detach().numpy().astype(numpy.float32)
                    matched_base += 1

        else:
            print('linear:\n', paddle_key, '\n', torch_key)
            print(paddle_key, torch_key)
            # new_weight_dict[paddle_key]=torch_weight[paddle_key].detach().numpy().astype(numpy.float32).T
            new_weight_dict[paddle_key] = paddle_weight[paddle_key]
            matched_linear += 1

    paddle_model.set_dict(new_weight_dict)
    paddle_weight = paddle_model.state_dict()
    fluid.dygraph.save_dygraph(paddle_model.state_dict(), save_path)

    print('matched_bn_mean:', matched_bn_mean)
    print('matched_bn_var:', matched_bn_var)
    print('matched_fc:', matched_fc)
    print('matched_base:', matched_base)
    print('matched_linear:', matched_linear)
    print('done!')
