from _collections import OrderedDict

import numpy
import paddle.fluid as fluid
import pickle
import copy
from model import ECO
path = 'checkpoints_models/ECO_FULL/4.pdparams'
path = 'checkpoints_models/ECO_FULL_RGB_seg16.pdparams'

with fluid.dygraph.guard():
    # paddle_model = ECO.ECO(num_classes=101, num_segments=24)
    # paddle_weight = paddle_model.state_dict()
    # state_dict = fluid.dygraph.load_dygraph(path)
    # paddle_model.set_dict(state_dict)
    # print('pretrained loaded')
    model, _ = fluid.dygraph.load_dygraph(path)
    model_1 = copy.deepcopy(model)
    model_2 = copy.deepcopy(model)
    for i,key in enumerate(model):
        print(i,key)
        if i < 400:
            model_1.pop(key)
        else:
            model_2.pop(key)
    f = open('best_model_0.pkl', 'wb')
    pickle.dump(model_1, f)
    f.close()
    f = open('best_model_1.pkl', 'wb')
    pickle.dump(model_2, f)
    f.close()
    print('pretrained loaded')
