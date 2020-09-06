import paddle.fluid as fluid
# import torch.utils.model_zoo as model_zoo
from paddle.fluid.dygraph.nn import Conv2D, Pool2D, BatchNorm, Linear, Dropout

pretrained_settings = {
    'bninception': {
        'imagenet': {
            # Was ported using python2 (may trigger warning)
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/bn_inception-52deb4733.pth',
            # 'url': 'http://yjxiong.me/others/bn_inception-9f5701afb96c8044.pth',
            'input_space': 'BGR',
            'input_size': [3, 224, 224],
            'input_range': [0, 255],
            'mean': [104, 117, 128],
            'std': [1, 1, 1],
            'num_classes': 101
        }
    }
}


# for myECO model
class BNInception(fluid.dygraph.Layer):

    def __init__(self):
        super(BNInception, self).__init__()
        self.conv1_7x7_s2 = Conv2D(3, 64, filter_size=(7, 7), stride=(2, 2), padding=(3, 3))
        self.conv1_7x7_s2_bn = BatchNorm(64)
        self.conv1_relu_7x7 = fluid.layers.relu()
        self.pool1_3x3_s2 = Pool2D((3, 3), stride=(2, 2), ceil_mode=True)

        self.conv2_3x3_reduce = Conv2D(64, 64, filter_size=(1, 1), stride=(1, 1))
        self.conv2_3x3_reduce_bn = BatchNorm(64)
        self.conv2_relu_3x3_reduce = fluid.layers.relu()
        self.conv2_3x3 = Conv2D(64, 192, filter_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv2_3x3_bn = BatchNorm(192)
        self.conv2_relu_3x3 = fluid.layers.relu()
        self.pool2_3x3_s2 = Pool2D((3, 3), stride=(2, 2), ceil_mode=True)

        self.inception_3a_1x1 = Conv2D(192, 64, filter_size=(1, 1), stride=(1, 1))
        self.inception_3a_1x1_bn = BatchNorm(64)
        self.inception_3a_relu_1x1 = fluid.layers.relu()
        self.inception_3a_3x3_reduce = Conv2D(192, 64, filter_size=(1, 1), stride=(1, 1))
        self.inception_3a_3x3_reduce_bn = BatchNorm(64)
        self.inception_3a_relu_3x3_reduce = fluid.layers.relu()
        self.inception_3a_3x3 = Conv2D(64, 64, filter_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_3a_3x3_bn = BatchNorm(64)
        self.inception_3a_relu_3x3 = fluid.layers.relu()
        self.inception_3a_double_3x3_reduce = Conv2D(192, 64, filter_size=(1, 1), stride=(1, 1))
        self.inception_3a_double_3x3_reduce_bn = BatchNorm(64)
        self.inception_3a_relu_double_3x3_reduce = fluid.layers.relu()
        self.inception_3a_double_3x3_1 = Conv2D(64, 96, filter_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_3a_double_3x3_1_bn = BatchNorm(96)
        self.inception_3a_relu_double_3x3_1 = fluid.layers.relu()
        self.inception_3a_double_3x3_2 = Conv2D(96, 96, filter_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_3a_double_3x3_2_bn = BatchNorm(96)
        self.inception_3a_relu_double_3x3_2 = fluid.layers.relu()
        self.inception_3a_pool = Pool2D(3, pooltype='avg', stride=1, padding=1,
                                        ceil_mode=True)  # , count_include_pad=True
        self.inception_3a_pool_proj = Conv2D(192, 32, filter_size=(1, 1), stride=(1, 1))
        self.inception_3a_pool_proj_bn = BatchNorm(32)
        self.inception_3a_relu_pool_proj = fluid.layers.relu()

        self.inception_3b_1x1 = Conv2D(256, 64, filter_size=(1, 1), stride=(1, 1))
        self.inception_3b_1x1_bn = BatchNorm(64)
        self.inception_3b_relu_1x1 = fluid.layers.relu()
        self.inception_3b_3x3_reduce = Conv2D(256, 64, filter_size=(1, 1), stride=(1, 1))
        self.inception_3b_3x3_reduce_bn = BatchNorm(64)
        self.inception_3b_relu_3x3_reduce = fluid.layers.relu()
        self.inception_3b_3x3 = Conv2D(64, 96, filter_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_3b_3x3_bn = BatchNorm(96)
        self.inception_3b_relu_3x3 = fluid.layers.relu()
        self.inception_3b_double_3x3_reduce = Conv2D(256, 64, filter_size=(1, 1), stride=(1, 1))
        self.inception_3b_double_3x3_reduce_bn = BatchNorm(64)
        self.inception_3b_relu_double_3x3_reduce = fluid.layers.relu()
        self.inception_3b_double_3x3_1 = Conv2D(64, 96, filter_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_3b_double_3x3_1_bn = BatchNorm(96)
        self.inception_3b_relu_double_3x3_1 = fluid.layers.relu()
        self.inception_3b_double_3x3_2 = Conv2D(96, 96, filter_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_3b_double_3x3_2_bn = BatchNorm(96)
        self.inception_3b_relu_double_3x3_2 = fluid.layers.relu()
        self.inception_3b_pool = Pool2D(3, pool_type='avg', stride=1, padding=1,
                                        ceil_mode=True)  # , count_include_pad=True
        self.inception_3b_pool_proj = Conv2D(256, 64, filter_size=(1, 1), stride=(1, 1))
        self.inception_3b_pool_proj_bn = BatchNorm(64)
        self.inception_3b_relu_pool_proj = fluid.layers.relu()

        self.inception_3c_3x3_reduce = Conv2D(320, 128, filter_size=(1, 1), stride=(1, 1))
        self.inception_3c_3x3_reduce_bn = BatchNorm(128)
        self.inception_3c_relu_3x3_reduce = fluid.layers.relu()
        self.inception_3c_3x3 = Conv2D(128, 160, filter_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.inception_3c_3x3_bn = BatchNorm(160)
        self.inception_3c_relu_3x3 = fluid.layers.relu()
        self.inception_3c_double_3x3_reduce = Conv2D(320, 64, filter_size=(1, 1), stride=(1, 1))
        self.inception_3c_double_3x3_reduce_bn = BatchNorm(64)
        self.inception_3c_relu_double_3x3_reduce = fluid.layers.relu()
        self.inception_3c_double_3x3_1 = Conv2D(64, 96, filter_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_3c_double_3x3_1_bn = BatchNorm(96)
        self.inception_3c_relu_double_3x3_1 = fluid.layers.relu()

    def features(self, input):
        conv1_7x7_s2_out = self.conv1_7x7_s2(input)
        conv1_7x7_s2_bn_out = self.conv1_7x7_s2_bn(conv1_7x7_s2_out)
        conv1_relu_7x7_out = self.conv1_relu_7x7(conv1_7x7_s2_bn_out)
        pool1_3x3_s2_out = self.pool1_3x3_s2(conv1_relu_7x7_out)
        conv2_3x3_reduce_out = self.conv2_3x3_reduce(pool1_3x3_s2_out)
        conv2_3x3_reduce_bn_out = self.conv2_3x3_reduce_bn(conv2_3x3_reduce_out)
        conv2_relu_3x3_reduce_out = self.conv2_relu_3x3_reduce(conv2_3x3_reduce_bn_out)
        conv2_3x3_out = self.conv2_3x3(conv2_relu_3x3_reduce_out)
        conv2_3x3_bn_out = self.conv2_3x3_bn(conv2_3x3_out)
        conv2_relu_3x3_out = self.conv2_relu_3x3(conv2_3x3_bn_out)
        pool2_3x3_s2_out = self.pool2_3x3_s2(conv2_relu_3x3_out)

        inception_3a_1x1_out = self.inception_3a_1x1(pool2_3x3_s2_out)
        inception_3a_1x1_bn_out = self.inception_3a_1x1_bn(inception_3a_1x1_out)
        inception_3a_relu_1x1_out = self.inception_3a_relu_1x1(inception_3a_1x1_bn_out)
        inception_3a_3x3_reduce_out = self.inception_3a_3x3_reduce(pool2_3x3_s2_out)
        inception_3a_3x3_reduce_bn_out = self.inception_3a_3x3_reduce_bn(inception_3a_3x3_reduce_out)
        inception_3a_relu_3x3_reduce_out = self.inception_3a_relu_3x3_reduce(inception_3a_3x3_reduce_bn_out)
        inception_3a_3x3_out = self.inception_3a_3x3(inception_3a_relu_3x3_reduce_out)
        inception_3a_3x3_bn_out = self.inception_3a_3x3_bn(inception_3a_3x3_out)
        inception_3a_relu_3x3_out = self.inception_3a_relu_3x3(inception_3a_3x3_bn_out)
        inception_3a_double_3x3_reduce_out = self.inception_3a_double_3x3_reduce(pool2_3x3_s2_out)
        inception_3a_double_3x3_reduce_bn_out = self.inception_3a_double_3x3_reduce_bn(
            inception_3a_double_3x3_reduce_out)
        inception_3a_relu_double_3x3_reduce_out = self.inception_3a_relu_double_3x3_reduce(
            inception_3a_double_3x3_reduce_bn_out)
        inception_3a_double_3x3_1_out = self.inception_3a_double_3x3_1(inception_3a_relu_double_3x3_reduce_out)
        inception_3a_double_3x3_1_bn_out = self.inception_3a_double_3x3_1_bn(inception_3a_double_3x3_1_out)
        inception_3a_relu_double_3x3_1_out = self.inception_3a_relu_double_3x3_1(inception_3a_double_3x3_1_bn_out)
        inception_3a_double_3x3_2_out = self.inception_3a_double_3x3_2(inception_3a_relu_double_3x3_1_out)
        inception_3a_double_3x3_2_bn_out = self.inception_3a_double_3x3_2_bn(inception_3a_double_3x3_2_out)
        inception_3a_relu_double_3x3_2_out = self.inception_3a_relu_double_3x3_2(inception_3a_double_3x3_2_bn_out)
        inception_3a_pool_out = self.inception_3a_pool(pool2_3x3_s2_out)
        inception_3a_pool_proj_out = self.inception_3a_pool_proj(inception_3a_pool_out)
        inception_3a_pool_proj_bn_out = self.inception_3a_pool_proj_bn(inception_3a_pool_proj_out)
        inception_3a_relu_pool_proj_out = self.inception_3a_relu_pool_proj(inception_3a_pool_proj_bn_out)
        inception_3a_output_out = fluid.layers.concat(
            [inception_3a_relu_1x1_out, inception_3a_relu_3x3_out, inception_3a_relu_double_3x3_2_out,
             inception_3a_relu_pool_proj_out], 1)

        inception_3b_1x1_out = self.inception_3b_1x1(inception_3a_output_out)
        inception_3b_1x1_bn_out = self.inception_3b_1x1_bn(inception_3b_1x1_out)
        inception_3b_relu_1x1_out = self.inception_3b_relu_1x1(inception_3b_1x1_bn_out)
        inception_3b_3x3_reduce_out = self.inception_3b_3x3_reduce(inception_3a_output_out)
        inception_3b_3x3_reduce_bn_out = self.inception_3b_3x3_reduce_bn(inception_3b_3x3_reduce_out)
        inception_3b_relu_3x3_reduce_out = self.inception_3b_relu_3x3_reduce(inception_3b_3x3_reduce_bn_out)
        inception_3b_3x3_out = self.inception_3b_3x3(inception_3b_relu_3x3_reduce_out)
        inception_3b_3x3_bn_out = self.inception_3b_3x3_bn(inception_3b_3x3_out)
        inception_3b_relu_3x3_out = self.inception_3b_relu_3x3(inception_3b_3x3_bn_out)
        inception_3b_double_3x3_reduce_out = self.inception_3b_double_3x3_reduce(inception_3a_output_out)
        inception_3b_double_3x3_reduce_bn_out = self.inception_3b_double_3x3_reduce_bn(
            inception_3b_double_3x3_reduce_out)
        inception_3b_relu_double_3x3_reduce_out = self.inception_3b_relu_double_3x3_reduce(
            inception_3b_double_3x3_reduce_bn_out)
        inception_3b_double_3x3_1_out = self.inception_3b_double_3x3_1(inception_3b_relu_double_3x3_reduce_out)
        inception_3b_double_3x3_1_bn_out = self.inception_3b_double_3x3_1_bn(inception_3b_double_3x3_1_out)
        inception_3b_relu_double_3x3_1_out = self.inception_3b_relu_double_3x3_1(inception_3b_double_3x3_1_bn_out)
        inception_3b_double_3x3_2_out = self.inception_3b_double_3x3_2(inception_3b_relu_double_3x3_1_out)
        inception_3b_double_3x3_2_bn_out = self.inception_3b_double_3x3_2_bn(inception_3b_double_3x3_2_out)
        inception_3b_relu_double_3x3_2_out = self.inception_3b_relu_double_3x3_2(inception_3b_double_3x3_2_bn_out)
        inception_3b_pool_out = self.inception_3b_pool(inception_3a_output_out)
        inception_3b_pool_proj_out = self.inception_3b_pool_proj(inception_3b_pool_out)
        inception_3b_pool_proj_bn_out = self.inception_3b_pool_proj_bn(inception_3b_pool_proj_out)
        inception_3b_relu_pool_proj_out = self.inception_3b_relu_pool_proj(inception_3b_pool_proj_bn_out)
        inception_3b_output_out = fluid.layers.concat(
            [inception_3b_relu_1x1_out, inception_3b_relu_3x3_out, inception_3b_relu_double_3x3_2_out,
             inception_3b_relu_pool_proj_out], 1)  # (1, 320, 28, 28)

        inception_3c_double_3x3_reduce_out = self.inception_3c_double_3x3_reduce(
            inception_3b_output_out)  # (1, 64, 28, 28)
        inception_3c_double_3x3_reduce_bn_out = self.inception_3c_double_3x3_reduce_bn(
            inception_3c_double_3x3_reduce_out)  # (1, 64, 28, 28)
        inception_3c_relu_double_3x3_reduce_out = self.inception_3c_relu_double_3x3_reduce(
            inception_3c_double_3x3_reduce_bn_out)  # (1, 64, 28, 28)
        inception_3c_double_3x3_1_out = self.inception_3c_double_3x3_1(
            inception_3c_relu_double_3x3_reduce_out)  # (1, 96, 28, 28)
        inception_3c_double_3x3_1_bn_out = self.inception_3c_double_3x3_1_bn(
            inception_3c_double_3x3_1_out)  # (1, 96, 28, 28)
        inception_3c_relu_double_3x3_1_out = self.inception_3c_relu_double_3x3_1(
            inception_3c_double_3x3_1_bn_out)  # (1, 96, 28, 28) ############## cut here
        return inception_3c_relu_double_3x3_1_out

    def forward(self, input):
        x = self.features(input)

        return x


class BNInception_pre(fluid.dygraph.Layer):

    def __init__(self, num_classes=1000):
        super(BNInception_pre, self).__init__()
        self.conv1_7x7_s2 = Conv2D(3, 64, filter_size=(7, 7), stride=(2, 2), padding=(3, 3))
        self.conv1_7x7_s2_bn = BatchNorm(64)
        # self.conv1_relu_7x7 = fluid.layers.relu()
        self.pool1_3x3_s2 = Pool2D((3, 3), pool_stride=(2, 2), ceil_mode=True)
        self.conv2_3x3_reduce = Conv2D(64, 64, filter_size=(1, 1), stride=(1, 1))
        self.conv2_3x3_reduce_bn = BatchNorm(64)
        # self.conv2_relu_3x3_reduce = fluid.layers.relu()
        self.conv2_3x3 = Conv2D(64, 192, filter_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv2_3x3_bn = BatchNorm(192)
        # self.conv2_relu_3x3 = fluid.layers.relu()
        self.pool2_3x3_s2 = Pool2D((3, 3), pool_stride=(2, 2), ceil_mode=True)
        self.inception_3a_1x1 = Conv2D(192, 64, filter_size=(1, 1), stride=(1, 1))
        self.inception_3a_1x1_bn = BatchNorm(64)
        # self.inception_3a_relu_1x1 = fluid.layers.relu()
        self.inception_3a_3x3_reduce = Conv2D(192, 64, filter_size=(1, 1), stride=(1, 1))
        self.inception_3a_3x3_reduce_bn = BatchNorm(64)
        # self.inception_3a_relu_3x3_reduce = fluid.layers.relu()
        self.inception_3a_3x3 = Conv2D(64, 64, filter_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_3a_3x3_bn = BatchNorm(64)
        # self.inception_3a_relu_3x3 = fluid.layers.relu()
        self.inception_3a_double_3x3_reduce = Conv2D(192, 64, filter_size=(1, 1), stride=(1, 1))
        self.inception_3a_double_3x3_reduce_bn = BatchNorm(64)
        # self.inception_3a_relu_double_3x3_reduce = fluid.layers.relu()
        self.inception_3a_double_3x3_1 = Conv2D(64, 96, filter_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_3a_double_3x3_1_bn = BatchNorm(96)
        # self.inception_3a_relu_double_3x3_1 = fluid.layers.relu()
        self.inception_3a_double_3x3_2 = Conv2D(96, 96, filter_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_3a_double_3x3_2_bn = BatchNorm(96)
        # self.inception_3a_relu_double_3x3_2 = fluid.layers.relu()
        self.inception_3a_pool = Pool2D(3, pool_type='avg', pool_stride=1, pool_padding=1,
                                        ceil_mode=True)  # , count_include_pad=True
        self.inception_3a_pool_proj = Conv2D(192, 32, filter_size=(1, 1), stride=(1, 1))
        self.inception_3a_pool_proj_bn = BatchNorm(32)
        # self.inception_3a_relu_pool_proj = fluid.layers.relu()
        self.inception_3b_1x1 = Conv2D(256, 64, filter_size=(1, 1), stride=(1, 1))
        self.inception_3b_1x1_bn = BatchNorm(64)
        # self.inception_3b_relu_1x1 = fluid.layers.relu()
        self.inception_3b_3x3_reduce = Conv2D(256, 64, filter_size=(1, 1), stride=(1, 1))
        self.inception_3b_3x3_reduce_bn = BatchNorm(64)
        # self.inception_3b_relu_3x3_reduce = fluid.layers.relu()
        self.inception_3b_3x3 = Conv2D(64, 96, filter_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_3b_3x3_bn = BatchNorm(96)
        # self.inception_3b_relu_3x3 = fluid.layers.relu()
        self.inception_3b_double_3x3_reduce = Conv2D(256, 64, filter_size=(1, 1), stride=(1, 1))
        self.inception_3b_double_3x3_reduce_bn = BatchNorm(64)
        # self.inception_3b_relu_double_3x3_reduce = fluid.layers.relu()
        self.inception_3b_double_3x3_1 = Conv2D(64, 96, filter_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_3b_double_3x3_1_bn = BatchNorm(96)
        # self.inception_3b_relu_double_3x3_1 = fluid.layers.relu()
        self.inception_3b_double_3x3_2 = Conv2D(96, 96, filter_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_3b_double_3x3_2_bn = BatchNorm(96)
        # self.inception_3b_relu_double_3x3_2 = fluid.layers.relu()
        self.inception_3b_pool = Pool2D(3, pool_type='avg', pool_stride=1, pool_padding=1,
                                        ceil_mode=True)  # , count_include_pad=True
        self.inception_3b_pool_proj = Conv2D(256, 64, filter_size=(1, 1), stride=(1, 1))
        self.inception_3b_pool_proj_bn = BatchNorm(64)
        # self.inception_3b_relu_pool_proj = fluid.layers.relu()
        self.inception_3c_3x3_reduce = Conv2D(320, 128, filter_size=(1, 1), stride=(1, 1))
        self.inception_3c_3x3_reduce_bn = BatchNorm(128)
        # self.inception_3c_relu_3x3_reduce = fluid.layers.relu()
        self.inception_3c_3x3 = Conv2D(128, 160, filter_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.inception_3c_3x3_bn = BatchNorm(160)
        # self.inception_3c_relu_3x3 = fluid.layers.relu()
        self.inception_3c_double_3x3_reduce = Conv2D(320, 64, filter_size=(1, 1), stride=(1, 1))
        self.inception_3c_double_3x3_reduce_bn = BatchNorm(64)
        # self.inception_3c_relu_double_3x3_reduce = fluid.layers.relu()
        self.inception_3c_double_3x3_1 = Conv2D(64, 96, filter_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_3c_double_3x3_1_bn = BatchNorm(96)
        # self.inception_3c_relu_double_3x3_1 = fluid.layers.relu()
        self.inception_3c_double_3x3_2 = Conv2D(96, 96, filter_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.inception_3c_double_3x3_2_bn = BatchNorm(96)
        # self.inception_3c_relu_double_3x3_2 = fluid.layers.relu()
        self.inception_3c_pool = Pool2D((3, 3), pool_stride=(2, 2), ceil_mode=True)
        self.inception_4a_1x1 = Conv2D(576, 224, filter_size=(1, 1), stride=(1, 1))
        self.inception_4a_1x1_bn = BatchNorm(224)
        # self.inception_4a_relu_1x1 = fluid.layers.relu()
        self.inception_4a_3x3_reduce = Conv2D(576, 64, filter_size=(1, 1), stride=(1, 1))
        self.inception_4a_3x3_reduce_bn = BatchNorm(64)
        # self.inception_4a_relu_3x3_reduce = fluid.layers.relu()
        self.inception_4a_3x3 = Conv2D(64, 96, filter_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_4a_3x3_bn = BatchNorm(96)
        # self.inception_4a_relu_3x3 = fluid.layers.relu()
        self.inception_4a_double_3x3_reduce = Conv2D(576, 96, filter_size=(1, 1), stride=(1, 1))
        self.inception_4a_double_3x3_reduce_bn = BatchNorm(96)
        # self.inception_4a_relu_double_3x3_reduce = fluid.layers.relu()
        self.inception_4a_double_3x3_1 = Conv2D(96, 128, filter_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_4a_double_3x3_1_bn = BatchNorm(128)
        # self.inception_4a_relu_double_3x3_1 = fluid.layers.relu()
        self.inception_4a_double_3x3_2 = Conv2D(128, 128, filter_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_4a_double_3x3_2_bn = BatchNorm(128)
        # self.inception_4a_relu_double_3x3_2 = fluid.layers.relu()
        self.inception_4a_pool = Pool2D(3, pool_type='avg', pool_stride=1, pool_padding=1,
                                        ceil_mode=True)  # , count_include_pad=True
        self.inception_4a_pool_proj = Conv2D(576, 128, filter_size=(1, 1), stride=(1, 1))
        self.inception_4a_pool_proj_bn = BatchNorm(128)
        # self.inception_4a_relu_pool_proj = fluid.layers.relu()
        self.inception_4b_1x1 = Conv2D(576, 192, filter_size=(1, 1), stride=(1, 1))
        self.inception_4b_1x1_bn = BatchNorm(192)
        # self.inception_4b_relu_1x1 = fluid.layers.relu()
        self.inception_4b_3x3_reduce = Conv2D(576, 96, filter_size=(1, 1), stride=(1, 1))
        self.inception_4b_3x3_reduce_bn = BatchNorm(96)
        # self.inception_4b_relu_3x3_reduce = fluid.layers.relu()
        self.inception_4b_3x3 = Conv2D(96, 128, filter_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_4b_3x3_bn = BatchNorm(128)
        # self.inception_4b_relu_3x3 = fluid.layers.relu()
        self.inception_4b_double_3x3_reduce = Conv2D(576, 96, filter_size=(1, 1), stride=(1, 1))
        self.inception_4b_double_3x3_reduce_bn = BatchNorm(96)
        # self.inception_4b_relu_double_3x3_reduce = fluid.layers.relu()
        self.inception_4b_double_3x3_1 = Conv2D(96, 128, filter_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_4b_double_3x3_1_bn = BatchNorm(128)
        # self.inception_4b_relu_double_3x3_1 = fluid.layers.relu()
        self.inception_4b_double_3x3_2 = Conv2D(128, 128, filter_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_4b_double_3x3_2_bn = BatchNorm(128)
        # self.inception_4b_relu_double_3x3_2 = fluid.layers.relu()
        self.inception_4b_pool = Pool2D(3, pool_type='avg', pool_stride=1, pool_padding=1,
                                        ceil_mode=True)  # , count_include_pad=Tru
        self.inception_4b_pool_proj = Conv2D(576, 128, filter_size=(1, 1), stride=(1, 1))
        self.inception_4b_pool_proj_bn = BatchNorm(128)
        # self.inception_4b_relu_pool_proj = fluid.layers.relu()
        self.inception_4c_1x1 = Conv2D(576, 160, filter_size=(1, 1), stride=(1, 1))
        self.inception_4c_1x1_bn = BatchNorm(160)
        # self.inception_4c_relu_1x1 = fluid.layers.relu()
        self.inception_4c_3x3_reduce = Conv2D(576, 128, filter_size=(1, 1), stride=(1, 1))
        self.inception_4c_3x3_reduce_bn = BatchNorm(128)
        # self.inception_4c_relu_3x3_reduce = fluid.layers.relu()
        self.inception_4c_3x3 = Conv2D(128, 160, filter_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_4c_3x3_bn = BatchNorm(160)
        # self.inception_4c_relu_3x3 = fluid.layers.relu()
        self.inception_4c_double_3x3_reduce = Conv2D(576, 128, filter_size=(1, 1), stride=(1, 1))
        self.inception_4c_double_3x3_reduce_bn = BatchNorm(128)
        # self.inception_4c_relu_double_3x3_reduce = fluid.layers.relu()
        self.inception_4c_double_3x3_1 = Conv2D(128, 160, filter_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_4c_double_3x3_1_bn = BatchNorm(160)
        # self.inception_4c_relu_double_3x3_1 = fluid.layers.relu()
        self.inception_4c_double_3x3_2 = Conv2D(160, 160, filter_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_4c_double_3x3_2_bn = BatchNorm(160)
        # self.inception_4c_relu_double_3x3_2 = fluid.layers.relu()
        self.inception_4c_pool = Pool2D(3, pool_type='avg', pool_stride=1, pool_padding=1,
                                        ceil_mode=True)  # , count_include_pad=Tru
        self.inception_4c_pool_proj = Conv2D(576, 128, filter_size=(1, 1), stride=(1, 1))
        self.inception_4c_pool_proj_bn = BatchNorm(128)
        # self.inception_4c_relu_pool_proj = fluid.layers.relu()
        self.inception_4d_1x1 = Conv2D(608, 96, filter_size=(1, 1), stride=(1, 1))
        self.inception_4d_1x1_bn = BatchNorm(96)
        # self.inception_4d_relu_1x1 = fluid.layers.relu()
        self.inception_4d_3x3_reduce = Conv2D(608, 128, filter_size=(1, 1), stride=(1, 1))
        self.inception_4d_3x3_reduce_bn = BatchNorm(128)
        # self.inception_4d_relu_3x3_reduce = fluid.layers.relu()
        self.inception_4d_3x3 = Conv2D(128, 192, filter_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_4d_3x3_bn = BatchNorm(192)
        # self.inception_4d_relu_3x3 = fluid.layers.relu()
        self.inception_4d_double_3x3_reduce = Conv2D(608, 160, filter_size=(1, 1), stride=(1, 1))
        self.inception_4d_double_3x3_reduce_bn = BatchNorm(160)
        # self.inception_4d_relu_double_3x3_reduce = fluid.layers.relu()
        self.inception_4d_double_3x3_1 = Conv2D(160, 192, filter_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_4d_double_3x3_1_bn = BatchNorm(192)
        # self.inception_4d_relu_double_3x3_1 = fluid.layers.relu()
        self.inception_4d_double_3x3_2 = Conv2D(192, 192, filter_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_4d_double_3x3_2_bn = BatchNorm(192)
        # self.inception_4d_relu_double_3x3_2 = fluid.layers.relu()
        self.inception_4d_pool = Pool2D(3, pool_type='avg', pool_stride=1, pool_padding=1,
                                        ceil_mode=True)  # , count_include_pad=Tru
        self.inception_4d_pool_proj = Conv2D(608, 128, filter_size=(1, 1), stride=(1, 1))
        self.inception_4d_pool_proj_bn = BatchNorm(128)
        # self.inception_4d_relu_pool_proj = fluid.layers.relu()
        self.inception_4e_3x3_reduce = Conv2D(608, 128, filter_size=(1, 1), stride=(1, 1))
        self.inception_4e_3x3_reduce_bn = BatchNorm(128)
        # self.inception_4e_relu_3x3_reduce = fluid.layers.relu()
        self.inception_4e_3x3 = Conv2D(128, 192, filter_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.inception_4e_3x3_bn = BatchNorm(192)
        # self.inception_4e_relu_3x3 = fluid.layers.relu()
        self.inception_4e_double_3x3_reduce = Conv2D(608, 192, filter_size=(1, 1), stride=(1, 1))
        self.inception_4e_double_3x3_reduce_bn = BatchNorm(192)
        # self.inception_4e_relu_double_3x3_reduce = fluid.layers.relu()
        self.inception_4e_double_3x3_1 = Conv2D(192, 256, filter_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_4e_double_3x3_1_bn = BatchNorm(256)
        # self.inception_4e_relu_double_3x3_1 = fluid.layers.relu()
        self.inception_4e_double_3x3_2 = Conv2D(256, 256, filter_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.inception_4e_double_3x3_2_bn = BatchNorm(256)
        # self.inception_4e_relu_double_3x3_2 = fluid.layers.relu()
        self.inception_4e_pool = Pool2D((3, 3), pool_stride=(2, 2), ceil_mode=True)
        self.inception_5a_1x1 = Conv2D(1056, 352, filter_size=(1, 1), stride=(1, 1))
        self.inception_5a_1x1_bn = BatchNorm(352)
        # self.inception_5a_relu_1x1 = fluid.layers.relu()
        self.inception_5a_3x3_reduce = Conv2D(1056, 192, filter_size=(1, 1), stride=(1, 1))
        self.inception_5a_3x3_reduce_bn = BatchNorm(192)
        # self.inception_5a_relu_3x3_reduce = fluid.layers.relu()
        self.inception_5a_3x3 = Conv2D(192, 320, filter_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_5a_3x3_bn = BatchNorm(320)
        # self.inception_5a_relu_3x3 = fluid.layers.relu()
        self.inception_5a_double_3x3_reduce = Conv2D(1056, 160, filter_size=(1, 1), stride=(1, 1))
        self.inception_5a_double_3x3_reduce_bn = BatchNorm(160)
        # self.inception_5a_relu_double_3x3_reduce = fluid.layers.relu()
        self.inception_5a_double_3x3_1 = Conv2D(160, 224, filter_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_5a_double_3x3_1_bn = BatchNorm(224)
        # self.inception_5a_relu_double_3x3_1 = fluid.layers.relu()
        self.inception_5a_double_3x3_2 = Conv2D(224, 224, filter_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_5a_double_3x3_2_bn = BatchNorm(224)
        # self.inception_5a_relu_double_3x3_2 = fluid.layers.relu()
        self.inception_5a_pool = Pool2D(3, pool_type='avg', pool_stride=1, pool_padding=1,
                                        ceil_mode=True)  # , count_include_pad=Tru
        self.inception_5a_pool_proj = Conv2D(1056, 128, filter_size=(1, 1), stride=(1, 1))
        self.inception_5a_pool_proj_bn = BatchNorm(128)
        # self.inception_5a_relu_pool_proj = fluid.layers.relu()
        self.inception_5b_1x1 = Conv2D(1024, 352, filter_size=(1, 1), stride=(1, 1))
        self.inception_5b_1x1_bn = BatchNorm(352)
        # self.inception_5b_relu_1x1 = fluid.layers.relu()
        self.inception_5b_3x3_reduce = Conv2D(1024, 192, filter_size=(1, 1), stride=(1, 1))
        self.inception_5b_3x3_reduce_bn = BatchNorm(192)
        # self.inception_5b_relu_3x3_reduce = fluid.layers.relu()
        self.inception_5b_3x3 = Conv2D(192, 320, filter_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_5b_3x3_bn = BatchNorm(320)
        # self.inception_5b_relu_3x3 = fluid.layers.relu()
        self.inception_5b_double_3x3_reduce = Conv2D(1024, 192, filter_size=(1, 1), stride=(1, 1))
        self.inception_5b_double_3x3_reduce_bn = BatchNorm(192)
        # self.inception_5b_relu_double_3x3_reduce = fluid.layers.relu()
        self.inception_5b_double_3x3_1 = Conv2D(192, 224, filter_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_5b_double_3x3_1_bn = BatchNorm(224)
        # self.inception_5b_relu_double_3x3_1 = fluid.layers.relu()
        self.inception_5b_double_3x3_2 = Conv2D(224, 224, filter_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.inception_5b_double_3x3_2_bn = BatchNorm(224)
        # self.inception_5b_relu_double_3x3_2 = fluid.layers.relu()
        self.inception_5b_pool = Pool2D((3, 3), pool_stride=(1, 1), pool_padding=(1, 1), ceil_mode=True)
        self.inception_5b_pool_proj = Conv2D(1024, 128, filter_size=(1, 1), stride=(1, 1))
        self.inception_5b_pool_proj_bn = BatchNorm(128)
        # self.inception_5b_relu_pool_proj = fluid.layers.relu()
        self.global_pool = Pool2D((7, 7), pool_type='avg', pool_stride=1, pool_padding=0,
                                  ceil_mode=True)  # , count_include_pad=Tru
        # self.last_linear = Linear(1024, num_classes)
        # self.last_linear_seg = Linear(1024 * 32, 1024)
        self.global_pool2D_drop = Dropout(p=0)

    def features(self, input):
        conv1_7x7_s2_out = self.conv1_7x7_s2(input)
        conv1_7x7_s2_bn_out = self.conv1_7x7_s2_bn(conv1_7x7_s2_out)
        conv1_relu_7x7_out = fluid.layers.relu(conv1_7x7_s2_bn_out)
        pool1_3x3_s2_out = self.pool1_3x3_s2(conv1_relu_7x7_out)
        conv2_3x3_reduce_out = self.conv2_3x3_reduce(pool1_3x3_s2_out)
        conv2_3x3_reduce_bn_out = self.conv2_3x3_reduce_bn(conv2_3x3_reduce_out)
        conv2_relu_3x3_reduce_out = fluid.layers.relu(conv2_3x3_reduce_bn_out)
        conv2_3x3_out = self.conv2_3x3(conv2_relu_3x3_reduce_out)
        conv2_3x3_bn_out = self.conv2_3x3_bn(conv2_3x3_out)
        conv2_relu_3x3_out = fluid.layers.relu(conv2_3x3_bn_out)
        pool2_3x3_s2_out = self.pool2_3x3_s2(conv2_relu_3x3_out)
        inception_3a_1x1_out = self.inception_3a_1x1(pool2_3x3_s2_out)
        inception_3a_1x1_bn_out = self.inception_3a_1x1_bn(inception_3a_1x1_out)
        inception_3a_relu_1x1_out = fluid.layers.relu(inception_3a_1x1_bn_out)
        inception_3a_3x3_reduce_out = self.inception_3a_3x3_reduce(pool2_3x3_s2_out)
        inception_3a_3x3_reduce_bn_out = self.inception_3a_3x3_reduce_bn(inception_3a_3x3_reduce_out)
        inception_3a_relu_3x3_reduce_out = fluid.layers.relu(inception_3a_3x3_reduce_bn_out)
        inception_3a_3x3_out = self.inception_3a_3x3(inception_3a_relu_3x3_reduce_out)
        inception_3a_3x3_bn_out = self.inception_3a_3x3_bn(inception_3a_3x3_out)
        inception_3a_relu_3x3_out = fluid.layers.relu(inception_3a_3x3_bn_out)
        inception_3a_double_3x3_reduce_out = self.inception_3a_double_3x3_reduce(pool2_3x3_s2_out)
        inception_3a_double_3x3_reduce_bn_out = self.inception_3a_double_3x3_reduce_bn(
            inception_3a_double_3x3_reduce_out)
        inception_3a_relu_double_3x3_reduce_out = fluid.layers.relu(
            inception_3a_double_3x3_reduce_bn_out)
        inception_3a_double_3x3_1_out = self.inception_3a_double_3x3_1(inception_3a_relu_double_3x3_reduce_out)
        inception_3a_double_3x3_1_bn_out = self.inception_3a_double_3x3_1_bn(inception_3a_double_3x3_1_out)
        inception_3a_relu_double_3x3_1_out = fluid.layers.relu(inception_3a_double_3x3_1_bn_out)
        inception_3a_double_3x3_2_out = self.inception_3a_double_3x3_2(inception_3a_relu_double_3x3_1_out)
        inception_3a_double_3x3_2_bn_out = self.inception_3a_double_3x3_2_bn(inception_3a_double_3x3_2_out)
        inception_3a_relu_double_3x3_2_out = fluid.layers.relu(inception_3a_double_3x3_2_bn_out)
        inception_3a_pool_out = self.inception_3a_pool(pool2_3x3_s2_out)
        inception_3a_pool_proj_out = self.inception_3a_pool_proj(inception_3a_pool_out)
        inception_3a_pool_proj_bn_out = self.inception_3a_pool_proj_bn(inception_3a_pool_proj_out)
        inception_3a_relu_pool_proj_out = fluid.layers.relu(inception_3a_pool_proj_bn_out)
        inception_3a_output_out = fluid.layers.concat(
            [inception_3a_relu_1x1_out, inception_3a_relu_3x3_out, inception_3a_relu_double_3x3_2_out,
             inception_3a_relu_pool_proj_out], 1)

        inception_3b_1x1_out = self.inception_3b_1x1(inception_3a_output_out)
        inception_3b_1x1_bn_out = self.inception_3b_1x1_bn(inception_3b_1x1_out)
        inception_3b_relu_1x1_out = fluid.layers.relu(inception_3b_1x1_bn_out)
        inception_3b_3x3_reduce_out = self.inception_3b_3x3_reduce(inception_3a_output_out)
        inception_3b_3x3_reduce_bn_out = self.inception_3b_3x3_reduce_bn(inception_3b_3x3_reduce_out)
        inception_3b_relu_3x3_reduce_out = fluid.layers.relu(inception_3b_3x3_reduce_bn_out)
        inception_3b_3x3_out = self.inception_3b_3x3(inception_3b_relu_3x3_reduce_out)
        inception_3b_3x3_bn_out = self.inception_3b_3x3_bn(inception_3b_3x3_out)
        inception_3b_relu_3x3_out = fluid.layers.relu(inception_3b_3x3_bn_out)
        inception_3b_double_3x3_reduce_out = self.inception_3b_double_3x3_reduce(inception_3a_output_out)
        inception_3b_double_3x3_reduce_bn_out = self.inception_3b_double_3x3_reduce_bn(
            inception_3b_double_3x3_reduce_out)
        inception_3b_relu_double_3x3_reduce_out = fluid.layers.relu(
            inception_3b_double_3x3_reduce_bn_out)
        inception_3b_double_3x3_1_out = self.inception_3b_double_3x3_1(inception_3b_relu_double_3x3_reduce_out)
        inception_3b_double_3x3_1_bn_out = self.inception_3b_double_3x3_1_bn(inception_3b_double_3x3_1_out)
        inception_3b_relu_double_3x3_1_out = fluid.layers.relu(inception_3b_double_3x3_1_bn_out)
        inception_3b_double_3x3_2_out = self.inception_3b_double_3x3_2(inception_3b_relu_double_3x3_1_out)
        inception_3b_double_3x3_2_bn_out = self.inception_3b_double_3x3_2_bn(inception_3b_double_3x3_2_out)
        inception_3b_relu_double_3x3_2_out = fluid.layers.relu(inception_3b_double_3x3_2_bn_out)
        inception_3b_pool_out = self.inception_3b_pool(inception_3a_output_out)
        inception_3b_pool_proj_out = self.inception_3b_pool_proj(inception_3b_pool_out)
        inception_3b_pool_proj_bn_out = self.inception_3b_pool_proj_bn(inception_3b_pool_proj_out)
        inception_3b_relu_pool_proj_out = fluid.layers.relu(inception_3b_pool_proj_bn_out)
        inception_3b_output_out = fluid.layers.concat(
            [inception_3b_relu_1x1_out, inception_3b_relu_3x3_out, inception_3b_relu_double_3x3_2_out,
             inception_3b_relu_pool_proj_out], 1)

        # 1*1 3*3
        inception_3c_3x3_reduce_out = self.inception_3c_3x3_reduce(inception_3b_output_out)
        inception_3c_3x3_reduce_bn_out = self.inception_3c_3x3_reduce_bn(inception_3c_3x3_reduce_out)
        inception_3c_relu_3x3_reduce_out = fluid.layers.relu(inception_3c_3x3_reduce_bn_out)
        inception_3c_3x3_out = self.inception_3c_3x3(inception_3c_relu_3x3_reduce_out)
        inception_3c_3x3_bn_out = self.inception_3c_3x3_bn(inception_3c_3x3_out)
        inception_3c_relu_3x3_out = fluid.layers.relu(inception_3c_3x3_bn_out)
        # 1*1 3*3 3*3
        inception_3c_double_3x3_reduce_out = self.inception_3c_double_3x3_reduce(inception_3b_output_out)
        inception_3c_double_3x3_reduce_bn_out = self.inception_3c_double_3x3_reduce_bn(
            inception_3c_double_3x3_reduce_out)
        inception_3c_relu_double_3x3_reduce_out = fluid.layers.relu(
            inception_3c_double_3x3_reduce_bn_out)
        inception_3c_double_3x3_1_out = self.inception_3c_double_3x3_1(inception_3c_relu_double_3x3_reduce_out)
        inception_3c_double_3x3_1_bn_out = self.inception_3c_double_3x3_1_bn(inception_3c_double_3x3_1_out)
        inception_3c_relu_double_3x3_1_out = fluid.layers.relu(inception_3c_double_3x3_1_bn_out)
        inception_3c_double_3x3_2_out = self.inception_3c_double_3x3_2(inception_3c_relu_double_3x3_1_out)
        inception_3c_double_3x3_2_bn_out = self.inception_3c_double_3x3_2_bn(inception_3c_double_3x3_2_out)
        inception_3c_relu_double_3x3_2_out = fluid.layers.relu(inception_3c_double_3x3_2_bn_out)
        # pool 1*1
        inception_3c_pool_out = self.inception_3c_pool(inception_3b_output_out)
        inception_3c_output_out = fluid.layers.concat(
            [inception_3c_relu_3x3_out, inception_3c_relu_double_3x3_2_out,
             inception_3c_pool_out], 1)  # (1, 320, 28, 28)
        # print('inception_3c_output_out:',inception_3c_output_out.shape)
        return inception_3c_relu_double_3x3_1_out, inception_3c_output_out

    def final_out(self, inception_3c_output_out, bs, ns):
        # 4a
        # 1*1
        inception_4a_1x1_out = self.inception_4a_1x1(inception_3c_output_out)
        inception_4a_1x1_bn_out = self.inception_4a_1x1_bn(inception_4a_1x1_out)
        inception_4a_relu_1x1_out = fluid.layers.relu(inception_4a_1x1_bn_out)
        # 1*1+3*3
        inception_4a_3x3_reduce_out = self.inception_4a_3x3_reduce(inception_3c_output_out)
        inception_4a_3x3_reduce_bn_out = self.inception_4a_3x3_reduce_bn(inception_4a_3x3_reduce_out)
        inception_4a_relu_3x3_reduce_out = fluid.layers.relu(inception_4a_3x3_reduce_bn_out)

        inception_4a_3x3_out = self.inception_4a_3x3(inception_4a_relu_3x3_reduce_out)
        inception_4a_3x3_bn_out = self.inception_4a_3x3_bn(inception_4a_3x3_out)
        inception_4a_relu_3x3_out = fluid.layers.relu(inception_4a_3x3_bn_out)
        # 1*1+3*3+3*3
        inception_4a_double_3x3_reduce_out = self.inception_4a_double_3x3_reduce(inception_3c_output_out)
        inception_4a_double_3x3_reduce_bn_out = self.inception_4a_double_3x3_reduce_bn(
            inception_4a_double_3x3_reduce_out)
        inception_4a_relu_double_3x3_reduce_out = fluid.layers.relu(
            inception_4a_double_3x3_reduce_bn_out)
        inception_4a_double_3x3_1_out = self.inception_4a_double_3x3_1(inception_4a_relu_double_3x3_reduce_out)
        inception_4a_double_3x3_1_bn_out = self.inception_4a_double_3x3_1_bn(inception_4a_double_3x3_1_out)
        inception_4a_relu_double_3x3_1_out = fluid.layers.relu(inception_4a_double_3x3_1_bn_out)
        inception_4a_double_3x3_2_out = self.inception_4a_double_3x3_2(inception_4a_relu_double_3x3_1_out)
        inception_4a_double_3x3_2_bn_out = self.inception_4a_double_3x3_2_bn(inception_4a_double_3x3_2_out)
        inception_4a_relu_double_3x3_2_out = fluid.layers.relu(inception_4a_double_3x3_2_bn_out)
        # pool
        inception_4a_pool_out = self.inception_4a_pool(inception_3c_output_out)
        inception_4a_pool_proj_out = self.inception_4a_pool_proj(inception_4a_pool_out)
        inception_4a_pool_proj_bn_out = self.inception_4a_pool_proj_bn(inception_4a_pool_proj_out)
        inception_4a_relu_pool_proj_out = fluid.layers.relu(inception_4a_pool_proj_bn_out)
        # concat
        inception_4a_output_out = fluid.layers.concat(
            [inception_4a_relu_1x1_out, inception_4a_relu_3x3_out, inception_4a_relu_double_3x3_2_out,
             inception_4a_relu_pool_proj_out], 1)

        # 4b
        # 1*1
        inception_4b_1x1_out = self.inception_4b_1x1(inception_4a_output_out)
        inception_4b_1x1_bn_out = self.inception_4b_1x1_bn(inception_4b_1x1_out)
        inception_4b_relu_1x1_out = fluid.layers.relu(inception_4b_1x1_bn_out)
        # 1*1+3*3
        inception_4b_3x3_reduce_out = self.inception_4b_3x3_reduce(inception_4a_output_out)
        inception_4b_3x3_reduce_bn_out = self.inception_4b_3x3_reduce_bn(inception_4b_3x3_reduce_out)
        inception_4b_relu_3x3_reduce_out = fluid.layers.relu(inception_4b_3x3_reduce_bn_out)

        inception_4b_3x3_out = self.inception_4b_3x3(inception_4b_relu_3x3_reduce_out)
        inception_4b_3x3_bn_out = self.inception_4b_3x3_bn(inception_4b_3x3_out)
        inception_4b_relu_3x3_out = fluid.layers.relu(inception_4b_3x3_bn_out)
        # 1*1+3*3+3*3
        inception_4b_double_3x3_reduce_out = self.inception_4b_double_3x3_reduce(inception_4a_output_out)
        inception_4b_double_3x3_reduce_bn_out = self.inception_4b_double_3x3_reduce_bn(
            inception_4b_double_3x3_reduce_out)
        inception_4b_relu_double_3x3_reduce_out = fluid.layers.relu(
            inception_4b_double_3x3_reduce_bn_out)
        inception_4b_double_3x3_1_out = self.inception_4b_double_3x3_1(inception_4b_relu_double_3x3_reduce_out)
        inception_4b_double_3x3_1_bn_out = self.inception_4b_double_3x3_1_bn(inception_4b_double_3x3_1_out)
        inception_4b_relu_double_3x3_1_out = fluid.layers.relu(inception_4b_double_3x3_1_bn_out)
        inception_4b_double_3x3_2_out = self.inception_4b_double_3x3_2(inception_4b_relu_double_3x3_1_out)
        inception_4b_double_3x3_2_bn_out = self.inception_4b_double_3x3_2_bn(inception_4b_double_3x3_2_out)
        inception_4b_relu_double_3x3_2_out = fluid.layers.relu(inception_4b_double_3x3_2_bn_out)
        # pool
        inception_4b_pool_out = self.inception_4b_pool(inception_4a_output_out)
        inception_4b_pool_proj_out = self.inception_4b_pool_proj(inception_4b_pool_out)
        inception_4b_pool_proj_bn_out = self.inception_4b_pool_proj_bn(inception_4b_pool_proj_out)
        inception_4b_relu_pool_proj_out = fluid.layers.relu(inception_4b_pool_proj_bn_out)
        # concat
        inception_4b_output_out = fluid.layers.concat(
            [inception_4b_relu_1x1_out, inception_4b_relu_3x3_out, inception_4b_relu_double_3x3_2_out,
             inception_4b_relu_pool_proj_out], 1)

        # 4c
        # 1*1
        inception_4c_1x1_out = self.inception_4c_1x1(inception_4b_output_out)
        inception_4c_1x1_bn_out = self.inception_4c_1x1_bn(inception_4c_1x1_out)
        inception_4c_relu_1x1_out = fluid.layers.relu(inception_4c_1x1_bn_out)
        # 1*1+3*3
        inception_4c_3x3_reduce_out = self.inception_4c_3x3_reduce(inception_4b_output_out)
        inception_4c_3x3_reduce_bn_out = self.inception_4c_3x3_reduce_bn(inception_4c_3x3_reduce_out)
        inception_4c_relu_3x3_reduce_out = fluid.layers.relu(inception_4c_3x3_reduce_bn_out)

        inception_4c_3x3_out = self.inception_4c_3x3(inception_4c_relu_3x3_reduce_out)
        inception_4c_3x3_bn_out = self.inception_4c_3x3_bn(inception_4c_3x3_out)
        inception_4c_relu_3x3_out = fluid.layers.relu(inception_4c_3x3_bn_out)
        # 1*1+3*3+3*3
        inception_4c_double_3x3_reduce_out = self.inception_4c_double_3x3_reduce(inception_4b_output_out)
        inception_4c_double_3x3_reduce_bn_out = self.inception_4c_double_3x3_reduce_bn(
            inception_4c_double_3x3_reduce_out)
        inception_4c_relu_double_3x3_reduce_out = fluid.layers.relu(
            inception_4c_double_3x3_reduce_bn_out)
        inception_4c_double_3x3_1_out = self.inception_4c_double_3x3_1(inception_4c_relu_double_3x3_reduce_out)
        inception_4c_double_3x3_1_bn_out = self.inception_4c_double_3x3_1_bn(inception_4c_double_3x3_1_out)
        inception_4c_relu_double_3x3_1_out = fluid.layers.relu(inception_4c_double_3x3_1_bn_out)
        inception_4c_double_3x3_2_out = self.inception_4c_double_3x3_2(inception_4c_relu_double_3x3_1_out)
        inception_4c_double_3x3_2_bn_out = self.inception_4c_double_3x3_2_bn(inception_4c_double_3x3_2_out)
        inception_4c_relu_double_3x3_2_out = fluid.layers.relu(inception_4c_double_3x3_2_bn_out)
        # pool
        inception_4c_pool_out = self.inception_4c_pool(inception_4b_output_out)
        inception_4c_pool_proj_out = self.inception_4c_pool_proj(inception_4c_pool_out)
        inception_4c_pool_proj_bn_out = self.inception_4c_pool_proj_bn(inception_4c_pool_proj_out)
        inception_4c_relu_pool_proj_out = fluid.layers.relu(inception_4c_pool_proj_bn_out)
        # concat
        inception_4c_output_out = fluid.layers.concat(
            [inception_4c_relu_1x1_out, inception_4c_relu_3x3_out, inception_4c_relu_double_3x3_2_out,
             inception_4c_relu_pool_proj_out], 1)

        # 4d
        # 1*1
        inception_4d_1x1_out = self.inception_4d_1x1(inception_4c_output_out)
        inception_4d_1x1_bn_out = self.inception_4d_1x1_bn(inception_4d_1x1_out)
        inception_4d_relu_1x1_out = fluid.layers.relu(inception_4d_1x1_bn_out)
        # 1*1+3*3
        inception_4d_3x3_reduce_out = self.inception_4d_3x3_reduce(inception_4c_output_out)
        inception_4d_3x3_reduce_bn_out = self.inception_4d_3x3_reduce_bn(inception_4d_3x3_reduce_out)
        inception_4d_relu_3x3_reduce_out = fluid.layers.relu(inception_4d_3x3_reduce_bn_out)

        inception_4d_3x3_out = self.inception_4d_3x3(inception_4d_relu_3x3_reduce_out)
        inception_4d_3x3_bn_out = self.inception_4d_3x3_bn(inception_4d_3x3_out)
        inception_4d_relu_3x3_out = fluid.layers.relu(inception_4d_3x3_bn_out)
        # 1*1+3*3+3*3
        inception_4d_double_3x3_reduce_out = self.inception_4d_double_3x3_reduce(inception_4c_output_out)
        inception_4d_double_3x3_reduce_bn_out = self.inception_4d_double_3x3_reduce_bn(
            inception_4d_double_3x3_reduce_out)
        inception_4d_relu_double_3x3_reduce_out = fluid.layers.relu(
            inception_4d_double_3x3_reduce_bn_out)
        inception_4d_double_3x3_1_out = self.inception_4d_double_3x3_1(inception_4d_relu_double_3x3_reduce_out)
        inception_4d_double_3x3_1_bn_out = self.inception_4d_double_3x3_1_bn(inception_4d_double_3x3_1_out)
        inception_4d_relu_double_3x3_1_out = fluid.layers.relu(inception_4d_double_3x3_1_bn_out)
        inception_4d_double_3x3_2_out = self.inception_4d_double_3x3_2(inception_4d_relu_double_3x3_1_out)
        inception_4d_double_3x3_2_bn_out = self.inception_4d_double_3x3_2_bn(inception_4d_double_3x3_2_out)
        inception_4d_relu_double_3x3_2_out = fluid.layers.relu(inception_4d_double_3x3_2_bn_out)
        # pool
        inception_4d_pool_out = self.inception_4d_pool(inception_4c_output_out)
        inception_4d_pool_proj_out = self.inception_4d_pool_proj(inception_4d_pool_out)
        inception_4d_pool_proj_bn_out = self.inception_4d_pool_proj_bn(inception_4d_pool_proj_out)
        inception_4d_relu_pool_proj_out = fluid.layers.relu(inception_4d_pool_proj_bn_out)
        # concat
        inception_4d_output_out = fluid.layers.concat(
            [inception_4d_relu_1x1_out, inception_4d_relu_3x3_out, inception_4d_relu_double_3x3_2_out,
             inception_4d_relu_pool_proj_out], 1)

        # 4e
        # 1*1+3*3
        inception_4e_3x3_reduce_out = self.inception_4e_3x3_reduce(inception_4d_output_out)
        inception_4e_3x3_reduce_bn_out = self.inception_4e_3x3_reduce_bn(inception_4e_3x3_reduce_out)
        inception_4e_relu_3x3_reduce_out = fluid.layers.relu(inception_4e_3x3_reduce_bn_out)

        inception_4e_3x3_out = self.inception_4e_3x3(inception_4e_relu_3x3_reduce_out)
        inception_4e_3x3_bn_out = self.inception_4e_3x3_bn(inception_4e_3x3_out)
        inception_4e_relu_3x3_out = fluid.layers.relu(inception_4e_3x3_bn_out)
        # 1*1+3*3+3*3
        inception_4e_double_3x3_reduce_out = self.inception_4e_double_3x3_reduce(inception_4d_output_out)
        inception_4e_double_3x3_reduce_bn_out = self.inception_4e_double_3x3_reduce_bn(
            inception_4e_double_3x3_reduce_out)
        inception_4e_relu_double_3x3_reduce_out = fluid.layers.relu(
            inception_4e_double_3x3_reduce_bn_out)
        inception_4e_double_3x3_1_out = self.inception_4e_double_3x3_1(inception_4e_relu_double_3x3_reduce_out)
        inception_4e_double_3x3_1_bn_out = self.inception_4e_double_3x3_1_bn(inception_4e_double_3x3_1_out)
        inception_4e_relu_double_3x3_1_out = fluid.layers.relu(inception_4e_double_3x3_1_bn_out)
        inception_4e_double_3x3_2_out = self.inception_4e_double_3x3_2(inception_4e_relu_double_3x3_1_out)
        inception_4e_double_3x3_2_bn_out = self.inception_4e_double_3x3_2_bn(inception_4e_double_3x3_2_out)
        inception_4e_relu_double_3x3_2_out = fluid.layers.relu(inception_4e_double_3x3_2_bn_out)
        # pool
        inception_4e_pool_out = self.inception_4e_pool(inception_4d_output_out)
        # concat
        inception_4e_output_out = fluid.layers.concat(
            [inception_4e_relu_3x3_out, inception_4e_relu_double_3x3_2_out,
             inception_4e_pool_out], 1)

        # 5a
        # 1*1
        inception_5a_1x1_out = self.inception_5a_1x1(inception_4e_output_out)
        inception_5a_1x1_bn_out = self.inception_5a_1x1_bn(inception_5a_1x1_out)
        inception_5a_relu_1x1_out = fluid.layers.relu(inception_5a_1x1_bn_out)
        # 1*1+3*3
        inception_5a_3x3_reduce_out = self.inception_5a_3x3_reduce(inception_4e_output_out)
        inception_5a_3x3_reduce_bn_out = self.inception_5a_3x3_reduce_bn(inception_5a_3x3_reduce_out)
        inception_5a_relu_3x3_reduce_out = fluid.layers.relu(inception_5a_3x3_reduce_bn_out)

        inception_5a_3x3_out = self.inception_5a_3x3(inception_5a_relu_3x3_reduce_out)
        inception_5a_3x3_bn_out = self.inception_5a_3x3_bn(inception_5a_3x3_out)
        inception_5a_relu_3x3_out = fluid.layers.relu(inception_5a_3x3_bn_out)
        # 1*1+3*3+3*3
        inception_5a_double_3x3_reduce_out = self.inception_5a_double_3x3_reduce(inception_4e_output_out)
        inception_5a_double_3x3_reduce_bn_out = self.inception_5a_double_3x3_reduce_bn(
            inception_5a_double_3x3_reduce_out)
        inception_5a_relu_double_3x3_reduce_out = fluid.layers.relu(
            inception_5a_double_3x3_reduce_bn_out)
        inception_5a_double_3x3_1_out = self.inception_5a_double_3x3_1(inception_5a_relu_double_3x3_reduce_out)
        inception_5a_double_3x3_1_bn_out = self.inception_5a_double_3x3_1_bn(inception_5a_double_3x3_1_out)
        inception_5a_relu_double_3x3_1_out = fluid.layers.relu(inception_5a_double_3x3_1_bn_out)
        inception_5a_double_3x3_2_out = self.inception_5a_double_3x3_2(inception_5a_relu_double_3x3_1_out)
        inception_5a_double_3x3_2_bn_out = self.inception_5a_double_3x3_2_bn(inception_5a_double_3x3_2_out)
        inception_5a_relu_double_3x3_2_out = fluid.layers.relu(inception_5a_double_3x3_2_bn_out)
        # pool
        inception_5a_pool_out = self.inception_5a_pool(inception_4e_output_out)
        inception_5a_pool_proj_out = self.inception_5a_pool_proj(inception_5a_pool_out)
        inception_5a_pool_proj_bn_out = self.inception_5a_pool_proj_bn(inception_5a_pool_proj_out)
        inception_5a_relu_pool_proj_out = fluid.layers.relu(inception_5a_pool_proj_bn_out)
        # concat
        inception_5a_output_out = fluid.layers.concat(
            [inception_5a_relu_1x1_out, inception_5a_relu_3x3_out, inception_5a_relu_double_3x3_2_out,
             inception_5a_relu_pool_proj_out], 1)

        # 5b
        # 1*1
        inception_5b_1x1_out = self.inception_5b_1x1(inception_5a_output_out)
        inception_5b_1x1_bn_out = self.inception_5b_1x1_bn(inception_5b_1x1_out)
        inception_5b_relu_1x1_out = fluid.layers.relu(inception_5b_1x1_bn_out)
        # 1*1+3*3
        inception_5b_3x3_reduce_out = self.inception_5b_3x3_reduce(inception_5a_output_out)
        inception_5b_3x3_reduce_bn_out = self.inception_5b_3x3_reduce_bn(inception_5b_3x3_reduce_out)
        inception_5b_relu_3x3_reduce_out = fluid.layers.relu(inception_5b_3x3_reduce_bn_out)

        inception_5b_3x3_out = self.inception_5b_3x3(inception_5b_relu_3x3_reduce_out)
        inception_5b_3x3_bn_out = self.inception_5b_3x3_bn(inception_5b_3x3_out)
        inception_5b_relu_3x3_out = fluid.layers.relu(inception_5b_3x3_bn_out)
        # 1*1+3*3+3*3
        inception_5b_double_3x3_reduce_out = self.inception_5b_double_3x3_reduce(inception_5a_output_out)
        inception_5b_double_3x3_reduce_bn_out = self.inception_5b_double_3x3_reduce_bn(
            inception_5b_double_3x3_reduce_out)
        inception_5b_relu_double_3x3_reduce_out = fluid.layers.relu(
            inception_5b_double_3x3_reduce_bn_out)
        inception_5b_double_3x3_1_out = self.inception_5b_double_3x3_1(inception_5b_relu_double_3x3_reduce_out)
        inception_5b_double_3x3_1_bn_out = self.inception_5b_double_3x3_1_bn(inception_5b_double_3x3_1_out)
        inception_5b_relu_double_3x3_1_out = fluid.layers.relu(inception_5b_double_3x3_1_bn_out)
        inception_5b_double_3x3_2_out = self.inception_5b_double_3x3_2(inception_5b_relu_double_3x3_1_out)
        inception_5b_double_3x3_2_bn_out = self.inception_5b_double_3x3_2_bn(inception_5b_double_3x3_2_out)
        inception_5b_relu_double_3x3_2_out = fluid.layers.relu(inception_5b_double_3x3_2_bn_out)
        # pool
        inception_5b_pool_out = self.inception_5b_pool(inception_5a_output_out)
        inception_5b_pool_proj_out = self.inception_5b_pool_proj(inception_5b_pool_out)
        inception_5b_pool_proj_bn_out = self.inception_5b_pool_proj_bn(inception_5b_pool_proj_out)
        inception_5b_relu_pool_proj_out = fluid.layers.relu(inception_5b_pool_proj_bn_out)
        # concat
        inception_5b_output_out = fluid.layers.concat(
            [inception_5b_relu_1x1_out, inception_5b_relu_3x3_out, inception_5b_relu_double_3x3_2_out,
             inception_5b_relu_pool_proj_out], 1)
        # print('inception_5b_output_out:', inception_5b_output_out.shape)
        global_pool_out = self.global_pool(inception_5b_output_out)
        global_pool_out = self.global_pool2D_drop(global_pool_out)

        # global_pool_out = fluid.layers.reshape(global_pool_out, [-1, 1024])
        global_pool_out_reshape = fluid.layers.reshape(global_pool_out, [bs, ns, 1024, 1, 1])
        # print('global_pool_out_reshape:', global_pool_out_reshape.shape)
        global_pool_out_reshape = fluid.layers.transpose(global_pool_out_reshape, [0, 2, 1, 3, 4])
        # print('global_pool_out_reshape:', global_pool_out_reshape.shape)
        global_pool2D_reshape_consensus = fluid.layers.pool3d(global_pool_out_reshape, pool_size=(ns, 1, 1),
                                                              pool_type='avg', pool_stride=(1, 1, 1))
        # print('global_pool2D_reshape_consensus:', global_pool2D_reshape_consensus.shape)
        global_pool2D_reshape_consensus = fluid.layers.reshape(global_pool2D_reshape_consensus, [-1, 1024])
        # last_linear_out = self.last_linear_seg(global_pool_out)
        # print('last_linear_out:', last_linear_out.shape)
        # last_linear_out = self.last_linear(global_pool_out)
        return global_pool2D_reshape_consensus

    def forward(self, input, bs, ns):
        inception_3c_relu_double_3x3_1_out, inception_3c_output_out = self.features(input)
        x = self.final_out(inception_3c_output_out, bs, ns)
        # x = self.logits(x)
        return inception_3c_relu_double_3x3_1_out, x





# for myECO model
def bninception():
    """
    BNInception model architecture from <https://arxiv.org/pdf/1502.03167.pdf>`_ paper
    In addition, we modified BNInception model to fit the ECO paper
    """
    model = BNInception()
    return model
