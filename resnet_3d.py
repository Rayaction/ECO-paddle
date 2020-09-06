import paddle.fluid as fluid
from paddle.fluid.dygraph.nn import BatchNorm, Conv3D


def conv3d(in_planes, out_planes, filter_size=3, stride=1, padding=1):
    # data = numpy.random.random((5, 3, 12, 32, 32)).astype('float32')
    # conv3d = fluid.dygraph.nn.Conv3D(
    #       'Conv3D', num_filters=2, filter_size=3, act="relu")
    # ret = conv3d(fluid.dygraph.base.to_variable(data))
    # 3*3 conv with padding
    return Conv3D(num_channels=in_planes,
                  num_filters=out_planes,
                  filter_size=filter_size,
                  stride = stride,
                  padding=padding,)


class res3a(fluid.dygraph.Layer):
    def __init__(self):
        super(res3a, self).__init__()
        self.res3a_2 = conv3d(96, 128, stride=1)
        self.res3a_bn = BatchNorm(128, act='relu')
        self.res3b_1 = conv3d(128, 128, stride=1)
        self.res3b_1_bn = BatchNorm(128, act='relu')
        self.res3b_2 = conv3d(128, 128, stride=1)
        self.res3b_bn = BatchNorm(128, act='relu')

    def forward(self, input):
        residual = self.res3a_2(input)
        x = self.res3a_bn(residual)  # result of res3a_2n
        x = self.res3b_1(x)
        x = self.res3b_1_bn(x)
        # x += residual
        x = fluid.layers.elementwise_add(x=x, y=residual)
        x = self.res3b_bn(x)
        return x  # res3b_relu


class res4(fluid.dygraph.Layer):
    def __init__(self):
        super(res4, self).__init__()

        self.res4a_1 = conv3d(128, 256, stride=2)
        self.res4a_1_bn = BatchNorm(256, act='relu')

        self.res4a_2 = conv3d(256, 256)

        self.res4a_down = conv3d(128, 256 * 1, filter_size=3, stride=2, padding=1)
        # self.res4a_down_bn = BatchNorm(256 * 1)

        self.res4a_bn = BatchNorm(256, act='relu')

        self.res4b_1 = conv3d(256, 256, stride=1)

        self.res4b_1_bn = BatchNorm(256, act='relu')

        self.res4b_2 = conv3d(256, 256, stride=1)

        self.res4b_bn = BatchNorm(256, act='relu')

    def forward(self, input):
        # print('res4a_down:',input.shape)
        residual = self.res4a_down(input)
        # residual = self.res4a_down_bn(residual)
        # print(input.shape)
        x = self.res4a_1(input)  # take res3b_relu directly, res4a_1
        # print(x.shape)
        x = self.res4a_1_bn(x)
        # print(x.shape)
        x = self.res4a_2(x)
        # print(x.shape)
        # x += residual  # res4b
        # print(x.shape,residual.shape)
        x = fluid.layers.elementwise_add(x=x, y=residual)
        residual2 = x

        x = self.res4a_bn(x)

        x = self.res4b_1(x)

        x = self.res4b_1_bn(x)

        x = self.res4b_2(x)

        # x += residual2
        x = fluid.layers.elementwise_add(x=x, y=residual2)

        x = self.res4b_bn(x)

        return x  # res4b_relu


class res5(fluid.dygraph.Layer):
    def __init__(self):
        super(res5, self).__init__()

        self.res5a_1 = conv3d(256, 512, stride=2)
        self.res5a_1_bn = BatchNorm(512, act='relu')

        self.res5a_2 = conv3d(512, 512)

        self.res5a_down = conv3d(256, 512 * 1, filter_size=3, stride=2,padding=1)
        # self.res5a_down_bn = BatchNorm(512 * 1)

        self.res5a_bn = BatchNorm(512, act='relu')

        self.res5b_1 = conv3d(512, 512, stride=1)

        self.res5b_1_bn = BatchNorm(512, act='relu')

        self.res5b_2 = conv3d(512, 512, stride=1)

        self.res5b_bn = BatchNorm(512, act='relu')

    def forward(self, input):
        residual = self.res5a_down(input)
        # residual = self.res5a_down_bn(residual)

        x = self.res5a_1(input)  # take res4b_relu directly, res5a_1
        x = self.res5a_1_bn(x)

        x = self.res5a_2(x)

        # x += residual  # res5a
        x = fluid.layers.elementwise_add(x=x, y=residual)

        residual2 = x

        x = self.res5a_bn(x)

        x = self.res5b_1(x)

        x = self.res5b_1_bn(x)

        x = self.res5b_2(x)

        # x += residual2  # res5b
        x = fluid.layers.elementwise_add(x=x, y=residual2)

        x = self.res5b_bn(x)

        return x  # res5b_relu


class resnet3d(fluid.dygraph.Layer):
    def __init__(self):
        super(resnet3d, self).__init__()

        # res3a
        # input: res3a_2n
        self.res3 = res3a()
        self.res4 = res4()
        self.res5 = res5()

    def forward(self, input):
        # input: (bs, 96, ns, 28 , 28)
        x = self.res3(input)  # (bs, 128, 16, 28, 28)
        x = self.res4(x)  # (bs, 256, 8, 14, 14)
        x = self.res5(x)  # (bs, 512, 4, 7, 7)

        return x


if __name__ == '__main__':
    model = resnet3d()
