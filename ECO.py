import paddle.fluid as fluid
from paddle.fluid.dygraph.nn import Linear, Dropout

from model import resnet_3d
from model import bninception


class ECO(fluid.dygraph.Layer):
    def __init__(self, num_classes, num_segments, dropout=0):
        super(ECO, self).__init__()
        self.num_segments = num_segments
        self.channel = 3
        self.reshape = True
        self.dropout = dropout
        self.input_size = 224
        self.bninception_pretrained = bninception.BNInception_pre(num_classes=num_classes)
        self.resnet3d = resnet_3d.resnet3d()
        self.dropout = Dropout(dropout)
        # self.fc_0 = Linear(512+1024, 400, bias_attr=True, act='leaky_relu')
        self.fc_final = Linear(512+1024, num_classes, bias_attr=True, act='softmax', param_attr=fluid.initializer.Xavier(uniform=False))

    def forward(self, input, label):
        # input: (bs, c*ns, h, w)
        bs, ns, c, h, w = input.shape
        input = fluid.layers.reshape(input, [-1, input.shape[2], input.shape[3], input.shape[4]])# (bs*ns, c, h, w)

        # base model: BNINception pretrained model
        feat, x1 = self.bninception_pretrained(input, bs, ns)
        feat = fluid.layers.reshape(feat, [bs, 96, self.num_segments, 28, 28])  # (bs, 96, ns, 28, 28)

        # 3D resnet
        x2 = self.resnet3d(feat)  # (bs, 512, 4, 7, 7)
        bs, _, fc, fh, hw = x2.shape
        x2 = fluid.layers.pool3d(x2, pool_size=(fc, fh, hw), pool_type='avg', pool_stride=(1, 1, 1))
        x2 = self.dropout(x2)
        # fully connected
        x2 = fluid.layers.reshape(x2, [-1, 512])

        x = fluid.layers.concat([x1, x2], 1)  # 1024+512=1536
        x = self.fc_final(x)

        if label is not None:
            acc = fluid.layers.accuracy(input=x, label=label)
            return x, acc
        else:
            return x

    @property
    def crop_size(self):
        return self.input_size

    @property
    def scale_size(self):
        return self.input_size * 256 // 224
#
if __name__ == '__main__':
    model = ECO(num_classes=101, num_segments=32)