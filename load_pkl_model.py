import pickle
import sys
sys.path.append("..")
from model import ECO
import paddle.fluid as fluid

# Load pickle, since pretrained model is too bigger than the threshold(150M), split them into 2 parts and then reload them

f0 = open('seg0.pkl', 'rb')
f1 = open('seg1.pkl', 'rb')
model_out = dict()
model_0 = pickle.load(f0)
model_1 = pickle.load(f1)
for i,key in enumerate(model_0):
    model_out[key]=model_0[key]
for i,key in enumerate(model_1):
    model_out[key]=model_1[key]
with fluid.dygraph.guard():
    paddle_model = ECO.ECO(num_classes=101, num_segments=24)
    paddle_model.load_dict(model_out)
    fluid.dygraph.save_dygraph(paddle_model.state_dict(), 'ECO_FULL_RGB__seg16')
    print('finished')
