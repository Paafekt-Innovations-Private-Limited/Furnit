import os
import numpy as np
import tempfile, zipfile
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    import torchvision
    import torchaudio
except:
    pass

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.convbn2d_0 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=1024, kernel_size=(1,1), out_channels=256, padding=(0,0), padding_mode='zeros', stride=(1,1))
        self.reduce_2 = nn.ReLU()
        self.convbn2d_1 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=256, kernel_size=(3,3), out_channels=256, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.reduce_5 = nn.ReLU()
        self.up1_0 = nn.Upsample(align_corners=False, mode='bilinear', scale_factor=(2.0,2.0), size=None)
        self.convbn2d_2 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=256, kernel_size=(3,3), out_channels=256, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.up1_3 = nn.ReLU()
        self.up2_0 = nn.Upsample(align_corners=False, mode='bilinear', scale_factor=(2.0,2.0), size=None)
        self.convbn2d_3 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=256, kernel_size=(3,3), out_channels=128, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.up2_3 = nn.ReLU()
        self.convbn2d_4 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=128, kernel_size=(3,3), out_channels=64, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.head_2 = nn.ReLU()
        self.head_3 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=64, kernel_size=(1,1), out_channels=14, padding=(0,0), padding_mode='zeros', stride=(1,1))

        archive = zipfile.ZipFile('gaussian_head.pnnx.bin', 'r')
        self.convbn2d_0.bias = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_0.bias', (256), 'float32')
        self.convbn2d_0.weight = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_0.weight', (256,1024,1,1), 'float32')
        self.convbn2d_1.bias = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_1.bias', (256), 'float32')
        self.convbn2d_1.weight = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_1.weight', (256,256,3,3), 'float32')
        self.convbn2d_2.bias = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_2.bias', (256), 'float32')
        self.convbn2d_2.weight = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_2.weight', (256,256,3,3), 'float32')
        self.convbn2d_3.bias = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_3.bias', (128), 'float32')
        self.convbn2d_3.weight = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_3.weight', (128,256,3,3), 'float32')
        self.convbn2d_4.bias = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_4.bias', (64), 'float32')
        self.convbn2d_4.weight = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_4.weight', (64,128,3,3), 'float32')
        self.head_3.bias = self.load_pnnx_bin_as_parameter(archive, 'head.3.bias', (14), 'float32')
        self.head_3.weight = self.load_pnnx_bin_as_parameter(archive, 'head.3.weight', (14,64,1,1), 'float32')
        archive.close()

    def load_pnnx_bin_as_parameter(self, archive, key, shape, dtype, requires_grad=True):
        return nn.Parameter(self.load_pnnx_bin_as_tensor(archive, key, shape, dtype), requires_grad)

    def load_pnnx_bin_as_tensor(self, archive, key, shape, dtype):
        fd, tmppath = tempfile.mkstemp()
        with os.fdopen(fd, 'wb') as tmpf, archive.open(key) as keyfile:
            tmpf.write(keyfile.read())
        m = np.memmap(tmppath, dtype=dtype, mode='r', shape=shape).copy()
        os.remove(tmppath)
        return torch.from_numpy(m)

    def forward(self, v_0):
        v_1 = self.convbn2d_0(v_0)
        v_2 = self.reduce_2(v_1)
        v_3 = self.convbn2d_1(v_2)
        v_4 = self.reduce_5(v_3)
        v_5 = self.up1_0(v_4)
        v_6 = self.convbn2d_2(v_5)
        v_7 = self.up1_3(v_6)
        v_8 = self.up2_0(v_7)
        v_9 = self.convbn2d_3(v_8)
        v_10 = self.up2_3(v_9)
        v_11 = self.convbn2d_4(v_10)
        v_12 = self.head_2(v_11)
        v_13 = self.head_3(v_12)
        return v_13

def export_torchscript():
    net = Model()
    net.float()
    net.eval()

    torch.manual_seed(0)
    v_0 = torch.rand(1, 1024, 96, 96, dtype=torch.float)

    mod = torch.jit.trace(net, v_0)
    mod.save("gaussian_head_pnnx.py.pt")

def export_onnx():
    net = Model()
    net.float()
    net.eval()

    torch.manual_seed(0)
    v_0 = torch.rand(1, 1024, 96, 96, dtype=torch.float)

    torch.onnx.export(net, v_0, "gaussian_head_pnnx.py.onnx", export_params=True, operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK, opset_version=13, input_names=['in0'], output_names=['out0'])

def export_pnnx():
    net = Model()
    net.float()
    net.eval()

    torch.manual_seed(0)
    v_0 = torch.rand(1, 1024, 96, 96, dtype=torch.float)

    import pnnx
    pnnx.export(net, "gaussian_head_pnnx.py.pt", v_0)

def export_ncnn():
    export_pnnx()

@torch.no_grad()
def test_inference():
    net = Model()
    net.float()
    net.eval()

    torch.manual_seed(0)
    v_0 = torch.rand(1, 1024, 96, 96, dtype=torch.float)

    return net(v_0)

if __name__ == "__main__":
    print(test_inference())
