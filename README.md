# CLRNet-onnxruntime-and-tensorrt-demo
This is the onnxruntime and tensorrt inference code for CLRNet: Cross Layer Refinement Network for Lane Detection (CVPR 2022). Official code: https://github.com/Turoad/CLRNet

# Note
1、Making onnx supported op grid_sampler. <br>
2、Using this code you can successfully convert to onnx model and inference an onnxruntime demo. But I can't completely abandon torch funcation when using nms. I will try again in the future. <br>
```
// clrnet/ops/csrc/nms.cpp
#include <torch/extension.h>
#include <torch/types.h>
...
// clrnet/ops/csrc/nms_kernel.cu
#include <torch/extension.h>
...
```
```
// demo_onnx.py
...
keep, num_to_keep, _ = nms(
      torch.tensor(nms_predictions).cuda(),
      torch.tensor(scores).cuda(),
      overlap=self.nms_thres,
      top_k=self.max_lanes)
...
```
3、Modifications according to the following operations will affect the training code, this code only for onnx inference. <br> 
4、It mainly includes two parts: model inference and post-processing. <br>

## conver and test onnx
1、git official code and install original environment by refer to https://github.com/Turoad/CLRNet <br>
2、git clone this code <br>
3、cp clr_head.py   to yout_path/CLRNet/clrnet/models/heads/ <br>
4、cp grid_sample.py  to   your_path/CLRNet/modules/ <br>
5、cp torch2onnx.py  to your_path/CLRNet/ <br>
6、For example, run
```
python torch2onnx.py configs/clrnet/clr_resnet18_tusimple.py  --load_from tusimple_r18.pth
```
7、cp test.jpg to your_path/CLRNet/ run
python demo_onnx.py

## output 

<img src="https://github.com/xuanandsix/CLRNet-onnxruntime-and-tensorrt-demo/raw/main/imgs/output_onnx.png">

## TO DO 
- [ ] Optimize post-processing. 
- [ ] Tensorrt demo.

