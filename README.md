# CLRNet-onnxruntime-and-tensorrt-demo
This is the onnxruntime and tensorrt inference code for CLRNet: Cross Layer Refinement Network for Lane Detection (CVPR 2022). Official code: https://github.com/Turoad/CLRNet

[skip to onnx demo](#onnx) and [skip to tensorrt demo](#tensorrt)

## Note
1、Making onnx supported op grid_sampler. <br>
2、Using this code you can successfully convert to onnx model and inference an onnxruntime demo. A new version demo only use numpy to do post-processing,  easy to deploy but more time cost for NMS. <br>
3、Modifications according to the following operations will affect the training code, this code only for onnx inference. <br> 
4、It mainly includes two parts: model inference and post-processing. <br>
5、Supporting convert to tensorrt engine.

## convert and test onnx <a name="onnx"></a>
1、git official code and install original environment by refer to https://github.com/Turoad/CLRNet <br>
2、git clone this code <br>
3、cp clr_head.py   to your_path/CLRNet/clrnet/models/heads/ <br>
4、mkdir your_path/CLRNet/modules/ and cp grid_sample.py to your_path/CLRNet/modules/ <br>
5、cp torch2onnx.py  to your_path/CLRNet/ <br>
6、For example, run
```
python torch2onnx.py configs/clrnet/clr_resnet18_tusimple.py  --load_from tusimple_r18.pth
```
my deployment log is here https://github.com/xuanandsix/CLRNet-onnxruntime-and-tensorrt-demo/blob/main/my_log/test_onnx.log <br>
7、cp test.jpg to your_path/CLRNet/  and run

1) NMS based on torch and cpython. 
```
python demo_onnx.py
````
2) NMS based on numpy.
```
python demo_onnx_new.py 
```
## onnx output 

<img src="https://github.com/xuanandsix/CLRNet-onnxruntime-and-tensorrt-demo/raw/main/imgs/output_onnx.png" width="320" height="180">

## convert and test tensorrt <a name="tensorrt"></a>
Tensorrt version needs to be greater than 8.4. This code is implemented in TensorRT-8.4.0.6. <br>
*GatherElements error、IShuffleLayer error、`is_tensor()' failed* have been resolved. <br>
1、install tensorrt and compilation tools *trtexec*. <br>
2、install *polygraphy* to help modify the onnx model. You can install by
```
pip install nvidia-pyindex
pip install polygraphy
pip install onnx-graphsurgeon
```
and run
```
polygraphy surgeon sanitize your_path/tusimple_r18.onnx --fold-constants --output your_path/tusimple_r18.onnx
```
3、 convert to tensorrt and get tusimple_r18.engine
```
./trtexec --onnx=your_path/tusimple_r18.onnx --saveEngine=your_path/tusimple_r18.engine --verbose
```
4、test tensorrt demo.
```
python demo_trt.py
```

## tensorrt output 

<img src="https://github.com/xuanandsix/CLRNet-onnxruntime-and-tensorrt-demo/raw/main/imgs/output_trt.png" width="320" height="180">


## TO DO 
- [x] Optimize post-processing. 
- [x] Tensorrt demo.
- [ ] Cpp demo.

