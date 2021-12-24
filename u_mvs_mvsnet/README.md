# U-MVS-MVSNet

## About

MVSNet is utilized as the backbone in default.
The pretrained weight is provided for a fast evaluation.
The training code will come in a few days.

## How to use?

### Environment

 - pytorch 1.3.0
 - torchvision 0.4.2
 - cuda 10.1

The conda environments are packed in `requirements.txt`.

### Fusion

 - To fuse the per-view depth maps to a 3D point cloud, we use [fusibile](https://github.com/kysucix/fusibile) for depth fusion.
 - Build the binary executable file from fusibile first:
   - Enter the directory of `fusion/fusibile`:
   - Check the gpu architecture in your server and modify the corresponding settings in `CMakeList.txt`:
     - If 1080 Ti GPU with a computation capability of 6.0 is used, please add: `set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3 --use_fast_math --ptxas-options=-v -std=c++11 --compiler-options -Wall -gencode arch=compute_60,code=sm_60 -gencode arch=compute_60,code=sm_60)`.
     - If 2080 Ti GPU with a computation capability of 7.5 is used, please add: `set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3 --use_fast_math --ptxas-options=-v -std=c++11 --compiler-options -Wall -gencode arch=compute_75,code=sm_75 -gencode arch=compute_75,code=sm_75)`.
     - For other GPUs, please check the computation capability of your GPU in: https://developer.nvidia.com/zh-cn/cuda-gpus.
   - Create the directory by running `mkdir build`.
   - Enter the created directory, `cd build`.
   - Configure the CMake setting, `cmake ...`.
   - Build the project, `make`.
   - Then you can find the binary executable file named as `fusibile` in the `build` directory.
 - The hyper-parameters for controling the `fusibile` during depth fusion are discussed in the following part.

### Evaluating pretrained model

 - Downlad the preprocessed DTU testing data [ [Google Cloud](https://drive.google.com/file/d/135oKPefcPTsdtLRzoDAQtPpHuoIrpRI_/view) or [Baidu Cloud](https://pan.baidu.com/s/1sQAC3pmceyochNvnqpE9oA#list/path=%2F) (The password is `mo8w`) ]
   - In the Baidu Cloud link, you can find the target file in the directory: `preprocessed_input/dtu.zip`.
 - Edit the testing settings in `test_pretrained.sh`:
   - `TESTPATH`: the path of the testing dataset.
 - Run the code by `bash scripts/test_pretrained.sh 0`.
   - It id noted that the `0` here is the id of your gpu.
 - Run `bash scripts/arange.sh`.
 - Hyper-parameters to control the `fusibile` during depth fusion step can be modified in `test_pretrained.sh`:
   - `--num_consistent / --prob_threshold --disp_threshold`

### Benchmark results on DTU

 - Download the [Sample Set.zip](https://github.com/ToughStoneX/Self-Supervised-MVS/blob/main/jdacs/roboimagedata2.compute.dtu.dk/data/MVS/SampleSet.zip) and [Points.zip](http://roboimagedata2.compute.dtu.dk/data/MVS/Points.zip) from [DTU](http://roboimagedata.compute.dtu.dk/?page_id=36)'s official website. Decompress the zip files and arange the ground truth point clouds following the official instructions of DTU.
 - Edit the path settings in `ComputeStat_web.m` and `BaseEvalMain_web.m`.
   - The `datapath` should be changed according to the path of your data. For example, `dataPath='/home/xhb/dtu_eval/SampleSet/MVS Data/'`;
 - Enter the `matlab_eval/dtu` directory and run the matlab evaluation code, `bash run.sh`. The results will be presented in a few hours. The time consumption is up to the available threads enabled in the Matlab environment.

### Note 

 - It is suggested to use pytorch 1.2.0-1.4.0. The newest ones like pytorch 1.9.0 may fail to reproduce the same results.