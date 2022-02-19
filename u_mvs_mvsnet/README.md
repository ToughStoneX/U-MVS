# U-MVS-MVSNet

## About

MVSNet is utilized as the backbone in default.
The pretrained weight is provided for a fast evaluation.
~~The training code will come in a few days~~.
The training code has been uploaded.

### Note 

 - It is suggested to use pytorch 1.2.0-1.4.0. The newest ones like pytorch 1.9.0 may fail to reproduce the same results. Please check the environment before running the evaluation code.

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

### Training the model by your own

#### Breif summary

As the paper suggests, our U-MVS is separated into two stages: self-supervised pretraining stage and pseudo-label post-training stage. The sequential procedure is as follows:

 1. Train the model unsupervisedly in self-supervised pretraining stage, `bash scripts/train.sh selfsup_pretrain`.
 2. Generate pseudo labels and uncertainty maps (aleatoric and epistemic uncertainty) with the pretrained model, `bash scripts/gen_pselbl.sh`.
 3. Train the model with the generated labels in pseudo-label post-training stage, `bash scripts/train.sh pselbl_postrain`.


#### Prepare the dataset

 1. Download the preprocessed DTU training dataset [ [Google Cloud](https://drive.google.com/file/d/1eDjh-_bxKKnEuz5h-HXS7EDJn59clx6V/view) or [Baidu Cloud](https://pan.baidu.com/s/1sQAC3pmceyochNvnqpE9oA#list/path=%2F) (The password is mo8w) ].
 2. Set the variable `MVS_TRAINING` to the exact path of the downloaded dataset in: `scripts/gen_pselbl.sh` and `scripts/train.sh`.
#### Self-supervised pretraining

 1. Prepare the optical flow with the multi-view images in DTU dataset. You can either refer to the provided script: `./arflow/gen_flow.py` or directly download our preprocessed [optical flows](https://mogface.oss-cn-zhangjiakou.aliyuncs.com/xhb/share/umvs_iccv/flow.tar.gz) (about 121G).
 2. After downloading the files, enter the folder: `cd arflow`, and decompress the file with: `tar -zxvf flow.tar.gz`. Then the generated folder `flow` contains the multi-view optical flows.
 3. Return to the root directory of this project `cd ..` and train the model using: `bash scripts/train.sh selfsup_pretrain`.
 4. Hyperparameters can be modifed in `scripts/train.sh`.

#### Generating pseudo label and uncertainty map

 1. When the aforementioned self-supervised pretraining is finished, we can generate the pseudo labels and epistemic/aleatoric uncertainty maps with the saved model. For a fast evaluation, we also provide a pretrained checkpoint in `./pretrained/selfsup_pretrain/model_00065000.ckpt`.
 2. Modify the `PRETRAINED_WEIGHT` in `./scripts/gen_pselbl.sh` according to the exact path of the pretrained model in your machine.
 3. Generate the pseudo label: `bash scripts/gen_pselbl.sh`.
 4. The generated pseudo labels and uncertainty maps will be saved in the directory named `uncertainty`.
 5. For a fast evaluation, you can also download our preprocessed data directly: [uncertainty.tar.gz](https://mogface.oss-cn-zhangjiakou.aliyuncs.com/xhb/share/umvs_iccv/uncertainty.tar.gz). Decompress it with `tar -zxvf uncertainty.tar.gz` and the pseudo labels are saved in `uncertainty`.

#### Pseudo-label post-training

 1. In the stage of pseudo-label post-training, the generated epistemic uncertainty is used to filter the unreliable regions in the generated pseudo label.
 2. Run `bash scripts/train.sh pselbl_postrain`.
 3. Hyperparameters can be modifed in `scripts/train.sh`.

#### Testing the model

 1. For evaluation, you can run `bash scripts/test.sh`. The setting of this script is the same as the aforementioned settings when evaluating the pretrained model.
 2. You can modify the `CKPT_FILE` with the absolute path of target model, for example: `CKPT_FILE="./checkpoints/selfsup_pretrain-2021-12-25/model_00065000.ckpt"` and `CKPT_FILE="./checkpoints/pselbl_postrain-2021-12-31/model_00040000.ckpt"`.
