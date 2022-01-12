# U-MVS-CASMVSNET

## About

CascadeMVSNet is utilized as the backbone.
The pretrained weight is provided for a fast evaluation.
The training code will come in a few days.

## How to use?

### Environment

 - pytorch 1.3.0
 - torchvision 0.4.2
 - cuda 10.1
 - apex
   - Note: apex is an optional library to accelerate the training, though it may degrade the performance a little.
   - Commands for installing apex: `git clone https://github.com/NVIDIA/apex; cd apex; python setup.py install;`
   - For more information, please to NVIDIA's official repository: https://github.com/NVIDIA/apex.

The conda environments are packed in `requirements.txt`.

### Evaluating pretrained model on Tanks&Temples benchmark

 - Download preprocessed dataset: [Tanks&Temples](https://drive.google.com/file/d/1gAfmeoGNEFl9dL4QcAU4kF0BAyTd-r8Z/view).
 - After decompressing the files, the dataset is organized as follows:

```
root_directory
 |--advanced
      |--Auditorium (scene name 1)
      |--Ballroom (scene name 2)
      |-- ...
 |--intermediate
      |--Family (scene name 1)
      |--Francis (scene name 2)
      |-- ...
```
 - Modify the `DATAPATH` variable in `scripts/test_tanks_pretrained.sh` according to the exact path of the Tanks&Temples dataset in your machine.
 - Run `bash scripts/test_tanks_pretrained.sh YOUR_DATA_SPLIT YOUR_GPU_ID` to evaluate the pretrained model.
    - `YOUR_DATA_SPLIT` means which partition of Tanks&Temples dataset is used, for example: `advanced`, or `intermediate`.
    - `YOUR_GPU_ID` represents the id of the used GPU. In default, `0` can be used.
    - For example, `bash scripts/test_tanks_pretrained.sh intermediate 0`, `bash scripts/test_tanks_pretrained.sh advanced 0`.
    - After running this command, the generated 3D point clouds will be saved in `./outputs_tanks` directory.
 - To submit the results to the official website of [Tanks&Temples](https://www.tanksandtemples.org/), please follow their official instructions to organize the files as follows:

```
upload_directory
 |--Auditorium.log (camera parameters in scene 1)
 |--Auditorium.ply (generated ply file from scene 1)
 |--Ballroom.log (camera parameters in scene 2)
 |--Ballroom.ply (generated ply file from scene 2)
 |-- ...
 |-- t2_submission_credentials.txt (please obtain this credential file from official website)
 |-- upload_t2_results.py (please download this script from official website)
```
 - Note:
   - The ply files of all scenes can be found in `./outputs_tanks`.
   - The log files can be found in each scene directory from provided data in [Tanks&Temples](https://drive.google.com/file/d/1gAfmeoGNEFl9dL4QcAU4kF0BAyTd-r8Z/view).
   - The `t2_submission_credentials.txt` and `upload_t2_results.py` should be obtained from official website： [Tanks&Temples](https://www.tanksandtemples.org/).
   - To submite the results, you can type the command: `python upload_t2_results.py --group (intermediate/advanced/both)`. Select each one of `intermediate`, `advanced`, `both` if you want to upload the results of intermediate dataset or advanced dataset or both.
   - After submitting the results, you can click the evaluation button on the official site of Tanks&Temples and wait for several hours. The evaluation results will be sent to your registered e-mail box once the evaluation is finished.

### Evaluating pretrained model on DTU benchmark

 - Download preprocessed dataset: [DTU](https://drive.google.com/file/d/135oKPefcPTsdtLRzoDAQtPpHuoIrpRI_/view) and unzip it as the $TESTPATH folder which is organized as follows:

```
unzipped_data_directory ($TESTPATH)
 |--dtu
     |-- scan1
     |-- scan4
     |-- ...
```
 - Modify the `$TESTPATH` in `test_pretrained.sh` according to the exact path in your machine.
 - Prepare the `fusibile` following the instructions in the `fusion` section of [u_mvs_mvsnet](../u_mvs_mvsnet).
 - Run the script: `bash scripts/test_pretrained.sh high`.
   - Hyperparameters: `num_consistent`, `prob_threshold`, `disp_threshold` can be modified by your own.
 - The results will be saved in `./outputs` directory.
 - Run `bash scripts/arange.sh`.
 - Evaluate the perfomance on DTU benchmark following the instructions in the `Benchmark results on DTU` section of [u_mvs_mvsnet](../u_mvs_mvsnet).

## Note

 - It is suggested to use pytorch 1.2.0-1.4.0. The newest ones like pytorch 1.9.0 may fail to reproduce the same results.