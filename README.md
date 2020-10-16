# SlowFast Networks for Action Recognition on EPIC-KITCHENS-100

The code in this repo is a clone from [https://github.com/facebookresearch/SlowFast](https://github.com/facebookresearch/SlowFast) and adapted to train on the EPIC-KITCHENS-100 dataset. Particularly:

- We added a dataloader for EPIC-KITCHENS-100
- We added a training configuration file for EPIC-KITCHENS-100
- We adapted the code to train on verb+noun as multi-task learning

All the code to support EPIC-KITCHENS-100 is written by [Evangelos Kazakos](https://github.com/ekazakos).

## Citing

When using this code, kindly reference:

```
@ARTICLE{Damen2020RESCALING,
   title={Rescaling Egocentric Vision},
   author={Damen, Dima and Doughty, Hazel and Farinella, Giovanni Maria  and and Furnari, Antonino 
           and Ma, Jian and Kazakos, Evangelos and Moltisanti, Davide and Munro, Jonathan 
           and Perrett, Toby and Price, Will and Wray, Michael},
           journal   = {CoRR},
           volume    = {abs/2006.13256},
           year      = {2020},
           ee        = {http://arxiv.org/abs/2006.13256},
} 
```
and
```
@misc{fan2020pyslowfast,
  author =       {Haoqi Fan and Yanghao Li and Bo Xiong and Wan-Yen Lo and
                  Christoph Feichtenhofer},
  title =        {PySlowFast},
  howpublished = {\url{https://github.com/facebookresearch/slowfast}},
  year =         {2020}
}
```


## Pretrained model

You can download our pretrained model on EPIC-KITCHENS-100 from [this link](https://www.dropbox.com/s/uxb6i2xkn91xqzi/SlowFast.pyth?dl=0)

## Preparation

- Please install all the requirements found in the original SlowFast repo ([link](https://github.com/facebookresearch/SlowFast/blob/master/INSTALL.md))
* Add this repository to $PYTHONPATH.
```
export PYTHONPATH=/path/to/SlowFast/slowfast:$PYTHONPATH
```
* From the annotation repository of EPIC-KITCHENS-100 ([link](https://github.com/epic-kitchens/epic-kitchens-100-annotations)), download: EPIC_100_train.pkl, EPIC_100_validation.pkl, and EPIC_100_test_timestamps.pkl. EPIC_100_train.pkl and EPIC_100_validation.pkl will be used for training/validation, while EPIC_100_test_timestamps.pkl will be used to obtain the scores to submit in the AR challenge.
* Download only the RGB frames of EPIC-KITCHENS-100 dataset using the download scripts found [here](https://github.com/epic-kitchens/epic-kitchens-download-scripts). 
The training/validation code expects the following folder structure for the dataset:
```
├── dataset_root
|   ├── P01
|   |   ├── rgb_frames
|   |   |   |    ├── P01_01
|   |   |   |    |    ├── frame_0000000000.jpg
|   |   |   |    |    ├── frame_0000000001.jpg
|   |   |   |    |    ├── .
|   |   |   |    |    ├── .
|   |   |   |    |    ├── .
|   |   |   |    .    
|   |   |   |    .    
|   |   |   |    .
|   ├── .
|   ├── .
|   ├── .
|   ├── P37
|   |   ├── rgb_frames
|   |   |   |    ├── P37_101
|   |   |   |    |    ├── frame_0000000000.jpg
|   |   |   |    |    ├── frame_0000000001.jpg
|   |   |   |    |    ├── .
|   |   |   |    |    ├── .
|   |   |   |    |    ├── .
|   |   |   |    .    
|   |   |   |    .    
|   |   |   |    .
```
So, after downloading the dataset navigate under <participant_id>/rgb_frames for each participant and untar each video's frames in its corresponding folder, e.g for P01_01.tar you should create a folder P01_01 and extract the contents of the tar file inside.

## Training/validation
To train the model run:
```
python tools/run_net.py --cfg configs/EPIC-KITCHENS/SLOWFAST_8x8_R50.yaml NUM_GPUS num_gpus 
OUTPUT_DIR /path/to/output_dir EPICKITCHENS.VISUAL_DATA_DIR /path/to/dataset 
EPICKITCHENS.ANNOTATIONS_DIR /path/to/annotations
```
To validate the model run:
```
python tools/run_net.py --cfg configs/EPIC-KITCHENS/SLOWFAST_8x8_R50.yaml NUM_GPUS num_gpus 
OUTPUT_DIR /path/to/experiment_dir EPICKITCHENS.VISUAL_DATA_DIR /path/to/dataset 
EPICKITCHENS.ANNOTATIONS_DIR /path/to/annotations TRAIN.ENABLE False TEST.ENABLE True 
TEST.CHECKPOINT_FILE_PATH /path/to/experiment_dir/checkpoints/checkpoint_best.pyth
```
After tuning the model's hyperparams using the validation set, we train the model that will be used for obtaining the test set's scores on the concatenation of the training and validation sets. To train the model on the concatenation of the training and validation sets run:
```
python tools/run_net.py --cfg configs/EPIC-KITCHENS/SLOWFAST_8x8_R50.yaml NUM_GPUS num_gpus 
OUTPUT_DIR /path/to/output_dir EPICKITCHENS.VISUAL_DATA_DIR /path/to/dataset 
EPICKITCHENS.ANNOTATIONS_DIR /path/to/annotations EPICKITCHENS.TRAIN_PLUS_VAL True
```
To obtain scores on the test set (using the model trained on the concatenation of the training and validation sets) run:
```
python tools/run_net.py --cfg configs/EPIC-KITCHENS/SLOWFAST_8x8_R50.yaml NUM_GPUS num_gpus 
OUTPUT_DIR /path/to/experiment_dir EPICKITCHENS.VISUAL_DATA_DIR /path/to/dataset 
EPICKITCHENS.ANNOTATIONS_DIR /path/to/annotations TRAIN.ENABLE False TEST.ENABLE True 
TEST.CHECKPOINT_FILE_PATH /path/to/experiment_dir/checkpoints/checkpoint_best.pyth 
EPICKITCHENS.TEST_LIST EPIC_100_test_timestamps.pkl EPICKITCHENS.TEST_SPLIT test
```

## License 

The code is published under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License, found [here](https://creativecommons.org/licenses/by-nc-sa/4.0/).
