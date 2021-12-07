# classify-handwritten-characters
Classify handwritten Chinese characters

## Project Status
* Model can be created with ~20% accuracy. Not so good yet.
* Works with the old-style "-c.gnt" files, but fails with "-f.gnt", not sure why yet
* Predictions can be made with the python code
* Android app uses binaries that no longer exist, so will need to be rewritten

## Prerequisites for training the model
* HWDB1.1 (1.1 million image samples. [info][3], [train.zip (1873 MB)][1], [test.zip (471 MB)][2])
* Tensorflow 2.7
* Python 3
* Python modules in requirements.txt

## Setting up the python environment
1. `brew install openblas` -- required for scikit-image to build correctly
1. `brew install hdf5` -- required for installing tensorflow-macos
1. `python3 -m venv env`
1. `source env/bin/activate`
1. `python3 -m pip install -r requirements.txt`

## Predicting something a trained model
1. Predict something with `python predict.py trained_model.tf png_image_sample`

## Recreating the characters.index:

Note -- you shouldn't need to do this unless something has gone horribly wrong.

1. `python create_character_index.py training_dir1 dir2 dir3...`

## Creating the model
1. Unzip the training and test sets
1. `python create_records.py hwdb.train.tfrecords training_dir1 training_dir2 training_dir3...`
1. `python create_records.py hwdb.test.tfrecords test_dir1 test_dir2 test_dir3...`
1. `python train_model.py`

Note that the model is saved (and overwritten) after every epoch.

## Copying the new model into the android app
1. `./update_android_app_model.sh`

## TODO
- [x] Revive and update model for Tensorflow 2.7 (Dec 2021)
- [ ] Make the new-style "-c.gnt" files work
- [ ] Remove the old git lfs pre-trained models
- [ ] Revive and update Android app
- [ ] Better comment and document python code
- [ ] Send drawings to server for better training data
- [ ] Add second model for processing characters that are composed of strokes

[1]: http://www.nlpr.ia.ac.cn/databases/download/feature_data/HWDB1.1trn_gnt.zip
[2]: http://www.nlpr.ia.ac.cn/databases/download/feature_data/HWDB1.1tst_gnt.zip
[3]: http://www.nlpr.ia.ac.cn/databases/handwriting/Offline_database.html
