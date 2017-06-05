# classify-handwritten-characters
Classify handwritten Chinese characters

## Prerequisites
* HWDB1.1 (1.1 million image samples. [info][3], [train.zip (1873 MB)][1], [test.zip (471 MB)][2])
* OLHWDB1.1 (1.1 million stroke samples. [info][6], [train.zip (187 MB)][4], [test.zip (47 MB)][5])
* Tensorflow 1.1
* Python 3
* Python modules in requirements.txt

## Creating the model
1. Unzip the training and test sets
2. $ ./convert_gnt_to_records.py unzipped_hwdb1.1_folder
3. $ ./gnt_model
4. $ ./gnt_predict data/model-stepN.ckpt png_image_sample
5. $ ./write_pb_file data/model-stepN.ckpt output_dir

## Updating the android tensorflow library
This repository comes prepackaged with the android tensorflow library. This might require updating in the future
for a new version of tensorflow. Binaries are available on https://ci.tensorflow.org/view/Nightly/job/nightly-android/.
To update:
 * Replace android/app/libs/libandroid_tensorflow_inference_java.jar with the [nightly version][nightly_jar]
 * Replace android/app/src/main/jniLibs/armeabi-v7a/libtensorflow_inference.so with the [nightly version][nightly_so].

## TODO
- [x] Update the android app to allow drawing at a higher resolution that gets downsampled before inference
- [x] Update the android app to crop the drawing before sending to the model
- [x] Make tapping on a character bring user to dictionary entry for that character
- [x] Add undo for strokes
- [ ] Add mobile web classifier
- [ ] Update to the latest model
- [ ] Comment python code
- [ ] Send drawings to server for better training data
- [ ] Refactor python code to better separate the steps in training
- [ ] Add second model for processing characters that are composed of strokes

[1]: http://www.nlpr.ia.ac.cn/databases/download/feature_data/HWDB1.1trn_gnt.zip
[2]: http://www.nlpr.ia.ac.cn/databases/download/feature_data/HWDB1.1tst_gnt.zip
[3]: http://www.nlpr.ia.ac.cn/databases/handwriting/Offline_database.html

[4]: http://www.nlpr.ia.ac.cn/databases/download/feature_data/OLHWDB1.1trn_pot.zip
[5]: http://www.nlpr.ia.ac.cn/databases/download/feature_data/OLHWDB1.1tst_pot.zip
[6]: http://www.nlpr.ia.ac.cn/databases/handwriting/Online_database.html

[nightly_so]: https://ci.tensorflow.org/view/Nightly/job/nightly-android/lastSuccessfulBuild/artifact/out/native/libtensorflow_inference.so/armeabi-v7a/libtensorflow_inference.so
[nightly_jar]: https://ci.tensorflow.org/view/Nightly/job/nightly-android/lastSuccessfulBuild/artifact/out/libandroid_tensorflow_inference_java.jar
