import os
import tensorflow as tf
import tqdm

count = 0
for root, dirs, files in os.walk('trainingdata/kaggle-hwdb-images-archive/Train/'):
    for name in files:
        path = os.path.join(root, name)
        # if root == 'trainingdata/kaggle-hwdb-images-archive/Train/':
        #     # print('skipping ^^')
        #     continue
        if name == '.DS_Store':
            print('skipping ds store')
            continue
        count += 1
        if count <= 10184 * 256:
            print(f'{count}: {path}')
            with open(path, 'rb') as f:
                data = f.read()
                image = tf.image.decode_png(data)
