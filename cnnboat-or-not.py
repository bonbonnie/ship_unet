

import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from keras import models, layers
from keras.optimizers import Adam
from skimage.util.montage import montage2d as montage
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import gc; gc.enable()  # memory is tight


GAUSSIAN_NOISE = 0.1
UPSAMPLE_MODE = 'SIMPLE'
# number of validation images to use
VALID_IMG_COUNT = 1000
# maximum number of training images
MAX_TRAIN_IMAGES = 15000   #10000
BASE_MODEL='DenseNet121' # ['VGG16', 'RESNET52', 'InceptionV3', 'Xception', 'DenseNet169', 'DenseNet121']
IMG_SIZE =(224, 224)  # [(224, 224), (299, 299), (384, 384), (512, 512), (640, 640)],  at least 221x221
BATCH_SIZE = 64 # [1, 8, 16, 32, 64]
DROPOUT = 0.5
DENSE_COUNT = 128
LEARN_RATE = 1e-4
RGB_FLIP = 1 # should rgb be flipped when rendering images

montage_rgb = lambda x: np.stack([montage(x[:, :, :, i]) for i in range(x.shape[3])], -1)
ship_dir = '/media/bonnie/zsdisk/kaggle/'
train_image_dir = os.path.join(ship_dir, 'train')
test_image_dir = os.path.join(ship_dir, 'test')



# Examine Number of Ship Images
masks = pd.read_csv(os.path.join('/media/bonnie/zsdisk/kaggle/', 'train_ship_segmentations.csv'))
print(masks.shape[0], 'all found')   # (131030, 'all found')
print(masks['ImageId'].value_counts().shape[0], 'images found')    # (104070, 'images found')
print(masks['EncodedPixels'].value_counts().shape[0], 'masks found')    # (56030, 'masks found')
# none_ship: 131030-56030=75000, has_ship: 104070-75000=29070

masks['path'] = masks['ImageId'].map(lambda x: os.path.join(train_image_dir, x))

# # Split into training and validation groups
# We stratify by the number of boats appearing so we have nice balances in each set
# Here we examine how often ships appear and replace the ones without any ships with 0

from sklearn.model_selection import train_test_split
masks['ships'] = masks['EncodedPixels'].map(lambda c_row: 1 if isinstance(c_row, str) else 0)
unique_img_ids = masks.groupby('ImageId').agg({'ships': 'sum'}).reset_index()
unique_img_ids['has_ship'] = unique_img_ids['ships'].map(lambda x: 1.0 if x>0 else 0.0)
unique_img_ids['has_ship_vec'] = unique_img_ids['has_ship'].map(lambda x: [x])
masks.drop(['ships'], axis=1, inplace=True)
train_ids, valid_ids = train_test_split(unique_img_ids, 
                 test_size = 0.3, 
                 stratify = unique_img_ids['ships'])
train_df = pd.merge(masks, train_ids)
valid_df = pd.merge(masks, valid_ids)
print(train_df.shape[0], 'training masks')  # (91709, 'training masks')
print(valid_df.shape[0], 'validation masks')  # (39321, 'validation masks')


# limit size of training set (otherwise it takes too long)

train_df = train_df.sample(min(MAX_TRAIN_IMAGES, train_df.shape[0]))



from keras.preprocessing.image import ImageDataGenerator
if BASE_MODEL=='VGG16':
    from keras.applications.vgg16 import VGG16 as PTModel, preprocess_input
elif BASE_MODEL=='RESNET52':
    from keras.applications.resnet50 import ResNet50 as PTModel, preprocess_input
elif BASE_MODEL=='InceptionV3':
    from keras.applications.inception_v3 import InceptionV3 as PTModel, preprocess_input
elif BASE_MODEL=='Xception':
    from keras.applications.xception import Xception as PTModel, preprocess_input
elif BASE_MODEL=='DenseNet169': 
    from keras.applications.densenet import DenseNet169 as PTModel, preprocess_input
elif BASE_MODEL=='DenseNet121':
    from keras.applications.densenet import DenseNet121 as PTModel, preprocess_input
else:
    raise ValueError('Unknown model: {}'.format(BASE_MODEL))


from keras.preprocessing.image import ImageDataGenerator
dg_args = dict(featurewise_center = False, 
               samplewise_center = False,
               rotation_range = 45,
               width_shift_range = 0.1,
               height_shift_range = 0.1,
               shear_range = 0.01,
               zoom_range = [0.9, 1.25],
               brightness_range = [0.5, 1.5],
               horizontal_flip = True,
               vertical_flip = True,
               fill_mode = 'reflect',
               data_format = 'channels_last',
               preprocessing_function = preprocess_input)
valid_args = dict(fill_mode = 'reflect',
                  data_format = 'channels_last',
                  preprocessing_function = preprocess_input)

core_idg = ImageDataGenerator(**dg_args)
valid_idg = ImageDataGenerator(**valid_args)


def flow_from_dataframe(img_data_gen, in_df, path_col, y_col, **dflow_args):
    base_dir = os.path.dirname(in_df[path_col].values[0])
    print('## Ignore next message from keras, values are replaced anyways')
    df_gen = img_data_gen.flow_from_directory(base_dir, 
                                     class_mode = 'sparse',
                                    **dflow_args)
    df_gen.filenames = in_df[path_col].values
    df_gen.classes = np.stack(in_df[y_col].values)
    df_gen.samples = in_df.shape[0]
    df_gen.n = in_df.shape[0]
    df_gen._set_index_array()
    df_gen.directory = '' # since we have the full path
    print('Reinserting dataframe: {} images'.format(in_df.shape[0]))
    return df_gen



train_gen = flow_from_dataframe(core_idg, train_df,
                             path_col = 'path',
                            y_col = 'has_ship_vec', 
                            target_size = IMG_SIZE,
                             color_mode = 'rgb',
                            batch_size = BATCH_SIZE)

# used a fixed dataset for evaluating the algorithm
valid_x, valid_y = next(flow_from_dataframe(valid_idg,
                               valid_df,
                             path_col = 'path',
                            y_col = 'has_ship_vec',
                            target_size = IMG_SIZE,
                             color_mode = 'rgb',
                            batch_size = VALID_IMG_COUNT))   # one big batch
print(valid_x.shape, valid_y.shape)

t_x, t_y = next(train_gen)

# # Build a Model
# We build the pre-trained top model and then use a global-max-pooling (we are trying to detect any ship in the image and thus max is better suited than averaging (which would tend to favor larger ships to smaller ones). 
base_pretrained_model = PTModel(input_shape=t_x.shape[1:],
                              include_top=False, weights='imagenet')
base_pretrained_model.trainable = False


# ## Setup the Subsequent Layers
# Here we setup the rest of the model which we will actually be training

img_in = layers.Input(t_x.shape[1:], name='Image_RGB_In')
img_noise = layers.GaussianNoise(GAUSSIAN_NOISE)(img_in)
pt_features = base_pretrained_model(img_noise)
pt_depth = base_pretrained_model.get_output_shape_at(0)[-1]
bn_features = layers.BatchNormalization()(pt_features)
feature_dropout = layers.SpatialDropout2D(DROPOUT)(bn_features)
gmp_dr = layers.GlobalMaxPooling2D()(feature_dropout)
dr_steps = layers.Dropout(DROPOUT)(layers.Dense(DENSE_COUNT, activation = 'relu')(gmp_dr))
out_layer = layers.Dense(1, activation = 'sigmoid')(dr_steps)

ship_model = models.Model(inputs = [img_in], outputs = [out_layer], name = 'full_model')

ship_model.compile(optimizer = Adam(lr=LEARN_RATE), 
                   loss = 'binary_crossentropy',
                   metrics = ['binary_accuracy'])

ship_model.summary()



weight_path="{}_weights.best.hdf5".format('boat_detector')

checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1, 
                             save_best_only=True, mode='min', save_weights_only = True)

reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=10, verbose=1, mode='auto', epsilon=0.0001, cooldown=5, min_lr=0.0001)
early = EarlyStopping(monitor="val_loss", 
                      mode="min", 
                      patience=10) # probably needs to be more patient, but kaggle time is limited
callbacks_list = [checkpoint, early, reduceLROnPlat]


train_gen.batch_size = BATCH_SIZE
ship_model.fit_generator(train_gen, 
                      validation_data = (valid_x, valid_y), 
                      epochs = 20,
                      callbacks = callbacks_list,
                      workers = 3)


ship_model.load_weights(weight_path)
ship_model.save('full_ship_model.h5')


# # Run the test data
# We use the sample_submission file as the basis for loading and running the images.
test_paths = os.listdir(test_image_dir)
print(len(test_paths), 'test images found')
submission_df = pd.read_csv('/media/bonnie/zsdisk/kaggle/sample_submission.csv')
submission_df['path'] = submission_df['ImageId'].map(lambda x: os.path.join(test_image_dir, x))


# # Setup Test Data Generator
# We use the same generator as before to read and preprocess images

BATCH_SIZE = BATCH_SIZE*2 # we can use larger batches for inference
test_gen = flow_from_dataframe(valid_idg, 
                               submission_df, 
                             path_col = 'path',
                            y_col = 'ImageId', 
                            target_size = IMG_SIZE,
                             color_mode = 'rgb',
                            batch_size = BATCH_SIZE, 
                              shuffle = False)


from tqdm import tqdm_notebook
all_scores = dict()
for _, (t_x, t_names) in zip(tqdm_notebook(range(test_gen.n//BATCH_SIZE+1)), test_gen):
    t_y = ship_model.predict(t_x)[:, 0]
    for c_id, c_score in zip(t_names, t_y):
        all_scores[c_id] = c_score

# # Make the RLE data if there is a ship
# Here we make the RLE data for a positive image (assume every pixel is ship)
# ref: https://www.kaggle.com/paulorzp/run-length-encode-and-decode
def rle_encode(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)
# empty image
zp_dim = 10
out_img = np.ones((768-2*zp_dim, 768-2*zp_dim), dtype=bool)
out_img = np.pad(out_img, ((zp_dim, zp_dim),), mode='constant', constant_values=0)
plt.matshow(out_img)
print(out_img.shape)
pos_ship_str = rle_encode(out_img)
print(pos_ship_str[:50])

# add the whole image if it is above the threshold
submission_df['EncodedPixels'] = submission_df['score'].map(lambda x: pos_ship_str if x>0.5 else None)
out_df = submission_df[['ImageId', 'EncodedPixels']]
out_df.to_csv('tbt_submission.csv', index=False)
out_df.head(20)


