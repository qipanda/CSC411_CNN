import pandas as pd
import ImageAugmenter as ia
import scipy as sp
import numpy as np
import glob
import math
import csv


'''from raw data, get counts of the class distribution of data and also create array of labels'''
def createDataframe(groupby, drop_col_name, agg_col_name, fielname):
    train_labels_df = pd.read_csv(filename)

    #create array of labels
    train_labels = train_labels_df.as_matrix([groupby]).reshape(-1)

    #create distribution of data pd dataframe
    train_groupby = train_labels_df.groupby(groupby).count().drop(drop_col_name, axis=1)
    train_groupby.columns=[agg_col_name]

    return train_labels, train_groupby

'''import all the images into a list of ndarrays'''
def insertImageToDataframe(image_folder):
    image_list = []
    for filename in glob.glob(image_folder + '/*.jpg'):
        im = sp.misc.imread(filename)
        image_list.append(im)

    return image_list

'''make list of images to be augmented and their also a vetor of their corresponding labels'''
def createListToBeSynthesized(train_diff, agg_col_name, original_train_images, original_train_labels, width, height, depth):

    synthetic_train_images = np.empty([0, width, height, depth], dtype=np.uint8)
    synthetic_train_labels = np.empty([0, width, height, depth], dtype=np.uint8)
    train_diff_labels = train_diff.index.values.reshape(-1)
    train_diff_counts = train_diff.as_matrix().reshape(-1)

    #look through each class and generate enough examples to get to max
    for i in range(train_diff_labels.size):
        if train_diff_counts[i] == 0: #if this is the maximum set, skip
            continue

        synthetic_train_labels = np.append(synthetic_train_labels,
        np.ones(train_diff_counts[i]) * train_diff_labels[i])

        #append appropriate amount of labels to synthetic label set
        num_original_samples = np.sum(original_train_labels==train_diff_labels[i])

        '''
        # mult = sets number of num_original_samples we need to make up the difference
        # remainer = remainder of above operation, we will randomly sample this many
        samples from num_original_samples to be appended onto synthetic_train_images
        '''
        mult = math.floor(train_diff_counts[i] / num_original_samples)
        remainder = train_diff_counts[i] % num_original_samples

        if mult > 0:
            synthetic_train_images = np.append(synthetic_train_images,
            np.tile(original_train_images[original_train_labels==train_diff_labels[i]],
            (mult, 1, 1, 1)), axis=0)

        synthetic_train_images = np.append(synthetic_train_images,
        np.random.permutation(original_train_images[original_train_labels==train_diff_labels[i]])[0:remainder],
        axis=0)

    return synthetic_train_images, synthetic_train_labels

def testImages(synthetic_train_images):
    ImageAugmenter = ia.ImageAugmenter(img_width_px=128, img_height_px=128)
    synthetic_train_images = 256.0*synthetic_train_images
    import pdb; pdb.set_trace()
    for i in range(163):
        ImageAugmenter.plot_image(synthetic_train_images[i+163+1414], nb_repeat=3)

def convertAugJPG(aug_synthetic_train_images, synthetic_train_labels):
    #create csv with headers id, label, usage=Aug
    with open('aug_train.csv', 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(['Id', 'Label', 'Usage'])

        for i in range(aug_synthetic_train_images.shape[0]):
            #save jpgs into a folder
            label = './aug_train/' + str(i+1).zfill(5) + '.jpg'
            sp.misc.toimage(255*aug_synthetic_train_images[i], high=256.0, low=0.0, channel_axis=2).save(label)

            #save labels into csv
            writer.writerow([i+1, synthetic_train_labels[i], 'Aug'])

if __name__ == '__main__':
    '''define data extraction parameters'''
    class_label = 'Label'
    drop_col_name = 'Usage'
    agg_col_name = 'Count'
    filename = 'train.csv'

    '''
    # original_train_labels is ndarray of which image relates to which class from original training set
    # train_groupby is the count of samples in each class
    # train_diff is the amount of samples each class needs to become equal to the top sampled class
    # original_train_images are the original trainset images matching order with labels
    '''
    original_train_labels, train_groupby = createDataframe(class_label, drop_col_name, agg_col_name, filename)
    train_diff = -1*(train_groupby - train_groupby.max())
    original_train_images = np.array(
    insertImageToDataframe('train'), dtype=np.uint8)

    synthetic_train_images, synthetic_train_labels = createListToBeSynthesized(train_diff, agg_col_name,
    original_train_images, original_train_labels, width=128, height=128, depth=3)

    '''
    now augment synthetic_train_images, first define image augmentation parameters
    '''
    ImageAugmenter = ia.ImageAugmenter(img_width_px=128, img_height_px=128, channel_is_first_axis=False,
    hflip=True,
    vflip=False,
    scale_to_percent=1.3,
    scale_axis_equally=False,
    rotation_deg=45,
    shear_deg=10,
    translation_x_px=5,
    translation_y_px=5,
    transform_channels_equally=True)

    #perform augmentations and modify to match original formats
    aug_synthetic_train_images = ImageAugmenter.augment_batch(synthetic_train_images)
    aug_synthetic_train_images = 255*aug_synthetic_train_images
    aug_synthetic_train_images = aug_synthetic_train_images.astype(np.uint8)
    synthetic_train_labels = synthetic_train_labels.astype(np.int64)

    #create dictionary of train_images, aug_train_images, test_images, train_labels, aug_train_labels
    test_images = np.array(
    insertImageToDataframe('val'), dtype=np.uint8)

    original_train_data = {
        'train_images':original_train_images,
        'train_labels':original_train_labels
    }

    aug_train_data = {
        'aug_train_images':aug_synthetic_train_images,
        'aug_train_labels':synthetic_train_labels
    }

    test_data = {
        'test_images':test_images
    }

    np.savez('original_train_data.npz', **original_train_data)
    np.savez('aug_train_data.npz', **aug_train_data)
    np.savez('test_data.npz', **test_data)

    # import pdb; pdb.set_trace()
    # o_test = np.load('original_train_data.npz')
    # a_test = np.load('aug_train_data.npz')
    # t_test = np.load('test_data.npz')

    # convertAugJPG(aug_synthetic_train_images, synthetic_train_labels)

    # import pdb; pdb.set_trace()
    # sp.misc.toimage(aug_synthetic_train_images[0], high=1.0, low=0.0, channel_axis=3).save('test.jpg')
    # testImages(aug_synthetic_train_images)
