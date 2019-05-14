# Create mask file for each source file, using data from segmentation file, which is RLE.
# In this case, RLE means that there is one line in the file for each boat in a file. Each line
# consists of a number of pairs. The first element in the pair is the starting pixel number and the
# second is the number of pixels that are also part of that boat. The pixel numbers start at 0 in the
# top left corner of the image then go down before going across.
# Masks will be black where objects are - 255 for each channel - and 0 everwhere else.
# November 20, 2018 change. Since second sample network is scaling and centering masks, to
# have values of either 0 or 1, am creating new function here to create mask of 0 for no boat,
# 1 for boat, so can skip the mask normalization when running the network, hopefully making
# target clearer.
# Modified May 8, 2019 to save masks as png, not jpg, to avoid jpeg compression losses.


import os
import matplotlib.image as mpimg
import numpy as np
import pandas as pd
import math


# Globals
# Live Values
base_dir = '/home/ubuntu/notebooks/kaggle/AirbusShipId/data'
#base_dir = '/home/ubuntu/fastai/practical/exercises/lesson1/ocean/train/bigboats'
labelsFile = 'train_ship_segmentations_v2.csv'      # File containing full data

# Test Values
#base_dir = '/home/ubuntu/notebooks/kaggle/AirbusShipId/test'
#base_dir = 'E:/code/kaggle/Airbus ShipId/data'
#base_dir = 'E:/code/kaggle/Airbus ShipId/segment_progtest'
#labelsFile = 'train_ship_segmentations_small_test.csv'  # File containing first 202 rows, for testing reading file.
#base_dir= '/home/gregw/code/Kaggle/ShipId'

train_dir = os.path.join(base_dir, 'train/images')
validate_dir = os.path.join(base_dir, 'validate/images')
test_dir = os.path.join(base_dir, 'test/images')

#test_dir= '/home/ubuntu/fastai/practical/exercises/lesson1/ocean/train/bigboats'
img_size = 768
img_shape = (img_size, img_size, 3)

segResultsFilename = os.path.join(base_dir, labelsFile)


def getFileToDirMap(img_dirs):
    # Create mapping from filename to directory name, for each file in each directory in given collection.
    file_map = {}
    for src_dir in img_dirs:
        srcFiles = [name.lower() for name in os.listdir(src_dir)
                    if os.path.isfile(os.path.join(src_dir, name))]
        num_files = len(srcFiles)
        print( "For directory ", src_dir, ", working on ", num_files, " files.")
        for name in srcFiles:
            if not ( name in file_map):
                file_map[name] = src_dir
            else:
                print("found duplicate filename: ", name, " in directory: ", src_dir)

    return file_map


def createMasks(file_to_dir_map):
    # Create a mask for each image, in parallel directory called labels.
    # Note that read_csv seems to automatically skip the header row at the top of the file.
    segResults = pd.read_csv(segResultsFilename, sep=',', index_col='ImageId')
    print("1. segResults.shape", segResults.shape)
    this_mask = np.zeros(img_shape, dtype = np.uint8)
    last_filename = 'zzz'
    n = 1
    for row in segResults.itertuples():
        print(row[0], ", ", row[1])
        this_filename = row[0].lower()
        # Extra check for testing - don't bother creating mask if not going to save this file, because
        # it isn't in the testing directories.
        if this_filename in file_to_dir_map:
            if not (this_filename == last_filename):
                if not (last_filename == 'zzz'):
                    saveMaskAsPng(this_mask, last_filename, file_to_dir_map)
                    #saveMaskAsCsv(this_mask, last_filename, file_to_dir_map)
                this_mask = np.zeros(img_shape, dtype=np.uint8)
                last_filename = this_filename
            if not pd.isnull(row[1]):
                pixels = getPixels(row[1])
                #applyPixels(pixels, this_mask )
                applyPixelsBinary(pixels, this_mask )
                #applyPixelsBW(pixels, this_mask )
        n += 1
        #if n > 40: break	# Used for testing, so don't have to go through all images.

    # Save last file.
    saveMaskAsPng(this_mask, last_filename, file_to_dir_map)
    #saveMaskAsCsv(this_mask, last_filename, file_to_dir_map)


def getPixels(src_line):
    # Data in file is pixel number, number of pixels pairs, all space delimited.
    # Want to return array of pixel locations (row, column) which are boat.

    boatPixels = []
    list = src_line.split(" " )
    #print( 'list: ', list )
    for i in range(0, len(list), 2):
        pixel1 = getPixel(list[i])

        # After finding row, col coordinates of given pixel number, store it, and next n pixels,
        # where n is the next item in list.
        #print( "adding ", list[i+1], " pixels." )
        for j in range(int(list[i+1])):
            boatPixels.append([pixel1[0], pixel1[1]])
            pixel1[0] += 1

    return boatPixels


def getPixel(pixel_num):
    # Convert a pixel number to a row, col coordinate.
    # Assuming picture size of 768*768. Turns out that they count down and then
    # across, so pixel 2 is row 1, column 0. Also, since first pixel is 1 rather than 0,
    # have to decrement before start arithmetic.
    nPixelNum = int(pixel_num) - 1
    y = math.trunc(nPixelNum / img_size)
    x = nPixelNum % img_size

    return [x, y]


def applyPixels( pixels, mask ):
    # Change values for specified pixels to 255, in given mask.
    # Changes provided mask.
    for row, col in pixels:
        mask[row, col] = [ 0, 128, 0]
        #mask[row, col, 0] = 255
        #mask[row, col, 1] = 255
        #mask[row, col, 2] = 255


def applyPixelsBinary( pixels, mask ):
    # Change green channel for specified pixels to 1, in given mask.
    # Changes provided mask.
    for row, col in pixels:
        mask[row, col] = [1, 1, 1]

def applyPixelsBW( pixels, mask ):
    # Expect single channel image, set given pixels to white.
    # Changes provided mask.
    # Tried to use this to create b&w png file, but matplotlib.imsave only accepts 3 channel.
    for row, col in pixels:
        mask[row, col] = 1


def saveMaskAsJpg(this_mask, filename, file_to_dir_map):
    # Save mask as jpg in labels dir next to directory original image was in.
    if filename in file_to_dir_map:
        src_dir = file_to_dir_map[filename]
        dest_dir, zz = os.path.split(src_dir)
        dest_dir = os.path.join(dest_dir, 'labels')
        ensureDirExists(dest_dir)
        dest_filename = os.path.join(dest_dir, filename)
        mpimg.imsave(dest_filename, this_mask, format='jpg')
    else:
        print( "Unable to find file to directory mapping for file ", filename)


def saveMaskAsPng(this_mask, filename, file_to_dir_map):
    # Receives b&w mask, name of source jpg file, directory original jpg file was in.
    # Save mask as png in labels dir next to directory original image was in.
    if filename in file_to_dir_map:
        src_dir = file_to_dir_map[filename]
        dest_dir, zz = os.path.split(src_dir)
        dest_dir = os.path.join(dest_dir, 'labels')
        ensureDirExists(dest_dir)
        base, ext = os.path.splitext(filename)
        new_filename = base + ".png"
        dest_filename = os.path.join(dest_dir, new_filename)
        mpimg.imsave(dest_filename, this_mask, format='png')
    else:
        print( "Unable to find file to directory mapping for file ", filename)


def saveMaskAsCsv(this_mask, filename, file_to_dir_map):
    # Save mask as csv in labels dir next to directory original image was in.
    if filename in file_to_dir_map:
        src_dir = file_to_dir_map[filename]
        dest_dir, zz = os.path.split(src_dir)
        dest_dir = os.path.join(dest_dir, 'labels')
        ensureDirExists(dest_dir)
        base, ext = os.path.splitext(filename)
        dest_filename = os.path.join(dest_dir, base + "r.csv")
        np.savetxt(dest_filename, this_mask[:, :, 0], delimiter=',')
        dest_filename = os.path.join(dest_dir, base + "g.csv")
        np.savetxt(dest_filename, this_mask[:, :, 1], delimiter=',')
        saveNonZeroAsCsv( os.path.join(dest_dir, base + "nz_r.csv"), this_mask[:, :, 0])
        saveNonZeroAsCsv( os.path.join(dest_dir, base + "nz_g.csv"), this_mask[:, :, 1])
        #dest_filename = os.path.join(dest_dir, base + "b.csv")
        #np.savetxt(dest_filename, this_mask[:, :, 2], delimiter=',')
    else:
        print( "Unable to find file to directory mapping for file ", filename)


def saveNonZeroAsCsv(outputFileName, img):
    # Given a 2d array, save x, y coordinates and values for all non-zero values to a csv file.

    print( "shape: ", img.shape)
    with open(outputFileName, 'wt') as f:
        for row in range(img.shape[0]):
            for col in range(img.shape[1]):
                if img[row,col] > 0:
                    f.write(str(row) + ", " + str(col) + ": " + str(img[row,col]) + "\n")


def ensureDirExists(targetDir) :
    # Ensure that given directory exists. (Create it if it doesn't.)

    if not ( os.path.isdir(targetDir)):
        os.makedirs(targetDir)
    return


def createEmptyMask():
    this_mask = np.zeros(img_shape, dtype = np.uint8)
    mpimg.imsave('empty.png', this_mask, format='png')

#if __name__ == "__main__":
    # Create dictionary to map from image file name to directory where it is located.
#img_dirs = [test_dir]
img_dirs = [train_dir, validate_dir, test_dir]
print ('Started creating file to dir map.')
file_to_dir_map = getFileToDirMap(img_dirs)
print ('Finished creating file to dir map.')
createMasks(file_to_dir_map)

#createEmptyMask()

'''
file_map = {}
file_map['allzeros.jpg'] = base_dir
file_map['all255s.jpg'] = base_dir

this_mask = np.zeros(img_shape, dtype=np.uint8)
saveMask(this_mask, 'allzeros.jpg', file_map)
for i in range(img_size):
    for j in range(200, 400):
        this_mask[i, j, 0] = 255
        this_mask[i, j, 1] = 255
        this_mask[i, j, 2] = 255
saveMask(this_mask, 'all255s.jpg', file_map)
'''

print("complete")
