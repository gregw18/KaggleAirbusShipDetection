# Program to display images from a directory, let user go backwards and forwards, copy some
# to one of two subdirectories. Go through images using left/right arrow keys. Copy to subdirectories
# using a for first, b for second. Can start in middle of images by providing starting filename as parameter.
# From Kaggle competition Airbus ship identification.

# Note: run "%matplotlib tk" before running this, if want images to pop up in separate window.
# 	Don't seem to be able to run it from a script.

#import cv2
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os, shutil
import pandas as pd
import sys

#from keras.preprocessing import image

# Command to pull up images in separate window, so can resize, zoom, etc.
#%matplotlib tk

#origImgName = "00d0a646b.jpg"
#origImgName = "000fd9827.jpg"
origImgName = "000e6378b.jpg"
segResultsFilename = r"train_ship_segmentations_v2.csv"

#srcDir = r"/media/gregw/Data/Code/Kaggle/Airbus ShipID/data/tanks"
#srcDir = r"/media/gregw/Data/Code/Kaggle/Airbus ShipID/data/test/images"
srcDir = r"/media/gregw/Data/Code/Kaggle/Airbus ShipID/data/train/images"
#srcDir = r"/home/gregw/code/Kaggle/ShipId/split1"


imgSize = 768
badDirName = "notreallyanyboats"
segResults = pd.DataFrame()
#segResults = pd.read_csv(segResultsFilename, sep=',', index_col='ImageId')
destDir = r"/home/gregw/code/Kaggle/ShipId/split1"

aDir = destDir + '/boat'
bDir = destDir + '/noboat'
#fig = plt.figure()
lastkey = ''
imageLabels = {}		# Dictionary, key = filename, value = label for image.


def yes_or_no(question):
	# Get yes/no response from user, return True/False.
	while True:
		reply = str(input(question + ' (y/n) ')).lower().strip()
		if len(reply) > 0:
			if reply[0] == 'y':
				return True
			if reply[0] == 'n':
				return False


def createImageLabels():
	# Populate imageLabels with list of unique filenames contained in segmentation file 
	# and associated label.

	global imageLabels
	imageLabels.clear()
	lastFname = "ZZZ"
	
	# Want to leave blanks in input as blanks, so can recognize by checking for empty strings.
	df = pd.read_csv(segResultsFilename, sep=',', index_col='ImageId', keep_default_na=False)
	for row in df.itertuples():
		if row[0].lower() != lastFname:
			lastFname = row[0].lower()
			if len(row[1].strip()) > 0:
				imageLabels[lastFname] = "BOAT"
			else:
				imageLabels[lastFname] = "NO BOAT"

	print( "# of files with labels:", len(imageLabels))


def ensureDirExists(targetDir) :
    # Ensure that given directory exists. (Create it if it doesn't.)

    if not ( os.path.isdir(targetDir)):
        os.makedirs(targetDir)
    return


def keypress(event):
	# Captures user's keystroke while image is displayed, stores in lastkey global.

	#print('keypress', event.key)
	sys.stdout.flush()
	global lastkey
	lastkey = event.key
	print('lastkey', lastkey)
	plt.close()


def show_next_image(origImgName, boatOnly):
	# Show given image, let user indicate what to do with it.
	# Receives full path and name of an image file. Second argument
	# tells us whether to show only images that contain at least one boat, or 
	# to show all images.

	file_name = os.path.basename(origImgName)

	# If doing just boats, clear last action, unless going back. If don't clear
	# last action, could end up copying all the images that don't want to even display.
	# If backing up, want to continue backing up to previous image containing a boat.
	global lastkey
	if not (lastkey == 'left'):
		lastkey = ''

	if (not boatOnly) or (imageLabels[file_name.lower()] == "BOAT"):
		img = mpimg.imread(origImgName)
	
		# Title is label + filename.
		if file_name.lower() in imageLabels:
			title = imageLabels[file_name.lower()]
		else:
			title = "UNKNOWN"
			print ("*** Found file without a label: ", file_name)
		title += " " + os.path.basename(origImgName)

		fig = plt.figure(figsize=[8,8])
		fig.canvas.mpl_connect('key_press_event', keypress)
		fig.add_subplot(1, 2, 1, title=title)
		plt.imshow(img)
		plt.show()


def review_images(dir):
	# Let user go through images in given directory, using arrow keys to move forward and back.
	# 'a' copies image to a directory, 'b' to b. 'q' or escape quits.	
        # Any other key will move to the next image.
	# User can specify file number to start at.
	# Each displayed image includes filename and label.

	ensureDirExists(aDir)
	ensureDirExists(bDir)

	createImageLabels()

	srcFiles = [name for name in os.listdir(dir)
		if os.path.isfile(os.path.join(dir, name))]
	num_files = len(srcFiles)
	numA = 0
	numB = 0

	startNum = input( "Image number to start at?" )
	if int(startNum) >= 0:
		startNum = int(startNum) - 1
	else:
		startNum = 0
	n = 0

	boatOnly = yes_or_no("Show only images containing boats?")

	# Using while loop so can move forward and backward in list of files.
	while n < len(srcFiles):
		tmpFile = srcFiles[n]
		if n >= startNum:
			srcFile = os.path.join(dir, tmpFile)
			print( "file ", n, " of ", num_files, ", ", numA, " dir a, ", numB, "dir b, fileName: ", srcFile)
			show_next_image(srcFile, boatOnly)
			print( "lastkey=", lastkey)
			if lastkey == 'a':		# Copy file to first directory.
				shutil.copy2(srcFile, aDir)
				numA += 1
			elif lastkey == 'b':		# Copy file to second directory.
				shutil.copy2(srcFile, bDir)
				numB += 1
			elif lastkey == 'left':		# Show previous image again.
				if n > 0:
					n -= 2
				else:
					n -= 1		# If on first image, decrement by one here, will increment below, leaving us unmoved.
			elif lastkey == 'q' or lastkey == 'escape':		# Exit loop.
				break

		n += 1
		#if n > 5: break	# Used for testing, so don't have to go through all images.
		

if __name__ == "__main__":
	review_images( srcDir )



