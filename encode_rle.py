# Hacking together an rle encoder. Code will be moved to a jupyter notebook once have it tested.
# May 6, 2019.
# Receives: Segmented mask, 0 = background, 1 = object 1, 2 = object 2, ...
# Returns: List of strings, one for each object in mask, containing rle pixels for given object, pixels
#	numbered starting at top left at 0, going down, then across.

import numpy as np
import unittest

def encode_rle(origMask):
	encoded_strings = []
	mask_shape = origMask.shape
	cols = mask_shape[0]
	rows = mask_shape[1]
	
	col = 0
	while col < cols:
		row = 0
		while row < rows:
			if origMask[row, col] > 0:
				objectNum = origMask[row, col]
				startPixel = row + (col * rows)
				numPixels = 1
				row += 1
				while row < rows:
					if origMask[row, col] == objectNum:
						numPixels += 1
						row += 1
					else:
						# We've gone past the object, decrement row here because incremented again below.
						row -= 1
						break
					#print("w: ", row)
				newEncoding = str(startPixel) + " " + str(numPixels)
				if len(encoded_strings) >= objectNum:
					if len(encoded_strings[objectNum - 1] ) > 0:
						# Add space if entry already contains data. (Sometimes is empty because was created to add a higher-numbered object.)
						newEncoding = " " + newEncoding
					encoded_strings[objectNum-1] = encoded_strings[objectNum-1] + newEncoding
				else:
					while len(encoded_strings) < objectNum:
						encoded_strings.append("")
					encoded_strings[objectNum-1] = newEncoding.strip()
			#print( "f: ", row)
			row += 1
		col += 1

	return encoded_strings

				

class rleTestCases_singleobj(unittest.TestCase):
	"""Tests for 'encode_rle', single object"""

	def test_firstPixel(self):
		testArr = np.array([	[1, 0, 0],
					[0, 0, 0],
					[0, 0, 0] ])
		rles = encode_rle(testArr)
		self.assertEqual(len(rles), 1)
		self.assertEqual(rles[0], "0 1")

	def test_lastPixel(self):
		testArr = np.array([	[0, 0, 0],
					[0, 0, 0],
					[0, 0, 1] ])
		rles = encode_rle(testArr)
		self.assertEqual(len(rles), 1)
		self.assertEqual(rles[0], "8 1")

	def test_noPixels(self):
		testArr = np.array([	[0, 0, 0],
					[0, 0, 0],
					[0, 0, 0] ])
		rles = encode_rle(testArr)
		self.assertFalse(rles)


	def test_colMiddle(self):
		testArr = np.array([	[0, 0, 0],
					[1, 0, 0],
					[0, 0, 0] ])
		rles = encode_rle(testArr)
		self.assertEqual(len(rles), 1)
		self.assertEqual(rles[0], "1 1")

	def test_rowMiddle(self):
		testArr = np.array([	[0, 1, 0],
					[0, 0, 0],
					[0, 0, 0] ])
		rles = encode_rle(testArr)
		self.assertEqual(len(rles), 1)
		self.assertEqual(rles[0], "3 1")

	def test_colEnd(self):
		testArr = np.array([	[0, 0, 0],
					[0, 0, 0],
					[0, 1, 0] ])
		rles = encode_rle(testArr)
		self.assertEqual(len(rles), 1)
		self.assertEqual(rles[0], "5 1")

	def test_rowEnd(self):
		testArr = np.array([	[0, 0, 0],
					[0, 0, 1],
					[0, 0, 0] ])
		rles = encode_rle(testArr)
		self.assertEqual(len(rles), 1)
		self.assertEqual(rles[0], "7 1")

	def test_2colpix(self):
		testArr = np.array([	[1, 0, 0],
					[1, 0, 0],
					[0, 0, 0] ])
		rles = encode_rle(testArr)
		self.assertEqual(len(rles), 1)
		self.assertEqual(rles[0], "0 2")

	def test_2rowpix(self):
		testArr = np.array([	[1, 1, 0],
					[0, 0, 0],
					[0, 0, 0] ])
		rles = encode_rle(testArr)
		self.assertEqual(len(rles), 1)
		self.assertEqual(rles[0], "0 1 3 1")


class rleTestCases_multipleobj(unittest.TestCase):
	"""Tests for 'encode_rle', multiple objects"""

	def test_firstCols(self):
		testArr = np.array([	[1, 2, 0],
					[0, 0, 0],
					[0, 0, 0] ])
		rles = encode_rle(testArr)
		self.assertEqual(len(rles), 2)
		self.assertEqual(rles[0], "0 1")
		self.assertEqual(rles[1], "3 1")

	def test_firstRows(self):
		testArr = np.array([	[1, 0, 0],
					[2, 0, 0],
					[0, 0, 0] ])
		rles = encode_rle(testArr)
		self.assertEqual(len(rles), 2)
		self.assertEqual(rles[0], "0 1")
		self.assertEqual(rles[1], "1 1")

	def test_firstCol(self):
		testArr = np.array([	[1, 0, 0],
					[2, 0, 0],
					[2, 0, 0] ])
		rles = encode_rle(testArr)
		self.assertEqual(len(rles), 2)
		self.assertEqual(rles[0], "0 1")
		self.assertEqual(rles[1], "1 2")

	def test_firstCol2(self):
		testArr = np.array([	[1, 0, 0],
					[1, 0, 0],
					[2, 0, 0] ])
		rles = encode_rle(testArr)
		self.assertEqual(len(rles), 2)
		self.assertEqual(rles[0], "0 2")
		self.assertEqual(rles[1], "2 1")

	def test_noncont(self):
		testArr = np.array([	[1, 1, 0],
					[1, 0, 2],
					[2, 0, 2] ])
		rles = encode_rle(testArr)
		self.assertEqual(len(rles), 2)
		self.assertEqual(rles[0], "0 2 3 1")
		self.assertEqual(rles[1], "2 1 7 2")

	def test_reverseOrder(self):
		testArr = np.array([	[2, 2, 0],
					[2, 0, 1],
					[1, 0, 1] ])
		rles = encode_rle(testArr)
		self.assertEqual(len(rles), 2)
		self.assertEqual(rles[0], "2 1 7 2")
		self.assertEqual(rles[1], "0 2 3 1")

	def test_reverseOrdergap(self):
		testArr = np.array([	[2, 2, 0],
					[0, 0, 1],
					[1, 0, 1] ])
		rles = encode_rle(testArr)
		self.assertEqual(len(rles), 2)
		self.assertEqual(rles[0], "2 1 7 2")
		self.assertEqual(rles[1], "0 1 3 1")

	def test_3obj(self):
		testArr = np.array([	[1, 2, 0],
					[1, 0, 0],
					[0, 0, 3] ])
		rles = encode_rle(testArr)
		self.assertEqual(len(rles), 3)
		self.assertEqual(rles[0], "0 2")
		self.assertEqual(rles[1], "3 1")
		self.assertEqual(rles[2], "8 1")


unittest.main()

