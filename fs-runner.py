# split into train and test set
from os import listdir
from xml.etree import ElementTree
from numpy import zeros
from numpy import asarray
from mrcnn.utils import Dataset
from matplotlib import pyplot
from mrcnn.utils import extract_bboxes
from mrcnn.visualize import display_instances


# class that defines and loads the kangaroo dataset
class fireSmokeDataset(Dataset):
	# load the dataset definitions
	def load_dataset(self, dataset_dir, is_train=True):
		# define one class
		self.add_class("dataset", 1, "Fire")
		self.add_class("dataset", 2, "Smoke")
		# define data locations
		images_dir = dataset_dir + '/images/'
		annotations_dir = dataset_dir + '/annots/'
		# find all images
		for filename in listdir(images_dir):
			# extract image id
			image_id = filename[:-4]
			# skip bad images
			if image_id in ['00090']:
				continue
			# skip all images after 150 if we are building the train set
			if is_train and int(image_id) >= 150:
				continue
			# skip all images before 150 if we are building the test/val set
			if not is_train and int(image_id) < 150:
				continue
			img_path = images_dir + filename
			ann_path = annotations_dir + image_id + '.xml'
			# add to dataset
			self.add_image('dataset', image_id=image_id, path=img_path, annotation=ann_path)

	# extract bounding boxes from an annotation file
	def extract_boxes(self, filename):
		# load and parse the file
		tree = ElementTree.parse(filename)
		# get the root of the document
		root = tree.getroot()
		# extract each bounding box
		boxes = list()
		label = list()
		for box in root.findall('.//object'):
			xmin = int(box.find('bndbox').find('xmin').text)
			ymin = int(box.find('bndbox').find('ymin').text)
			xmax = int(box.find('bndbox').find('xmax').text)
			ymax = int(box.find('bndbox').find('ymax').text)
			coors = [xmin, ymin, xmax, ymax]
			boxes.append(coors)
			label.append(box.find('name').text)
		# extract image dimensions
		width = int(root.find('.//size/width').text)
		height = int(root.find('.//size/height').text)

		return boxes, width, height, label

	# load the masks for an image
	def load_mask(self, image_id):
		# get details of image
		info = self.image_info[image_id]
		# define box file location
		path = info['annotation']
		# load XML
		boxes, w, h, label = self.extract_boxes(path)
		# create one array for all masks, each on a different channel
		masks = zeros([h, w, len(boxes)], dtype='uint8')
		# create masks
		class_ids = list()
		for i in range(len(boxes)):
			box = boxes[i]
			row_s, row_e = box[1], box[3]
			col_s, col_e = box[0], box[2]
			masks[row_s:row_e, col_s:col_e, i] = 1
			class_ids.append(self.class_names.index(label[i]))

		return masks, asarray(class_ids, dtype='int32')

	# load an image reference
	def image_reference(self, image_id):
		info = self.image_info[image_id]
		return info['path']

# train set
train_set = fireSmokeDataset()
train_set.load_dataset('fireSmoke', is_train=True)
train_set.prepare()
print('Train: %d' % len(train_set.image_ids))

for image_id in range(10):
	# load the image
	image = train_set.load_image(image_id)
	# load the masks and the class ids
	mask, class_ids = train_set.load_mask(image_id)
	# extract bounding boxes from the masks
	bbox = extract_bboxes(mask)
	# display image with masks and bounding boxes
	display_instances(image, bbox, mask, class_ids, train_set.class_names)

