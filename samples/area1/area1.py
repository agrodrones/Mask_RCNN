"""
Mask R-CNN
Configurations and data loading code for Area1.

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 area1.py train --dataset=/path/to/area1/ --model=coco

    # Train a new model starting from ImageNet weights.
    python3 area1.py train --dataset=/path/to/area1/ --model=imagenet

    # Continue training a model that you had trained earlier
    python3 area1.py train --dataset=/path/to/area1/ --model=/path/to/weights.h5

    # Continue training the last model you trained
    python3 area1.py train --dataset=/path/to/area1/ --model=last
"""

import os
import sys
import time
import numpy as np
import imgaug
import json
import cv2
import skimage

import zipfile
import urllib.request
import shutil

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Configurations
############################################################

class Area1Config(Config):
    """Configuration for training on Area 1
    Derives from the base Config class and overrides values specific
    to the Area 1 dataset.
    """
    NAME = "area1"

    IMAGES_PER_GPU = 2

    IMAGE_MAX_DIM = 1024
    
    NUM_CLASSES = 3  # BG + buva + capim

############################################################
#  Dataset
############################################################

class Area1Dataset(utils.Dataset):
    
    def load_labelme(self, dataset_dir, subset):
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)
        
        jsonfiles, annotations = [], []
        
        filenames = os.listdir(dataset_dir)
        
        for filename in filenames:
            if filename.endswith(".json"):
                jsonfiles.append(filename)
                annotation = json.load(open(os.path.join(dataset_dir,filename)))
                
                imagename = annotation['imagePath']
                if not os.path.isfile(os.path.join(dataset_dir, imagename)):
                    continue
                if len(annotation["shapes"]) == 0:
                    continue
                
                annotations.append(annotation)
                
        print("In {source} {subset} dataset we have {number:d} annotation files."
            .format(source=Area1Config.NAME, subset=subset,number=len(jsonfiles)))
        print("In {source} {subset} dataset we have {number:d} valid annotations."
            .format(source=Area1Config.NAME, subset=subset,number=len(annotations)))
        
        labelslist = []
        
        for annotation in annotations:
            shapes, classids = [], []
 
            for shape in annotation["shapes"]:
                label = shape["label"]
                if labelslist.count(label) == 0:
                    labelslist.append(label)
                classids.append(labelslist.index(label) + 1)
                shapes.append(shape["points"])
            
            width = annotation["imageWidth"]
            height = annotation["imageHeight"]
            
            self.add_image(
                Area1Config.NAME,
                image_id=annotation["imagePath"],  # use file name as a unique image id
                path=os.path.join(dataset_dir,annotation["imagePath"]),
                width=width, height=height,
                shapes=shapes, classids=classids)
 
        print("In {source} {subset} dataset we have {number:d} class item"
            .format(source=Area1Config.NAME, subset=subset,number=len(labelslist)))
 
        for labelid, labelname in enumerate(labelslist):
            self.add_class(Area1Config.NAME, labelid, labelname)

    def load_mask(self,image_id):
        """
        Generate instance masks for an image.
        Returns:
        masks: A bool array of shape [height, width, instance count] with one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        image_info = self.image_info[image_id]
        
        if image_info["source"] != Area1Config.NAME:
            return super(self.__class__, self).load_mask(image_id)
 
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["shapes"])], dtype=np.uint8)
        
        for idx, points in enumerate(info["shapes"]):
            pointsy, pointsx = zip(*points)
            rr, cc = skimage.draw.polygon(pointsx, pointsy)
            mask[rr, cc, idx] = 1
            
        masks_np = mask.astype(np.bool)
        classids_np = np.array(image_info["classids"]).astype(np.int32)
        
        return masks_np, classids_np
    
    def load_image(self, image_id):
        image = cv2.imread(self.image_info[image_id]['path'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if image.ndim != 3:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        if image.shape[-1] == 4:
            image = image[..., :3]
            
        return image
 
    def image_reference(self,image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == Area1Config.NAME:
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN on Area 1.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' on Area 1")
    parser.add_argument('--dataset', required=True,
                        metavar="/path/to/area1/",
                        help='Directory of the Area 1 dataset')
    parser.add_argument('--model', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    
    args = parser.parse_args()
    print("Command: ", args.command)
    print("Model: ", args.model)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    if args.command == "train":
        config = Area1Config()
    else:
        class InferenceConfig(Area1Config):
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            DETECTION_MIN_CONFIDENCE = 0
        config = InferenceConfig()
    config.display()

    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    if args.model.lower() == "coco":
        model_path = COCO_MODEL_PATH
    elif args.model.lower() == "last":
        model_path = model.find_last()
    elif args.model.lower() == "imagenet":
        model_path = model.get_imagenet_weights()
    else:
        model_path = args.model

    print("Loading weights ", model_path)
    model.load_weights(model_path, by_name=True)

    if args.command == "train":
        dataset_train = Area1Dataset()
        dataset_train.load_labelme(args.dataset, "train")
        dataset_train.prepare()

        dataset_val = Area1Dataset()
        dataset_val.load_labelme(args.dataset, "val")
        dataset_val.prepare()

        augmentation = imgaug.augmenters.Fliplr(0.5)

        print("Training network heads")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=40,
                    layers='heads',
                    augmentation=augmentation)
        
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'evaluate'".format(args.command))
