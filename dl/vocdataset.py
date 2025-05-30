import os
import random
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from functools import partial
import pathlib
import numpy as np
from functools import partial
from PIL import Image
from misc.log import log
from misc import img, xml, misc
from misc.img import rescale2Target, hsvAdjust
from misc.bbox import rescaleBoxes
from dl.aug import DataAugmentationProcessor, ImageTransformedInfo


class VocDataset(Dataset):
    @staticmethod
    def collate(batch):
        """
        Used by PyTorch DataLoader class (collate_fn)
        """
        images  = []
        labels  = []
        tinfos = []
        rawImages = []

        for i, data in enumerate(batch):
            img = data[0]
            label = data[1]
            images.append(img)
            label[:, 0] = i # enrich image index in batch
            labels.append(label)
            if len(data) > 3:
                tinfo = data[2]
                tinfos.append(tinfo)
                rawImage = data[3]
                rawImages.append(rawImage)

        # The image data returned from __getitem__ is now a numpy array AFTER augmentation
        # The labels are also numpy arrays
        # We need to handle the case where boxList_cropped_np is empty (no objects after crop/filtering)
        if all(label.shape[0] == 0 for label in labels):
             # If all batches are empty after augmentation, return empty tensors of correct shape
             # Determine shape from expected postprocess output
             # Note: If using mixup, the image shape might change. Assuming standard inputShape for dummy tensor.
             dummy_image_shape = (3, *images[0].shape[-2:]) if len(images) > 0 and isinstance(images[0], np.ndarray) and images[0].ndim == 3 else (3, 640, 640) # Assuming common input shape and CHW format after postprocess
             dummy_label_shape = (0, 6) # Empty label tensor shape
             return torch.empty((len(batch), *dummy_image_shape), dtype=torch.float32), torch.empty(dummy_label_shape, dtype=torch.float32)

        images  = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
        labels  = torch.from_numpy(np.concatenate(labels, 0)).type(torch.FloatTensor)

        if len(rawImages) > 0:
            return images, labels, tinfos, rawImages

        return images, labels

    @staticmethod
    def workerInit(seed, workerId):
        workerSeed = workerId + seed
        random.seed(workerSeed)
        np.random.seed(workerSeed)
        torch.manual_seed(workerSeed)

    @staticmethod
    def getDataLoader(mcfg, splitName, isTest, fullInfo, selectedClasses=None):
        if splitName not in mcfg.subsetMap:
            raise ValueError("Split not found in mcfg: {}".format(splitName))

        dataset = VocDataset(
            imageDir=mcfg.imageDir,
            annotationDir=mcfg.annotationDir,
            classList=mcfg.classList,
            inputShape=mcfg.inputShape,
            subset=mcfg.subsetMap[splitName],
            isTest=isTest,
            fullInfo=fullInfo,
            suffix=mcfg.suffix,
            splitName=splitName,
            selectedClasses=selectedClasses,
            mcfg=mcfg,
        )
        return DataLoader(
            dataset,
            shuffle=True,
            batch_size=mcfg.batchSize,
            num_workers=mcfg.dcore,
            pin_memory=True,
            drop_last=False,
            sampler=None,
            collate_fn=VocDataset.collate,
            worker_init_fn=partial(VocDataset.workerInit, mcfg.seed)
        )

    def __init__(self, imageDir, annotationDir, classList, inputShape, subset, isTest, fullInfo, suffix, splitName, selectedClasses, mcfg):
        super(VocDataset, self).__init__()
        self.imageDir = imageDir
        self.annotationDir = annotationDir
        self.classList = classList
        self.inputShape = inputShape
        self.mcfg = mcfg
        
        aug_params = getattr(mcfg, 'augmentation', {})
        self.augp = DataAugmentationProcessor(
            inputShape=inputShape,
            jitter=aug_params.get('jitter', 0.3),
            rescalef=aug_params.get('rescalef', (0.25, 2)),
            flipProb=aug_params.get('flipProb', 0.5),
            huef=aug_params.get('huef', 0.1),
            satf=aug_params.get('satf', 0.7),
            valf=aug_params.get('valf', 0.4),
            eraseProb=aug_params.get('eraseProb', 0.0),
            eraseArea=aug_params.get('eraseArea', (0.02, 0.4)),
            use_flip=aug_params.get('use_flip', True),
            use_hsv=aug_params.get('use_hsv', True),
            use_erase=aug_params.get('use_erase', False),
            use_crop=aug_params.get('use_crop', False),
            cropArea=aug_params.get('cropArea', (0.5, 1.0)),
            use_mixup=aug_params.get('use_mixup', False),
            mixupProb=aug_params.get('mixupProb', 0.0),
            mixupAlpha=aug_params.get('mixupAlpha', 1.5)
        )
        
        self.isTest = isTest
        self.fullInfo = fullInfo
        self.suffix = suffix
        self.splitName = splitName
        self.selectedClasses = selectedClasses

        if subset is None:
            self.imageFiles = [os.path.join(imageDir, x) for x in os.listdir(imageDir) if pathlib.Path(x).suffix == self.suffix]
        else:
            self.imageFiles = [os.path.join(imageDir, x) for x in subset]
            for imFile in self.imageFiles:
                if not os.path.exists(imFile):
                    raise ValueError("Image file in subset not exists: {}".format(imFile))
        if len(self.imageFiles) == 0:
            raise ValueError("Empty image directory: {}".format(imageDir))

        self.annotationFiles = [os.path.join(annotationDir, "{}.xml".format(pathlib.Path(x).stem)) for x in self.imageFiles]
        for annFile in self.annotationFiles:
            if not os.path.exists(annFile):
                raise ValueError("Annotation file not exists: {}".format(annFile))

        log.inf("VOC dataset [{}] initialized from {} with {} images".format(self.splitName, imageDir, len(self.imageFiles)))
        if self.selectedClasses is not None:
            log.inf("VOC dataset [{}] set with selected classes: {}".format(self.splitName, self.selectedClasses))

    def postprocess(self, imageData, boxList):
        # imageData is now a numpy array (HWC, uint8) after augmentation
        # boxList is now a numpy array ([x1, y1, x2, y2, class_id]) after augmentation

        # Handle case where boxList is empty after augmentation (e.g., no objects in cropped region or after mixup)
        if boxList.shape[0] == 0:
             # Return empty tensors of correct shape
             # Image data needs to be normalized and transposed
             processed_imageData = np.transpose(imageData.astype(np.float32), (2, 0, 1)) / 255.0
             labels = np.zeros((0, 6), dtype=np.float32) # Empty labels tensor
             return processed_imageData, labels


        # Original postprocess logic (assuming boxList is not empty)
        imageData = imageData.astype(np.float32) / 255.0 # Normalize image data
        imageData = np.transpose(imageData, (2, 0, 1)) # Transpose HWC to CHW
        
        # boxList is already np.float32 from randomCrop/randomMixup, no need to convert again
        # boxList = np.array(boxList, dtype=np.float32)
        
        labels = np.zeros((boxList.shape[0], 6), dtype=np.float32) # add one dim (5 + 1 = 6) as image batch index (VocDataset.collate)
        
        # Normalize box coordinates to 0-1 range based on current image shape
        img_h, img_w = imageData.shape[-2:] # Get H, W from CHW format
        labels[:, 1] = boxList[:, -1]
        labels[:, 2] = boxList[:, 0] / img_w # x1
        labels[:, 3] = boxList[:, 1] / img_h # y1
        labels[:, 4] = boxList[:, 2] / img_w # x2
        labels[:, 5] = boxList[:, 3] / img_h # y2
        
        return imageData, labels

    def __len__(self):
        return len(self.imageFiles)

    def __getitem__(self, index):
        ii = index % len(self.imageFiles)
        imgFile = self.imageFiles[ii]
        image = img.loadRGBImage(imgFile)
        annFile = self.annotationFiles[ii]
        boxList = xml.XmlBbox.loadXmlObjectList(annFile, self.classList, selectedClasses=self.selectedClasses, asArray=True)

        if self.isTest:
            imageData, boxList, tinfo = self.augp.processSimple(image, boxList)
        else:
            # Apply Mixup first if enabled and lucky
            if self.augp.use_mixup and random.random() < self.augp.mixupProb:
                # Get a random image and its labels from the dataset
                jj = random.randint(0, len(self.imageFiles) - 1)
                imgFile2 = self.imageFiles[jj]
                image2 = img.loadRGBImage(imgFile2)
                annFile2 = self.annotationFiles[jj]
                boxList2 = xml.XmlBbox.loadXmlObjectList(annFile2, self.classList, selectedClasses=self.selectedClasses, asArray=True)

                # Process both images with base augmentations (rescale/jitter/flip) separately
                # Note: Mixup should typically happen AFTER individual image augmentations
                # Let's apply the initial processEnhancement steps (rescale/jitter/flip) to both images first

                # Apply initial enhancement steps to the first image
                boxList_np1 = np.array(boxList) # Convert original boxList to numpy
                targetHeight, targetWidth = self.inputShape
                oriWidth1, oriHeight1 = image.size
                scalef1 = misc.randAB(self.augp.rescalef[0], self.augp.rescalef[1])
                oriAspectRatio1 = oriWidth1 / oriHeight1
                newAspectRatio1 = oriAspectRatio1 * misc.randAB(1 - self.augp.jitter, 1 + self.augp.jitter) / misc.randAB(1 - self.augp.jitter, 1 + self.augp.jitter)
                if newAspectRatio1 < 1:
                    scaledHeight1 = int(scalef1 * targetHeight)
                    scaledWidth1 = int(newAspectRatio1 * scaledHeight1)
                else:
                    scaledWidth1 = int(scalef1 * targetWidth)
                    scaledHeight1 = int(scaledWidth1 / newAspectRatio1)
                image_rescaled_pil1, xoffset1, yoffset1 = rescale2Target(image, scaledWidth1, scaledHeight1, targetWidth, targetHeight)

                flipFlag1 = False
                if self.augp.use_flip and misc.randAB(0, 1) < self.augp.flipProb:
                    image_rescaled_pil1 = image_rescaled_pil1.transpose(Image.FLIP_LEFT_RIGHT)
                    flipFlag1 = True
                boxList_adjusted_np1 = rescaleBoxes(boxList_np1, oriWidth1, oriHeight1, scaledWidth1, scaledHeight1, targetWidth, targetHeight, xoffset1, yoffset1, flipFlag1)
                boxList_adjusted_np1 = np.array(boxList_adjusted_np1)
                imageData_np1 = np.array(image_rescaled_pil1, np.uint8)

                # Apply initial enhancement steps to the second image
                boxList_np2 = np.array(boxList2)
                oriWidth2, oriHeight2 = image2.size
                scalef2 = misc.randAB(self.augp.rescalef[0], self.augp.rescalef[1])
                oriAspectRatio2 = oriWidth2 / oriHeight2
                newAspectRatio2 = oriAspectRatio2 * misc.randAB(1 - self.augp.jitter, 1 + self.augp.jitter) / misc.randAB(1 - self.augp.jitter, 1 + self.augp.jitter)
                if newAspectRatio2 < 1:
                    scaledHeight2 = int(scalef2 * targetHeight)
                    scaledWidth2 = int(newAspectRatio2 * scaledHeight2)
                else:
                    scaledWidth2 = int(scalef2 * targetWidth)
                    scaledHeight2 = int(scaledWidth2 / newAspectRatio2)
                image_rescaled_pil2, xoffset2, yoffset2 = rescale2Target(image2, scaledWidth2, scaledHeight2, targetWidth, targetHeight)

                flipFlag2 = False
                if self.augp.use_flip and misc.randAB(0, 1) < self.augp.flipProb:
                    image_rescaled_pil2 = image_rescaled_pil2.transpose(Image.FLIP_LEFT_RIGHT)
                    flipFlag2 = True
                boxList_adjusted_np2 = rescaleBoxes(boxList_np2, oriWidth2, oriHeight2, scaledWidth2, scaledHeight2, targetWidth, targetHeight, xoffset2, yoffset2, flipFlag2)
                boxList_adjusted_np2 = np.array(boxList_adjusted_np2)
                imageData_np2 = np.array(image_rescaled_pil2, np.uint8)

                # Now apply Mixup on the initially processed images and boxes
                imageData_np, boxList_np = self.augp.randomMixup(imageData_np1, boxList_adjusted_np1, imageData_np2, boxList_adjusted_np2)
                tinfo = None # Tinfo is complex for mixup, set to None for now

                # Continue with color adjustments and erasing on the mixed image
                if self.augp.use_hsv:
                    imageData_np = hsvAdjust(imageData_np, self.augp.huef, self.augp.satf, self.augp.valf)

                if self.augp.use_erase:
                    imageData_np = self.augp.randomErasing(imageData_np)

            else: # No Mixup
                # Original processEnhancement flow (without Mixup)
                boxList_np = np.array(boxList) # Convert original boxList to numpy

                targetHeight, targetWidth = self.inputShape
                oriWidth, oriHeight = image.size
                scalef = misc.randAB(self.augp.rescalef[0], self.augp.rescalef[1])
                oriAspectRatio = oriWidth / oriHeight
                newAspectRatio = oriAspectRatio * misc.randAB(1 - self.augp.jitter, 1 + self.augp.jitter) / misc.randAB(1 - self.augp.jitter, 1 + self.augp.jitter)
                if newAspectRatio < 1:
                    scaledHeight = int(scalef * targetHeight)
                    scaledWidth = int(newAspectRatio * scaledHeight)
                else:
                    scaledWidth = int(scalef * targetWidth)
                    scaledHeight = int(scaledWidth / newAspectRatio)
                image_rescaled_pil, xoffset, yoffset = rescale2Target(image, scaledWidth, scaledHeight, targetWidth, targetHeight)

                flipFlag = False
                if self.augp.use_flip and misc.randAB(0, 1) < self.augp.flipProb:
                    image_rescaled_pil = image_rescaled_pil.transpose(Image.FLIP_LEFT_RIGHT)
                    flipFlag = True

                boxList_adjusted_np = rescaleBoxes(boxList_np, oriWidth, oriHeight, scaledWidth, scaledHeight, targetWidth, targetHeight, xoffset, yoffset, flipFlag) # Adjust boxes based on initial transform
                boxList_adjusted_np = np.array(boxList_adjusted_np)

                tinfo = ImageTransformedInfo(oriWidth, oriHeight, scaledWidth, scaledHeight, targetWidth, targetHeight, xoffset, yoffset, flipFlag)

                imageData_np = np.array(image_rescaled_pil, np.uint8)

                # Apply Random Crop
                imageData_np, boxList_np, crop_bounds = self.augp.randomCrop(imageData_np, boxList_adjusted_np) # Passes scaled/flipped boxes

                # Apply HSV adjustment
                if self.augp.use_hsv:
                    imageData_np = hsvAdjust(imageData_np, self.augp.huef, self.augp.satf, self.augp.valf)

                # Apply Random Erasing
                if self.augp.use_erase:
                    imageData_np = self.augp.randomErasing(imageData_np)

            # postprocess expects numpy arrays
            imageData, labels = self.postprocess(imageData_np, boxList_np)
        
        if not self.fullInfo:
            return imageData, labels

        tinfo.imgFile = imgFile # Note: Tinfo might be None if mixup was applied
        return imageData, labels, tinfo, image
