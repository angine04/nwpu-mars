import numpy as np
from PIL import Image
from misc import misc
from misc.img import rescale2Target, hsvAdjust
from misc.bbox import rescaleBoxes
import random # Add random import
import math # Add math import


class ImageTransformedInfo(object):
    def __init__(self, oriWidth, oriHeight, scaledWidth, scaledHeight, targetWidth, targetHeight, xoffset, yoffset, flip):
        self.oriWidth = oriWidth
        self.oriHeight = oriHeight
        self.scaledWidth = scaledWidth
        self.scaledHeight = scaledHeight
        self.targetWidth = targetWidth
        self.targetHeight = targetHeight
        self.xoffset = xoffset
        self.yoffset = yoffset
        self.flip = flip
        self.imgFile = None


class DataAugmentationProcessor(object):
    def __init__(self, inputShape, jitter=0.3, rescalef=(0.25, 2), flipProb=0.5, huef=0.1, satf=0.7, valf=0.4, eraseProb=0.0, eraseArea=(0.02, 0.4), use_flip=True, use_hsv=True, use_erase=False, use_crop=False, cropArea=(0.5, 1.0), use_mixup=False, mixupProb=0.0, mixupAlpha=1.5):
        self.inputShape = inputShape
        self.jitter = jitter
        self.rescalef = rescalef
        self.flipProb = flipProb
        self.huef = huef
        self.satf = satf
        self.valf = valf
        self.eraseProb = eraseProb
        self.eraseArea = eraseArea # (min_area_ratio, max_area_ratio) of erased region
        self.cropArea = cropArea # (min_area_ratio, max_area_ratio) of cropped region
        self.mixupProb = mixupProb
        self.mixupAlpha = mixupAlpha # Alpha parameter for Beta distribution

        self.use_flip = use_flip
        self.use_hsv = use_hsv
        self.use_erase = use_erase
        self.use_crop = use_crop
        self.use_mixup = use_mixup

    def randomErasing(self, image_np):
        """Apply random erasing to the image (numpy array) """
        if not self.use_erase or np.random.random() >= self.eraseProb:
            return image_np

        img_h, img_w, _ = image_np.shape
        img_area = img_h * img_w

        # Randomly select erasing area
        erase_area = misc.randAB(self.eraseArea[0], self.eraseArea[1]) * img_area
        aspect_ratio = misc.randAB(0.3, 3.3) # Random aspect ratio

        erase_h = int(np.sqrt(erase_area / aspect_ratio))
        erase_w = int(np.sqrt(erase_area * aspect_ratio))

        # Ensure erase region is within image bounds
        if erase_h >= img_h or erase_w >= img_w:
            return image_np # Cannot erase region larger than image

        # Randomly select top-left corner of erasing region
        x1 = np.random.randint(0, img_w - erase_w)
        y1 = np.random.randint(0, img_h - erase_h)
        x2 = x1 + erase_w
        y2 = y1 + erase_h

        # Erase the region (fill with random value or mean)
        image_np[y1:y2, x1:x2] = np.random.randint(0, 256, size=(erase_h, erase_w, 3), dtype=np.uint8)

        return image_np

    def randomCrop(self, image_np, boxList_np):
        """Apply random crop to the image and bounding boxes (numpy arrays) """
        if not self.use_crop:
            return image_np, boxList_np, (0, 0, image_np.shape[1], image_np.shape[0]) # Return original image, boxes and full image bounds

        img_h, img_w, _ = image_np.shape
        img_area = img_h * img_w

        # Randomly select crop area
        for attempt in range(10):
            crop_area = misc.randAB(self.cropArea[0], self.cropArea[1]) * img_area
            aspect_ratio = misc.randAB(0.5, 2.0) # Random aspect ratio for crop area

            crop_h = int(np.sqrt(crop_area / aspect_ratio))
            crop_w = int(np.sqrt(crop_area * aspect_ratio))

            # Ensure crop region is within image bounds
            if crop_h > img_h or crop_w > img_w:
                continue # Try again

            # Randomly select top-left corner of crop region
            x1 = np.random.randint(0, img_w - crop_w)
            y1 = np.random.randint(0, img_h - crop_h)
            x2 = x1 + crop_w
            y2 = y1 + crop_h

            crop_box = np.array([x1, y1, x2, y2])

            # Filter bounding boxes that are at least partially inside the crop area
            # Box format: [x1, y1, x2, y2, class_id]
            overlap_boxes_mask = (boxList_np[:, 0] < crop_box[2]) & \
                               (boxList_np[:, 2] > crop_box[0]) & \
                               (boxList_np[:, 1] < crop_box[3]) & \
                               (boxList_np[:, 3] > crop_box[1])

            valid_boxes = boxList_np[overlap_boxes_mask].copy()

            if len(valid_boxes) > 0:
                # Adjust box coordinates relative to the cropped region
                valid_boxes[:, 0:4] = valid_boxes[:, 0:4] - np.tile(crop_box[0:4], (valid_boxes.shape[0], 1))

                # Clip boxes to the bounds of the cropped region
                valid_boxes[:, 0:4] = np.clip(valid_boxes[:, 0:4], 0, [crop_w, crop_h, crop_w, crop_h])

                cropped_image = image_np[y1:y2, x1:x2]
                return cropped_image, valid_boxes, (x1, y1, x2, y2)

        # If no valid crop found after attempts, return original image and boxes
        return image_np, boxList_np, (0, 0, img_w, img_h)

    def randomMixup(self, image1_np, boxList1_np, image2_np, boxList2_np):
        """Apply Mixup augmentation"""
        # Ensure images are numpy arrays with float type for calculations
        image1_np = image1_np.astype(np.float32)
        image2_np = image2_np.astype(np.float32)

        # Sample lambda from Beta distribution
        if self.mixupAlpha > 0:
            lam = np.random.beta(self.mixupAlpha, self.mixupAlpha)
        else:
            lam = 1.0

        # Mix images
        mixed_image_np = lam * image1_np + (1 - lam) * image2_np
        mixed_image_np = mixed_image_np.astype(np.uint8) # Convert back to uint8

        # Combine bounding boxes - just concatenate them.
        # Adjusting labels based on lambda is more complex and often not done directly for box coordinates.
        # We assume the labels are simply combined.
        combined_boxList_np = np.concatenate((boxList1_np, boxList2_np), axis=0)

        # Note: Class labels are already included in boxList_np [x1, y1, x2, y2, class_id]
        # If there were confidence scores, we might adjust them here based on lambda.

        return mixed_image_np, combined_boxList_np

    def processSimple(self, image, boxList):
        # rescale image
        targetHeight, targetWidth = self.inputShape
        oriWidth, oriHeight = image.size
        scaleFactor = min(targetWidth / oriWidth, targetHeight / oriHeight)
        scaledWidth = int(oriWidth * scaleFactor)
        scaledHeight = int(oriHeight * scaleFactor)
        newImage, xoffset, yoffset = rescale2Target(image, scaledWidth, scaledHeight, targetWidth, targetHeight)
        imageData = np.array(newImage, np.float32)
        # rescale boxes accordingly
        boxList = rescaleBoxes(boxList, oriWidth, oriHeight, scaledWidth, scaledHeight, targetWidth, targetHeight, xoffset, yoffset, False)
        transformInfo = ImageTransformedInfo(oriWidth, oriHeight, scaledWidth, scaledHeight, targetWidth, targetHeight, xoffset, yoffset, False)
        return imageData, boxList, transformInfo

    def processEnhancement(self, image, boxList):
        # Initial rescale/jitter/flip and box adjustment
        # BoxList needs to be converted to numpy array at this stage for geometric transforms
        boxList_np = np.array(boxList) # Convert original boxList to numpy

        targetHeight, targetWidth = self.inputShape
        oriWidth, oriHeight = image.size
        scalef = misc.randAB(self.rescalef[0], self.rescalef[1])
        oriAspectRatio = oriWidth / oriHeight
        newAspectRatio = oriAspectRatio * misc.randAB(1 - self.jitter, 1 + self.jitter) / misc.randAB(1 - self.jitter, 1 + self.jitter)
        if newAspectRatio < 1:
            scaledHeight = int(scalef * targetHeight)
            scaledWidth = int(newAspectRatio * scaledHeight)
        else:
            scaledWidth = int(scalef * targetWidth)
            scaledHeight = int(scaledWidth / newAspectRatio)
        image_rescaled_pil, xoffset, yoffset = rescale2Target(image, scaledWidth, scaledHeight, targetWidth, targetHeight)

        # Apply Flip
        flipFlag = False
        if self.use_flip and misc.randAB(0, 1) < self.flipProb:
            image_rescaled_pil = image_rescaled_pil.transpose(Image.FLIP_LEFT_RIGHT)
            flipFlag = True

        # Adjust boxes after initial rescale/flip (before crop and color adjustments)
        boxList_adjusted_np = rescaleBoxes(boxList_np, oriWidth, oriHeight, scaledWidth, scaledHeight, targetWidth, targetHeight, xoffset, yoffset, flipFlag) # Adjust boxes based on initial transform
        # Note: rescaleBoxes returns list of lists, convert back to numpy
        boxList_adjusted_np = np.array(boxList_adjusted_np)

        tinfo = ImageTransformedInfo(oriWidth, oriHeight, scaledWidth, scaledHeight, targetWidth, targetHeight, xoffset, yoffset, flipFlag)

        # Convert to numpy for Crop, HSV, Erase
        imageData_np = np.array(image_rescaled_pil, np.uint8)

        # Apply Random Crop (expects numpy array and boxes already adjusted by initial transform)
        # randomCrop returns cropped image (numpy) and boxes relative to cropped image
        imageData_np, boxList_cropped_np, crop_bounds = self.randomCrop(imageData_np, boxList_adjusted_np)
        # After crop, boxList_cropped_np contains boxes relative to the cropped image dimensions

        # Apply HSV adjustment
        if self.use_hsv:
            imageData_np = hsvAdjust(imageData_np, self.huef, self.satf, self.valf)

        # Apply Random Erasing
        if self.use_erase:
            imageData_np = self.randomErasing(imageData_np)

        # The tinfo object currently only reflects the initial rescale/flip. It does not account for crop.
        # This might be acceptable depending on how tinfo is used downstream (e.g., for visualization).
        # If precise inverse transformation is needed, tinfo structure would need update.

        # Return numpy array for image data and boxes (boxes are relative to the final image size after crop)
        return imageData_np, boxList_cropped_np, tinfo
