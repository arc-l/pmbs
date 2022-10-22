import math
import os
import glob

import torch
import numpy as np
from PIL import Image as im
import torch.nn.functional as F
import torchvision.transforms as TT
from torchvision.transforms import transforms
from torchvision.transforms.functional import InterpolationMode

import utils
from constants import (
    IMAGE_OBJ_CROP_SIZE,
    IMAGE_SIZE,
    WORKSPACE_LIMITS,
    PIXEL_SIZE,
    PUSH_Q,
    GRASP_Q,
    COLOR_MEAN,
    COLOR_STD,
    DEPTH_MEAN,
    DEPTH_STD,
    BINARY_IMAGE_MEAN,
    BINARY_IMAGE_STD,
    BINARY_OBJ_MEAN,
    BINARY_OBJ_STD,
    DEPTH_MIN,
    PUSH_DISTANCE,
    GRIPPER_PUSH_RADIUS_PIXEL,
    GRIPPER_PUSH_RADIUS_SAFE_PIXEL,
    GRIPPER_GRASP_OUTER_DISTANCE_PIXEL,
    GRIPPER_GRASP_INNER_DISTANCE_PIXEL,
    IMAGE_PAD_WIDTH,
    PUSH_DISTANCE_PIXEL,
    GRIPPER_GRASP_WIDTH_PIXEL,
    NUM_ROTATION,
    IMAGE_PAD_DIFF,
    TARGET_LOWER,
    TARGET_UPPER,
)


class LifelongEvalDataset(torch.utils.data.Dataset):
    """For lifelong learning"""

    def __init__(self, actions, mask_image):
        # focus on target, so make one extra channel
        target_mask_img = np.zeros_like(mask_image, dtype=np.uint8)
        target_mask_img[mask_image == 255] = 255
        mask_heightmap = np.dstack((target_mask_img, mask_image))
        mask_heightmap_pad = np.pad(
            mask_heightmap,
            ((IMAGE_PAD_WIDTH, IMAGE_PAD_WIDTH), (IMAGE_PAD_WIDTH, IMAGE_PAD_WIDTH), (0, 0)),
            "constant",
            constant_values=0,
        )

        self.mask_heightmap_pad = mask_heightmap_pad
        self.actions = actions

    def __getitem__(self, idx):
        action = self.actions[idx]
        action_start = (action[0][1], action[0][0])
        action_end = (action[1][1], action[1][0])
        current = (
            action_end[0] - action_start[0],
            action_end[1] - action_start[1],
        )
        right = (1, 0)
        dot = right[0] * current[0] + right[1] * current[1]  # dot product between [x1, y1] and [x2, y2]
        det = right[0] * current[1] - right[1] * current[0]  # determinant
        rot_angle = math.atan2(det, dot)  # atan2(y, x) or atan2(sin, cos)
        rot_angle = math.degrees(rot_angle)

        mask_heightmap_rotated = utils.rotate(self.mask_heightmap_pad, rot_angle, is_mask=True)
        input_image = mask_heightmap_rotated.astype(float) / 255
        input_image.shape = (
            input_image.shape[0],
            input_image.shape[1],
            input_image.shape[2],
        )

        with torch.no_grad():
            rot_angle = torch.tensor(rot_angle)
            input_data = torch.from_numpy(input_image.astype(np.float32)).permute(2, 0, 1)

        return rot_angle, input_data

    def __len__(self):
        return len(self.actions)


class GraspDataset(torch.utils.data.Dataset):
    """For grasp learning"""

    def __init__(self, color_heightmap, depth_heightmap, num_rotation):
        color_heightmap_pad = np.copy(color_heightmap)
        depth_heightmap_pad = np.copy(depth_heightmap)

        # Add extra padding (to handle rotations inside network)
        color_heightmap_pad = np.pad(
            color_heightmap_pad,
            ((IMAGE_PAD_WIDTH, IMAGE_PAD_WIDTH), (IMAGE_PAD_WIDTH, IMAGE_PAD_WIDTH), (0, 0)),
            "constant",
            constant_values=0,
        )
        depth_heightmap_pad = np.pad(depth_heightmap_pad, IMAGE_PAD_WIDTH, "constant", constant_values=0)

        # Pre-process color image (scale and normalize)
        image_mean = COLOR_MEAN
        image_std = COLOR_STD
        input_color_image = color_heightmap_pad.astype(float) / 255
        for c in range(3):
            input_color_image[:, :, c] = (input_color_image[:, :, c] - image_mean[c]) / image_std[c]

        # Pre-process depth image (normalize)
        image_mean = DEPTH_MEAN
        image_std = DEPTH_STD
        depth_heightmap_pad.shape = (depth_heightmap_pad.shape[0], depth_heightmap_pad.shape[1], 1)
        input_depth_image = np.copy(depth_heightmap_pad)
        input_depth_image[:, :, 0] = (input_depth_image[:, :, 0] - image_mean[0]) / image_std[0]

        self.input_color_data = torch.from_numpy(input_color_image.astype(np.float32)).permute(2, 0, 1)
        self.input_depth_data = torch.from_numpy(input_depth_image.astype(np.float32)).permute(2, 0, 1)
        self.rotation_idx = np.arange(num_rotation)
        self.unit_angle = 360 / num_rotation

    def __getitem__(self, idx):
        rotate_theta = self.rotation_idx[idx] * self.unit_angle

        rotate_color = TT.functional.rotate(self.input_color_data, rotate_theta, TT.InterpolationMode.NEAREST)
        rotate_depth = TT.functional.rotate(self.input_depth_data, rotate_theta, TT.InterpolationMode.NEAREST)

        input_data = torch.cat((rotate_color, rotate_depth), dim=0)

        return input_data

    def __len__(self):
        return len(self.rotation_idx)

class SegmentationDataset(torch.utils.data.Dataset):
    """
    Create segmentation dataset for training Mask R-CNN.
    One uses pre-defined color range to separate objects (assume the color in one image is unique).
    One directly reads masks.
    """

    def __init__(self, root, transforms, is_real=False, background=None):
        self.root = root
        self.transforms = transforms
        self.is_real = is_real
        # load all image files, sorting them to ensure that they are aligned
        self.color_imgs = list(sorted(os.listdir(os.path.join(root, "color-heightmaps"))))
        self.depth_imgs = list(sorted(os.listdir(os.path.join(root, "depth-heightmaps"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "masks"))))
        self.background = background
        if self.background is not None:
            self.background = cv2.imread(background)

    def __getitem__(self, idx):
        # load images
        color_path = os.path.join(self.root, "color-heightmaps", self.color_imgs[idx])
        # depth_path = os.path.join(self.root, "depth-heightmaps", self.depth_imgs[idx])

        # color image input
        color_img = cv2.imread(color_path)
        color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)

        mask_path = os.path.join(self.root, "masks", self.masks[idx])
        mask_img = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

        if self.background is not None:
            # random background
            color_img = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)
            # background = cv2.resize(self.background, color_img.shape[:2], interpolation=cv2.INTER_AREA)
            color_img[mask_img == 0, :] = self.background[mask_img == 0, :]
            color_img = color_img.astype(np.int16)
            for channel in range(color_img.shape[2]):  # R, G, B
                c_random = np.random.rand(1)
                c_random *= 30
                c_random -= 15
                c_random = c_random.astype(np.int16)
                color_img[mask_img == 0, channel] = color_img[mask_img == 0, channel] + c_random
            color_img = np.clip(color_img, 0, 255)
            color_img = color_img.astype(np.uint8)
            color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)

        # get masks
        masks = []
        labels = []
        if self.is_real:
            gray = cv2.cvtColor(color_img, cv2.COLOR_RGB2GRAY)
            gray = gray.astype(np.uint8)
            blurred = cv2.medianBlur(gray, 5)
            thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]
            cnts = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            cnts = imutils.grab_contours(cnts)
            for c in cnts:
                if cv2.contourArea(c) > 100:
                    mask = np.zeros(color_img.shape[:2], np.uint8)
                    cv2.drawContours(mask, [c], -1, (1), -1)
                    masks.append(mask)
                    # cv2.imshow('mask' + self.color_imgs[idx], mask)
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()
        else:
            for ci in np.unique(mask_img):
                if ci != 0:
                    mask = mask_img == ci
                    if np.sum((mask == True)) > 100:
                        masks.append(mask)
                        # NOTE: assume there is a single type of objects will have more than 1000 instances
                        labels.append(ci // 1000)

        num_objs = len(masks)
        if num_objs > 0:
            masks = np.stack(masks, axis=0)

        # get bounding box coordinates for each mask
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])
            if xmin == xmax or ymin == ymax:
                num_objs = 0

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        if num_objs > 0:
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        else:
            area = torch.as_tensor([0], dtype=torch.float32)
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        num_objs = torch.tensor(num_objs)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        target["num_obj"] = num_objs

        if self.transforms is not None:
            # img, target = self.transforms(img, target)
            img, target = self.transforms(color_img, target)

        return img, target

    def __len__(self):
        # return len(self.imgs)
        return len(self.color_imgs)



class GraspClassificationDataset(torch.utils.data.Dataset):
    """For grasp classification"""

    def __init__(self, root):

        all_images = []
        all_labels = []
        sub_roots = glob.glob(f"{root}/*/")

        for root in sub_roots:
            images = list(sorted(glob.glob(os.path.join(root, "depths", "*.depth.png"))))
            label = os.path.join(root, "labels", "labels_5.txt")
            label = np.loadtxt(label)
            all_images.extend(images)
            all_labels.extend(label)
        self.images = all_images
        self.labels = all_labels
        print(f"Dataset size: {len(all_images)}")

    def __getitem__(self, idx):
        image = im.open(self.images[idx])
        label = self.labels[idx]

        image = self.transform(image)
        label = torch.tensor([label], dtype=torch.float)

        return image, label

    def transform(self, image):
        # random rotation
        angle = int(torch.rand(1) * 360 - 180)
        image = TT.functional.rotate(image, angle)
        # to tensor
        image = TT.functional.pil_to_tensor(image)
        image = image.to(torch.float32)
        # convert the 100000
        image /= 100000
        # normalize
        image = TT.functional.normalize(image, DEPTH_MEAN, DEPTH_STD, inplace=True)

        return image

    def __len__(self):
        return len(self.images)


def post_process_grasp_label():
    pos_threshold = 5
    action_dir = "logs_grasp/train/hard-random-1999/actions/*.txt"
    label_dir = f"logs_grasp/train/hard-random-1999/labels/labels_{pos_threshold}.txt"

    actions = sorted(glob.glob(action_dir))
    labels = []
    for action in actions:
        record = np.loadtxt(action)
        pos = record[:, 0].sum()
        if pos >= pos_threshold:
            labels.append([1])
        else:
            labels.append([0])

    np.savetxt(label_dir, labels, fmt="%s")


if __name__ == "__main__":
    test = GraspClassificationDataset("logs_grasp/test")
    loader = torch.utils.data.DataLoader(test, shuffle=False, num_workers=0, batch_size=1)
    for data in loader:
        data = data[0].numpy()
        print(np.unique(data))

    # pos_threshold = 5
    # action_dir = "logs_grasp/train/hard-random-1999/actions/*.txt"
    # label_dir = f"logs_grasp/train/hard-random-1999/labels/labels_{pos_threshold}.txt"

    # actions = sorted(glob.glob(action_dir))
    # labels = []
    # for action in actions:
    #     record = np.loadtxt(action)
    #     pos = record[:, 0].sum()
    #     if pos >= pos_threshold:
    #         labels.append([1])
    #     else:
    #         labels.append([0])

    # np.savetxt(label_dir, labels, fmt="%s")
