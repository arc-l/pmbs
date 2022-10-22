import math
import random
from multiprocessing import Pool, cpu_count, Process
import time
import numpy as np
import cv2
import torch
from scipy.spatial.transform import Rotation

import utils
from dataset import LifelongEvalDataset
from constants import (
    GRIPPER_PUSH_RADIUS_PIXEL,
    GRIPPER_PUSH_RADIUS_SAFE_PAD_PIXEL,
    IS_REAL,
    PIXEL_SIZE,
    GRIPPER_PUSH_ADD_PIXEL,
    IMAGE_PAD_WIDTH,
    GRIPPER_GRASP_WIDTH_PIXEL,
    IMAGE_PAD_WIDTH,
    IMAGE_SIZE,
    WORKSPACE_LIMITS,
    PUSH_DISTANCE,
)


def get_orientation(pts):
    sz = len(pts)
    data_pts = np.empty((sz, 2), dtype=np.float64)
    for i in range(data_pts.shape[0]):
        data_pts[i, 0] = pts[i, 0, 0]
        data_pts[i, 1] = pts[i, 0, 1]
    # Perform PCA analysis
    mean = np.empty((0))
    mean, eigenvectors, eigenvalues = cv2.PCACompute2(data_pts, mean)
    angle = math.atan2(eigenvectors[0, 1], eigenvectors[0, 0])  # orientation in radians

    return angle


def get_sign_line(pose0, pose1, pose2):
    """
    Line is from pose1 to pose2.
    if value > 0, pose0 is on the left side of the line.
    if value = 0, pose0 is on the same line.
    if value < 0, pose0 is on the right side of the line.
    """
    return (pose2[0] - pose1[0]) * (pose0[1] - pose1[1]) - (pose0[0] - pose1[0]) * (pose2[1] - pose1[1])


def distance_to_line(pose0, pose1, pose2):
    """
    Line is from pose1 to pose2.
    """
    return abs(
        (pose2[0] - pose1[0]) * (pose1[1] - pose0[1]) - (pose1[0] - pose0[0]) * (pose2[1] - pose1[1])
    ) / math.sqrt((pose2[0] - pose1[0]) ** 2 + (pose2[1] - pose1[1]) ** 2)


def adjust_push_start_point(
    pose0, pose1, contour, pi=-1, distance=GRIPPER_PUSH_RADIUS_PIXEL, add_distance=GRIPPER_PUSH_ADD_PIXEL,
):
    """
    Give two points, find the most left and right point on the contour within a given range based on pose1->pose0.
    So the push will not collide with the contour
    pose0: the center of contour
    pose1: the point on the contour
    pi: the index of pose1 on contour
    """
    r = math.sqrt((pose1[0] - pose0[0]) ** 2 + (pose1[1] - pose0[1]) ** 2)
    dx = round(distance / r * (pose0[1] - pose1[1]))
    dy = round(distance / r * (pose1[0] - pose0[0]))
    pose2 = (pose0[0] + dx, pose0[1] + dy)
    pose3 = (pose1[0] + dx, pose1[1] + dy)
    pose4 = (pose0[0] - dx, pose0[1] - dy)
    pose5 = (pose1[0] - dx, pose1[1] - dy)
    pose1_sign23 = get_sign_line(pose1, pose2, pose3)
    pose1_sign45 = get_sign_line(pose1, pose4, pose5)
    assert pose1_sign23 * pose1_sign45 < 0
    center_distance = distance_to_line(pose1, pose2, pose4)
    max_distance = center_distance
    if pi == -1:
        for p in range(0, len(contour)):
            test_pose = contour[p][0]
            test_pose_sign23 = get_sign_line(test_pose, pose2, pose3)
            test_pose_sign45 = get_sign_line(test_pose, pose4, pose5)
            # in the range, between two lines
            if pose1_sign23 * test_pose_sign23 >= 0 and pose1_sign45 * test_pose_sign45 >= 0:
                # is far enough
                test_center_distance = distance_to_line(test_pose, pose2, pose4)
                if test_center_distance >= max_distance:
                    # in the correct side
                    test_edge_distance = distance_to_line(test_pose, pose3, pose5)
                    if test_edge_distance < test_center_distance:
                        if test_center_distance > max_distance:
                            max_distance = test_center_distance
    else:
        for p in range(pi, len(contour)):
            test_pose = contour[p][0]
            test_pose_sign23 = get_sign_line(test_pose, pose2, pose3)
            test_pose_sign45 = get_sign_line(test_pose, pose4, pose5)
            # in the range, between two lines
            if pose1_sign23 * test_pose_sign23 >= 0 and pose1_sign45 * test_pose_sign45 >= 0:
                # is far enough
                test_center_distance = distance_to_line(test_pose, pose2, pose4)
                if test_center_distance >= max_distance:
                    # in the correct side
                    test_edge_distance = distance_to_line(test_pose, pose3, pose5)
                    if test_edge_distance < test_center_distance:
                        if test_center_distance > max_distance:
                            max_distance = test_center_distance
            else:
                break
        for p in range(pi, -1, -1):
            test_pose = contour[p][0]
            test_pose_sign23 = get_sign_line(test_pose, pose2, pose3)
            test_pose_sign45 = get_sign_line(test_pose, pose4, pose5)
            # in the range, between two lines
            if pose1_sign23 * test_pose_sign23 >= 0 and pose1_sign45 * test_pose_sign45 >= 0:
                # is far enough
                test_center_distance = distance_to_line(test_pose, pose2, pose4)
                if test_center_distance >= max_distance:
                    # in the correct side
                    test_edge_distance = distance_to_line(test_pose, pose3, pose5)
                    if test_edge_distance < test_center_distance:
                        if test_center_distance > max_distance:
                            max_distance = test_center_distance
            else:
                break

    diff_distance = abs(max_distance - center_distance)
    return math.ceil(diff_distance) + add_distance


def check_valid(point, point_on_contour, thresh):
    """TODO: for sampling that is not pre-defined, the out of boundary should also be checked but in a different way.
    Basically both ends of an push should be in the workspace limits"""
    # out of boundary
    if (
        not (GRIPPER_PUSH_RADIUS_SAFE_PAD_PIXEL < point[0] < IMAGE_SIZE - GRIPPER_PUSH_RADIUS_SAFE_PAD_PIXEL)
        or not (GRIPPER_PUSH_RADIUS_SAFE_PAD_PIXEL < point[1] < IMAGE_SIZE - GRIPPER_PUSH_RADIUS_SAFE_PAD_PIXEL)
        or not (
            GRIPPER_PUSH_RADIUS_SAFE_PAD_PIXEL < point_on_contour[0] < IMAGE_SIZE - GRIPPER_PUSH_RADIUS_SAFE_PAD_PIXEL
        )
        or not (
            GRIPPER_PUSH_RADIUS_SAFE_PAD_PIXEL < point_on_contour[1] < IMAGE_SIZE - GRIPPER_PUSH_RADIUS_SAFE_PAD_PIXEL
        )
    ):
        qualify = False
    elif thresh[point[1], point[0]] > 0:
        qualify = False
    else:
        # compute rotation angle
        down = (0, 1)
        current = (
            point_on_contour[0] - point[0],
            point_on_contour[1] - point[1],
        )
        dot = down[0] * current[0] + down[1] * current[1]  # dot product between [x1, y1] and [x2, y2]
        det = down[0] * current[1] - down[1] * current[0]  # determinant
        angle = math.atan2(det, dot)  # atan2(y, x) or atan2(sin, cos)
        angle = math.degrees(angle)
        crop = thresh[
            point[1] - GRIPPER_PUSH_RADIUS_SAFE_PAD_PIXEL : point[1] + GRIPPER_PUSH_RADIUS_SAFE_PAD_PIXEL + 1,
            point[0] - GRIPPER_PUSH_RADIUS_SAFE_PAD_PIXEL : point[0] + GRIPPER_PUSH_RADIUS_SAFE_PAD_PIXEL + 1,
        ]
        # test the rotated crop part
        crop = utils.rotate(crop, angle, is_mask=True)
        (h, w) = crop.shape
        crop_cy, crop_cx = (h // 2, w // 2)
        crop = crop[
            crop_cy
            - math.ceil(GRIPPER_GRASP_WIDTH_PIXEL / 2)
            - 1 : crop_cy
            + math.ceil(GRIPPER_GRASP_WIDTH_PIXEL / 2)
            + 2,
            crop_cx - GRIPPER_PUSH_RADIUS_PIXEL - 1 : crop_cx + GRIPPER_PUSH_RADIUS_PIXEL + 2,
        ]
        qualify = np.sum(crop > 0) == 0

    return qualify


def global_adjust(point, point_on_contour, thresh):
    """"Try different back distance"""
    # adjust_dis = [0.01, 0.02]
    adjust_dis = [0.01]
    for dis in adjust_dis:
        dis = dis / PIXEL_SIZE
        diff_x = point_on_contour[0] - point[0]
        diff_y = point_on_contour[1] - point[1]
        diff_norm = math.sqrt(diff_x ** 2 + diff_y ** 2)
        diff_x /= diff_norm
        diff_y /= diff_norm
        test_point = [int(round(point[0] - diff_x * dis)), int(round(point[1] - diff_y * dis))]
        qualify = check_valid(test_point, point_on_contour, thresh)
        if qualify:
            return qualify, test_point

    return False, None


def global_adjust_new(point, point_to, thresh):
    """"Try different back distance"""
    # adjust_dis = [0.01, 0.02]
    adjust_dis = [0.01]
    for dis in adjust_dis:
        dis = dis / PIXEL_SIZE
        diff_x = point_to[0] - point[0]
        diff_y = point_to[1] - point[1]
        diff_norm = math.sqrt(diff_x ** 2 + diff_y ** 2)
        diff_x /= diff_norm
        diff_y /= diff_norm
        test_point = [int(round(point[0] - diff_x * dis)), int(round(point[1] - diff_y * dis))]
        test_point_to = [int(round(point_to[0] - diff_x * dis)), int(round(point_to[1] - diff_y * dis))]
        qualify = check_valid(test_point, test_point_to, thresh)
        if qualify:
            return qualify, test_point, test_point_to

    return False, None, None


def sample_actions(color_image, mask_image, plot=False):
    """
    Sample actions around the objects, from the boundary to the center.
    Assume there is no object in "black"
    Output the rotated image, such that the push action is from left to right
    """

    # Process mask into binary format
    masks = []
    for i in np.unique(mask_image):
        if i == 0:
            continue
        mask = np.where(mask_image == i, 255, 0).astype(np.uint8)
        masks.append(mask)
    if len(masks) == 0:
        return []

    gray = cv2.cvtColor(color_image, cv2.COLOR_RGB2GRAY)
    gray = gray.astype(np.uint8)
    if plot:
        plot_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
    thresh = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)[1]

    # find the contour of a single object
    points_on_contour = []
    points = []
    four_idx = []
    other_idx = []
    for oi in range(len(masks)):
        obj_cnt = cv2.findContours(masks[oi], cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]
        obj_cnt = sorted(obj_cnt, key=lambda x: cv2.contourArea(x))
        if len(obj_cnt) == 0:
            continue
        else:
            obj_cnt = obj_cnt[-1]
            # if too small, then, we skip
            if cv2.contourArea(obj_cnt) < 10:
                continue
        # get center
        M = cv2.moments(obj_cnt)
        cX = round(M["m10"] / M["m00"])
        cY = round(M["m01"] / M["m00"])
        if plot:
            cv2.circle(plot_image, (cX, cY), 3, (255, 255, 255), -1)
        # get pca angle
        angle = get_orientation(obj_cnt)
        # get contour points
        # skip_num = len(obj_cnt) // 12  # 12 possible pushes for an object
        # skip_count = 0
        diff_angle_limit = 0.75  # around 45 degrees
        # target_diff_angles = np.array([0, np.pi, np.pi / 2, 3 * np.pi / 2])
        target_diff_angles = []
        # add four directions to center of object
        four_poses = [
            (cX + math.cos(angle) * 2, cY + math.sin(angle) * 2),
            (cX + math.cos(angle + np.pi / 2) * 2, cY + math.sin(angle + np.pi / 2) * 2),
            (cX + math.cos(angle - np.pi / 2) * 2, cY + math.sin(angle - np.pi / 2) * 2),
            (cX - math.cos(angle) * 2, cY - math.sin(angle) * 2),
        ]
        for pose in four_poses:
            x = pose[0]
            y = pose[1]
            diff_x = cX - x
            diff_y = cY - y
            test_angle = math.atan2(diff_y, diff_x)
            diff_norm = math.sqrt(diff_x ** 2 + diff_y ** 2)
            diff_x /= diff_norm
            diff_y /= diff_norm
            point_on_contour = (round(x), round(y))
            diff_mul = adjust_push_start_point((cX, cY), point_on_contour, obj_cnt)
            point = (round(x - diff_x * diff_mul), round(y - diff_y * diff_mul))
            should_append = check_valid(point, point_on_contour, thresh)
            if not should_append:
                should_append, point = global_adjust(point, point_on_contour, thresh)
            if should_append:
                points_on_contour.append(point_on_contour)
                points.append(point)
                four_idx.append(len(points) - 1)
                target_diff_angles.append(test_angle)
        for pi, p in enumerate(obj_cnt):
            x = p[0][0]
            y = p[0][1]
            if x == cX or y == cY:
                continue
            diff_x = cX - x
            diff_y = cY - y
            test_angle = math.atan2(diff_y, diff_x)
            # avoid similar directions to center of object
            if len(target_diff_angles) > 0:
                test_target_diff_angles = np.abs(np.array(target_diff_angles) - test_angle)
                should_append = (
                    np.min(test_target_diff_angles) > diff_angle_limit
                    and np.max(test_target_diff_angles) < math.pi * 2 - diff_angle_limit
                )
            else:
                should_append = True
            if should_append:
                diff_norm = math.sqrt(diff_x ** 2 + diff_y ** 2)
                diff_x /= diff_norm
                diff_y /= diff_norm
                point_on_contour = (int(round(x)), int(round(y)))
                diff_mul = adjust_push_start_point((cX, cY), point_on_contour, obj_cnt, pi)
                point = (int(round(x - diff_x * diff_mul)), int(round(y - diff_y * diff_mul)))
                should_append = check_valid(point, point_on_contour, thresh)
                if not should_append:
                    should_append, point = global_adjust(point, point_on_contour, thresh)
                if should_append:
                    points_on_contour.append(point_on_contour)
                    points.append(point)
                    other_idx.append(len(points) - 1)
                    target_diff_angles.append(test_angle)

    # random actions, adding priority points at the end
    random.shuffle(four_idx)
    random.shuffle(other_idx)
    new_points = []
    new_points_on_contour = []
    for idx in other_idx:
        new_points.append(points[idx])
        new_points_on_contour.append(points_on_contour[idx])
    for idx in four_idx:
        new_points.append(points[idx])
        new_points_on_contour.append(points_on_contour[idx])
    points = new_points
    points_on_contour = new_points_on_contour
    # idx_list = list(range(len(points)))
    # random.shuffle(idx_list)
    # new_points = []
    # new_points_on_contour = []
    # for idx in idx_list:
    #     new_points.append(points[idx])
    #     new_points_on_contour.append(points_on_contour[idx])
    # points = new_points
    # points_on_contour = new_points_on_contour

    if plot:
        # loop over the contours
        cnts = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]
        for c in cnts:
            cv2.drawContours(plot_image, [c], -1, (133, 137, 140), 2)

    actions = []
    for pi in range(len(points)):
        if plot:
            diff_x = points_on_contour[pi][0] - points[pi][0]
            diff_y = points_on_contour[pi][1] - points[pi][1]
            diff_norm = math.sqrt(diff_x ** 2 + diff_y ** 2)
            diff_x /= diff_norm
            diff_y /= diff_norm
            point_to = (
                int(points[pi][0] + diff_x * PUSH_DISTANCE / PIXEL_SIZE / 2),
                int(points[pi][1] + diff_y * PUSH_DISTANCE / PIXEL_SIZE / 2),
            )
            if pi < len(other_idx):
                cv2.arrowedLine(
                    plot_image, points[pi], point_to, (0, 0, 255), 2, tipLength=0.2,
                )
            else:
                cv2.arrowedLine(
                    plot_image, points[pi], point_to, (255, 0, 0), 2, tipLength=0.2,
                )
        push_start = (points[pi][1], points[pi][0])
        push_vector = np.array([points_on_contour[pi][1] - points[pi][1], points_on_contour[pi][0] - points[pi][0]])
        unit_push = push_vector / np.linalg.norm(push_vector)
        # for ratio in [0.5, 1, 1.5]:
        #     push_end = (
        #         int(round(push_start[0] + unit_push[0] * (ratio * PUSH_DISTANCE) / PIXEL_SIZE)),
        #         int(round(push_start[1] + unit_push[1] * (ratio * PUSH_DISTANCE) / PIXEL_SIZE)),
        #     )
        #     actions.append([push_start, push_end])
        push_end = (
            int(round(push_start[0] + unit_push[0] * PUSH_DISTANCE / PIXEL_SIZE)),
            int(round(push_start[1] + unit_push[1] * PUSH_DISTANCE / PIXEL_SIZE)),
        )
        actions.append([push_start, push_end])

    if plot:
        cv2.imwrite("test.png", plot_image)

    # random actions
    random.shuffle(actions)

    return actions


def _sample_action_one_object(inputs):
    mask, thresh = inputs
    points_on_contour = []
    points = []
    four_idx = []
    other_idx = []
    obj_cnt = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]
    obj_cnt = sorted(obj_cnt, key=lambda x: cv2.contourArea(x))
    if len(obj_cnt) == 0:
        return None
    else:
        obj_cnt = obj_cnt[-1]
        # if too small, then, we skip
        if cv2.contourArea(obj_cnt) < 10:
            return None
    # get center
    M = cv2.moments(obj_cnt)
    cX = round(M["m10"] / M["m00"])
    cY = round(M["m01"] / M["m00"])
    # get pca angle
    angle = get_orientation(obj_cnt)
    # get contour points
    diff_angle_limit = 0.75  # around 45 degrees
    target_diff_angles = []
    # add four directions to center of object
    four_poses = [
        (cX + math.cos(angle) * 2, cY + math.sin(angle) * 2),
        (cX + math.cos(angle + np.pi / 2) * 2, cY + math.sin(angle + np.pi / 2) * 2),
        (cX + math.cos(angle - np.pi / 2) * 2, cY + math.sin(angle - np.pi / 2) * 2),
        (cX - math.cos(angle) * 2, cY - math.sin(angle) * 2),
    ]
    for pose in four_poses:
        x = pose[0]
        y = pose[1]
        diff_x = cX - x
        diff_y = cY - y
        test_angle = math.atan2(diff_y, diff_x)
        diff_norm = math.sqrt(diff_x ** 2 + diff_y ** 2)
        diff_x /= diff_norm
        diff_y /= diff_norm
        point_on_contour = (round(x), round(y))
        diff_mul = adjust_push_start_point((cX, cY), point_on_contour, obj_cnt)
        point = (round(x - diff_x * diff_mul), round(y - diff_y * diff_mul))
        should_append = check_valid(point, point_on_contour, thresh)
        if not should_append:
            should_append, point = global_adjust(point, point_on_contour, thresh)
        if should_append:
            points_on_contour.append(point_on_contour)
            points.append(point)
            four_idx.append(len(points) - 1)
            target_diff_angles.append(test_angle)
    for pi, p in enumerate(obj_cnt):
        x = p[0][0]
        y = p[0][1]
        if x == cX or y == cY:
            continue
        diff_x = cX - x
        diff_y = cY - y
        test_angle = math.atan2(diff_y, diff_x)
        # avoid similar directions to center of object
        if len(target_diff_angles) > 0:
            test_target_diff_angles = np.abs(np.array(target_diff_angles) - test_angle)
            should_append = (
                np.min(test_target_diff_angles) > diff_angle_limit
                and np.max(test_target_diff_angles) < math.pi * 2 - diff_angle_limit
            )
        else:
            should_append = True
        if should_append:
            diff_norm = math.sqrt(diff_x ** 2 + diff_y ** 2)
            diff_x /= diff_norm
            diff_y /= diff_norm
            point_on_contour = (int(round(x)), int(round(y)))
            diff_mul = adjust_push_start_point((cX, cY), point_on_contour, obj_cnt, pi)
            point = (int(round(x - diff_x * diff_mul)), int(round(y - diff_y * diff_mul)))
            should_append = check_valid(point, point_on_contour, thresh)
            if not should_append:
                should_append, point = global_adjust(point, point_on_contour, thresh)
            if should_append:
                points_on_contour.append(point_on_contour)
                points.append(point)
                other_idx.append(len(points) - 1)
                target_diff_angles.append(test_angle)
    return points_on_contour, points, four_idx, other_idx


def sample_actions_wrapper(inputs):
    return sample_actions(inputs[0], inputs[1])


def sample_actions_parallel_batch(color_images, mask_images, pool):
    inputs = []
    for color_image, mask_image in zip(color_images, mask_images):
        inputs.append((color_image, mask_image))

    actions = pool.map(sample_actions_wrapper, inputs)

    return actions


def sample_actions_parallel(color_image, mask_image, pool, plot=False):
    """
    Sample actions around the objects, from the boundary to the center.
    Assume there is no object in "black"
    Output the rotated image, such that the push action is from left to right
    """

    # Process mask into binary format
    masks = []
    for i in np.unique(mask_image):
        if i == 0:
            continue
        mask = np.where(mask_image == i, 255, 0).astype(np.uint8)
        masks.append(mask)
    if len(masks) == 0:
        return []

    gray = cv2.cvtColor(color_image, cv2.COLOR_RGB2GRAY)
    gray = gray.astype(np.uint8)
    if plot:
        plot_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
    thresh = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)[1]

    # find the contour of a single object
    inputs = [(mask, thresh) for mask in masks]
    # with Pool() as pool:
    results = pool.map(_sample_action_one_object, inputs)

    # group result
    points_on_contour = []
    points = []
    four_idx = []
    other_idx = []
    for res in results:
        if res is not None:
            four = res[2]
            four = [i + len(points) for i in four]
            four_idx.extend(four)
            other = res[3]
            other = [i + len(points) for i in other]
            other_idx.extend(other)
            points_on_contour.extend(res[0])
            points.extend(res[1])

    # random actions, adding priority points at the end
    random.shuffle(four_idx)
    random.shuffle(other_idx)
    new_points = []
    new_points_on_contour = []
    for idx in other_idx:
        new_points.append(points[idx])
        new_points_on_contour.append(points_on_contour[idx])
    for idx in four_idx:
        new_points.append(points[idx])
        new_points_on_contour.append(points_on_contour[idx])
    points = new_points
    points_on_contour = new_points_on_contour
    # idx_list = list(range(len(points)))
    # random.shuffle(idx_list)
    # new_points = []
    # new_points_on_contour = []
    # for idx in idx_list:
    #     new_points.append(points[idx])
    #     new_points_on_contour.append(points_on_contour[idx])
    # points = new_points
    # points_on_contour = new_points_on_contour

    if plot:
        # loop over the contours
        cnts = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]
        for c in cnts:
            cv2.drawContours(plot_image, [c], -1, (133, 137, 140), 2)

    actions = []
    for pi in range(len(points)):
        if plot:
            diff_x = points_on_contour[pi][0] - points[pi][0]
            diff_y = points_on_contour[pi][1] - points[pi][1]
            diff_norm = math.sqrt(diff_x ** 2 + diff_y ** 2)
            diff_x /= diff_norm
            diff_y /= diff_norm
            point_to = (
                int(points[pi][0] + diff_x * PUSH_DISTANCE / PIXEL_SIZE),
                int(points[pi][1] + diff_y * PUSH_DISTANCE / PIXEL_SIZE),
            )
            if pi < len(other_idx):
                cv2.arrowedLine(
                    plot_image, points[pi], point_to, (235, 89, 5), 2, tipLength=0.2,
                )
            else:
                cv2.arrowedLine(
                    plot_image, points[pi], point_to, (235, 89, 5), 2, tipLength=0.2,
                )
        push_start = (points[pi][1], points[pi][0])
        push_vector = np.array([points_on_contour[pi][1] - points[pi][1], points_on_contour[pi][0] - points[pi][0]])
        unit_push = push_vector / np.linalg.norm(push_vector)
        # for ratio in [0.5, 1, 1.5]:
        #     push_end = (
        #         int(round(push_start[0] + unit_push[0] * (ratio * PUSH_DISTANCE) / PIXEL_SIZE)),
        #         int(round(push_start[1] + unit_push[1] * (ratio * PUSH_DISTANCE) / PIXEL_SIZE)),
        #     )
        #     actions.append([push_start, push_end])
        push_end = (
            int(round(push_start[0] + unit_push[0] * PUSH_DISTANCE / PIXEL_SIZE)),
            int(round(push_start[1] + unit_push[1] * PUSH_DISTANCE / PIXEL_SIZE)),
        )
        actions.append([push_start, push_end])

    if plot:
        cv2.imwrite("test.png", plot_image)

    # random actions
    random.shuffle(actions)

    return actions


@torch.no_grad()
def predict_action_q(model, actions, mask_image, device):
    model.pre_train = True
    dataset = LifelongEvalDataset(actions, mask_image)
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=len(actions), shuffle=False, num_workers=8, drop_last=False
    )
    rot_angle, input_data = next(iter(data_loader))
    input_data = input_data.to(device)
    # get output
    output = model(input_data)
    output = output.cpu().numpy()
    rot_angle = rot_angle.numpy()

    out_q = []
    for idx, out in enumerate(output):
        out = utils.rotate(out[0], -rot_angle[idx])
        action = actions[idx]
        q = np.max(
            out[
                action[0][0] + IMAGE_PAD_WIDTH - 3 : action[0][0] + IMAGE_PAD_WIDTH + 4,
                action[0][1] + IMAGE_PAD_WIDTH - 3 : action[0][1] + IMAGE_PAD_WIDTH + 4,
            ]
        )
        out_q.append(q)

    return out_q


def generate_sample_actions_for_all():
    import pickle

    obj_names = ["concave", "cube", "cylinder", "half-cube", "half-cylinder", "rect", "triangle"]

    for obj_name in obj_names:
        color_image = cv2.imread(f"actions/images/color_{obj_name}.png")
        mask_image = cv2.imread(f"actions/images/segm_{obj_name}.png", cv2.IMREAD_UNCHANGED)

        actions = generate_sample_actions(color_image, mask_image, obj_name, plot=True)
        print(obj_name, actions)

        if IS_REAL:
            pickle.dump(actions, open(f"actions/pre-defined/real_{obj_name}.p", "wb"))
        else:
            pickle.dump(actions, open(f"actions/pre-defined/{obj_name}.p", "wb"))


def generate_sample_actions(color_image, mask_image, name, plot=False):
    """
    Sample actions around the objects, from the boundary to the center.
    Assume there is no object in "black"
    Output the rotated image, such that the push action is from left to right
    """

    # Process mask into binary format
    masks = []
    for i in np.unique(mask_image):
        if i == 0:
            continue
        mask = np.where(mask_image == i, 255, 0).astype(np.uint8)
        masks.append(mask)
    if len(masks) == 0:
        return []

    gray = cv2.cvtColor(color_image, cv2.COLOR_RGB2GRAY)
    gray = gray.astype(np.uint8)
    if plot:
        plot_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
    thresh = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)[1]

    # find the contour of a single object
    points_on_contour = []
    points = []
    four_idx = []
    other_idx = []
    for oi in range(len(masks)):
        obj_cnt = cv2.findContours(masks[oi], cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]
        obj_cnt = sorted(obj_cnt, key=lambda x: cv2.contourArea(x))
        if len(obj_cnt) == 0:
            continue
        else:
            obj_cnt = obj_cnt[-1]
            # if too small, then, we skip
            if cv2.contourArea(obj_cnt) < 10:
                continue
        # get center
        M = cv2.moments(obj_cnt)
        cX = round(M["m10"] / M["m00"])
        cY = round(M["m01"] / M["m00"])
        if plot:
            cv2.circle(plot_image, (cX, cY), 3, (255, 255, 255), -1)
        # get pca angle
        angle = get_orientation(obj_cnt)
        # get contour points
        diff_angle_limit = 0.7  # around 45 degrees
        target_diff_angles = []
        # add four directions to center of object
        four_poses = [
            (cX + math.cos(angle) * 2, cY + math.sin(angle) * 2),
            (cX + math.cos(angle + np.pi / 2) * 2, cY + math.sin(angle + np.pi / 2) * 2),
            (cX + math.cos(angle - np.pi / 2) * 2, cY + math.sin(angle - np.pi / 2) * 2),
            (cX - math.cos(angle) * 2, cY - math.sin(angle) * 2),
            (cX + math.cos(angle + np.pi / 4) * 2, cY + math.sin(angle + np.pi / 4) * 2),
            (cX + math.cos(angle - np.pi / 4) * 2, cY + math.sin(angle - np.pi / 4) * 2),
            (cX + math.cos(angle + np.pi * 3 / 4) * 2, cY + math.sin(angle + np.pi * 3 / 4) * 2),
            (cX + math.cos(angle - np.pi * 3 / 4) * 2, cY + math.sin(angle - +np.pi * 3 / 4) * 2),
        ]
        # four_poses = [
        #     (cX + math.cos(angle) * 2, cY + math.sin(angle) * 2),
        #     (cX + math.cos(angle + np.pi / 2) * 2, cY + math.sin(angle + np.pi / 2) * 2),
        #     (cX + math.cos(angle - np.pi / 2) * 2, cY + math.sin(angle - np.pi / 2) * 2),
        #     (cX - math.cos(angle) * 2, cY - math.sin(angle) * 2),
        #     (cX + math.cos(angle + np.pi / 6) * 2, cY + math.sin(angle + np.pi / 6) * 2),
        #     (cX + math.cos(angle + np.pi * 2 / 6) * 2, cY + math.sin(angle + np.pi * 2 / 6) * 2),
        #     (cX + math.cos(angle - np.pi / 6) * 2, cY + math.sin(angle - np.pi / 6) * 2),
        #     (cX + math.cos(angle - np.pi * 2 / 6) * 2, cY + math.sin(angle - np.pi * 2 / 6) * 2),
        #     (cX + math.cos(angle + np.pi * 4 / 6) * 2, cY + math.sin(angle + np.pi * 4 / 6) * 2),
        #     (cX + math.cos(angle + np.pi * 5 / 6) * 2, cY + math.sin(angle + np.pi * 5 / 6) * 2),
        #     (cX + math.cos(angle - np.pi * 4 / 6) * 2, cY + math.sin(angle - np.pi * 4 / 6) * 2),
        #     (cX + math.cos(angle - np.pi * 5 / 6) * 2, cY + math.sin(angle - np.pi * 5 / 6) * 2),
        # ]
        for pose in four_poses:
            x = pose[0]
            y = pose[1]
            diff_x = cX - x
            diff_y = cY - y
            test_angle = math.atan2(diff_y, diff_x)
            diff_norm = math.sqrt(diff_x ** 2 + diff_y ** 2)
            diff_x /= diff_norm
            diff_y /= diff_norm
            point_on_contour = [round(x), round(y)]
            diff_mul = adjust_push_start_point((cX, cY), point_on_contour, obj_cnt)
            point = [round(x - diff_x * diff_mul), round(y - diff_y * diff_mul)]
            should_append = check_valid(point, point_on_contour, thresh)
            if not should_append:
                should_append, point = global_adjust(point, point_on_contour, thresh)
            if should_append:
                points_on_contour.append(point_on_contour)
                points.append(point)
                four_idx.append(len(points) - 1)
                target_diff_angles.append(test_angle)
        # for pi, p in enumerate(obj_cnt):
        #     x = p[0][0]
        #     y = p[0][1]
        #     if x == cX or y == cY:
        #         continue
        #     diff_x = cX - x
        #     diff_y = cY - y
        #     test_angle = math.atan2(diff_y, diff_x)
        #     # avoid similar directions to center of object
        #     if len(target_diff_angles) > 0:
        #         test_target_diff_angles = np.abs(np.array(target_diff_angles) - test_angle)
        #         should_append = (
        #             np.min(test_target_diff_angles) > diff_angle_limit
        #             and np.max(test_target_diff_angles) < math.pi * 2 - diff_angle_limit
        #         )
        #     else:
        #         should_append = True
        #     if should_append:
        #         diff_norm = math.sqrt(diff_x ** 2 + diff_y ** 2)
        #         diff_x /= diff_norm
        #         diff_y /= diff_norm
        #         point_on_contour = [int(round(x)), int(round(y))]
        #         diff_mul = adjust_push_start_point((cX, cY), point_on_contour, obj_cnt, pi)
        #         point = [int(round(x - diff_x * diff_mul)), int(round(y - diff_y * diff_mul))]
        #         should_append = check_valid(point, point_on_contour, thresh)
        #         if not should_append:
        #             should_append, point = global_adjust(point, point_on_contour, thresh)
        #         if should_append:
        #             points_on_contour.append(point_on_contour)
        #             points.append(point)
        #             other_idx.append(len(points) - 1)
        #             target_diff_angles.append(test_angle)

    new_points = []
    new_points_on_contour = []
    for idx in other_idx:
        new_points.append(points[idx])
        new_points_on_contour.append(points_on_contour[idx])
    for idx in four_idx:
        new_points.append(points[idx])
        new_points_on_contour.append(points_on_contour[idx])
    points = new_points
    points_on_contour = new_points_on_contour

    if plot:
        # loop over the contours
        cnts = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]
        for c in cnts:
            cv2.drawContours(plot_image, [c], -1, (133, 137, 140), 2)

    actions = []
    for pi in range(len(points)):
        if plot:
            diff_x = points_on_contour[pi][0] - points[pi][0]
            diff_y = points_on_contour[pi][1] - points[pi][1]
            diff_norm = math.sqrt(diff_x ** 2 + diff_y ** 2)
            diff_x /= diff_norm
            diff_y /= diff_norm
            point_to = (
                int(points[pi][0] + diff_x * PUSH_DISTANCE / PIXEL_SIZE / 2),
                int(points[pi][1] + diff_y * PUSH_DISTANCE / PIXEL_SIZE / 2),
            )
            if pi < len(other_idx):
                cv2.arrowedLine(
                    plot_image, points[pi], point_to, (0, 0, 255), 2, tipLength=0.2,
                )
            else:
                cv2.arrowedLine(
                    plot_image, points[pi], point_to, (255, 0, 0), 2, tipLength=0.2,
                )
        points[pi][0] -= IMAGE_SIZE // 2
        points[pi][1] -= IMAGE_SIZE // 2
        points_on_contour[pi][0] -= IMAGE_SIZE // 2
        points_on_contour[pi][1] -= IMAGE_SIZE // 2
        push_start = (points[pi][0], points[pi][1])
        push_vector = np.array([points_on_contour[pi][0] - points[pi][0], points_on_contour[pi][1] - points[pi][1]])
        unit_push = push_vector / np.linalg.norm(push_vector)
        push_end = (
            int(round(push_start[0] + unit_push[0] * PUSH_DISTANCE / PIXEL_SIZE)),
            int(round(push_start[1] + unit_push[1] * PUSH_DISTANCE / PIXEL_SIZE)),
        )
        actions.append([push_start, push_end])

    if plot:
        cv2.imwrite(f"test-{name}.png", plot_image)

    return actions


def load_pre_defined_actions(obj_names):
    import pickle

    all_obj_actions = []
    for obj_name in obj_names:
        if IS_REAL:
            with open(f"actions/pre-defined/real_{obj_name}.p", "rb") as input_file:
                actions = pickle.load(input_file)
                all_obj_actions.append(np.array(actions))
        else:
            with open(f"actions/pre-defined/{obj_name}.p", "rb") as input_file:
                actions = pickle.load(input_file)
                all_obj_actions.append(np.array(actions))

    return all_obj_actions


def quaternion_to_euler_angle_vectorized(quat):
    x = quat[:, 0]
    y = quat[:, 1]
    z = quat[:, 2]
    w = quat[:, 3]
    # x, y, z, w = quat
    ysqr = y * y

    # t0 = +2.0 * (w * x + y * z)
    # t1 = +1.0 - 2.0 * (x * x + ysqr)
    # X = np.arctan2(t0, t1)

    # t2 = +2.0 * (w * y - z * x)

    # t2 = np.clip(t2, a_min=-1.0, a_max=1.0)
    # Y = np.arcsin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (ysqr + z * z)
    Z = np.arctan2(t3, t4)

    return Z


def _sample_pre_defined_actions_one_object(inputs):
    actions, object_pose, angle, thresh = inputs
    actions = actions.copy()

    points = []
    points_to = []

    num_actions = len(actions)
    actions.shape = (-1, 2)
    trans = object_pose[0:2]
    sim_x = (trans[0] - WORKSPACE_LIMITS[0][0]) / PIXEL_SIZE
    sim_y = (trans[1] - WORKSPACE_LIMITS[1][0]) / PIXEL_SIZE
    trans[0] = sim_y
    trans[1] = sim_x
    x = actions[:, 0].copy()
    y = actions[:, 1].copy()
    actions[:, 0] = y * np.sin(angle) + x * np.cos(angle)
    actions[:, 1] = y * np.cos(angle) - x * np.sin(angle)
    actions = actions + trans
    actions.shape = (num_actions, 2, 2)
    actions = np.round(actions).astype(np.int32)

    for action in actions:
        point = action[0]
        point_to = action[1]
        should_append = check_valid(point, point_to, thresh)
        if not should_append:
            should_append, point, point_to = global_adjust_new(point, point_to, thresh)
        if should_append:
            points_to.append(point_to)
            points.append(point)

    return points, points_to


def sample_pre_defined_actions_parallel(color_image, defined_actions, object_poses, pool, plot=False):
    object_poses = object_poses.copy()

    gray = cv2.cvtColor(color_image, cv2.COLOR_RGB2GRAY)
    gray = gray.astype(np.uint8)
    if plot:
        plot_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
    thresh = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)[1]

    angles = quaternion_to_euler_angle_vectorized(object_poses[:, 3:7])

    inputs = [(defined_actions[idx], object_poses[idx], angles[idx], thresh) for idx in range(len(object_poses))]
    results = pool.map(_sample_pre_defined_actions_one_object, inputs)

    # group result
    points_to = []
    points = []
    for res in results:
        if res is not None:
            points.extend(res[0])
            points_to.extend(res[1])

    if plot:
        # loop over the contours
        cnts = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]
        for c in cnts:
            cv2.drawContours(plot_image, [c], -1, (133, 137, 140), 2)

    actions = []
    for pi in range(len(points)):
        if plot:
            # diff_x = points_to[pi][0] - points[pi][0]
            # diff_y = points_to[pi][1] - points[pi][1]
            # diff_norm = math.sqrt(diff_x ** 2 + diff_y ** 2)
            # diff_x /= diff_norm
            # diff_y /= diff_norm
            # point_to = (
            #     int(points[pi][0] + diff_x * PUSH_DISTANCE / PIXEL_SIZE / 2),
            #     int(points[pi][1] + diff_y * PUSH_DISTANCE / PIXEL_SIZE / 2),
            # )
            cv2.arrowedLine(
                plot_image, points[pi], points_to[pi], (235, 89, 5), 2, tipLength=0.2,
            )
        push_start = (points[pi][1], points[pi][0])
        push_end = (points_to[pi][1], points_to[pi][0])
        actions.append([push_start, push_end])
        # push_vector = np.array([points_to[pi][1] - points[pi][1], points_to[pi][0] - points[pi][0]])
        # unit_push = push_vector / np.linalg.norm(push_vector)
        # for ratio in [0.5, 1, 1.5]:
        #     push_end = (
        #         int(round(push_start[0] + unit_push[0] * (ratio * PUSH_DISTANCE) / PIXEL_SIZE)),
        #         int(round(push_start[1] + unit_push[1] * (ratio * PUSH_DISTANCE) / PIXEL_SIZE)),
        #     )
        #     actions.append([push_start, push_end])

    if plot:
        cv2.imwrite("test.png", plot_image)

    # random actions
    random.shuffle(actions)

    return actions


def sample_pre_defined_actions_wrapper(inputs):
    return sample_pre_defined_actions(inputs[0], inputs[1], inputs[2])


def sample_pre_defined_actions_parallel_batch(inputs, pool):
    actions = pool.map(sample_pre_defined_actions_wrapper, inputs)
    return actions


def sample_pre_defined_actions(color_image, defined_actions, object_poses, plot=False):
    """
    Sample actions around the objects, from the boundary to the center.
    Assume there is no object in "black"
    """

    object_poses = object_poses.copy()

    gray = cv2.cvtColor(color_image, cv2.COLOR_RGB2GRAY)
    gray = gray.astype(np.uint8)
    if plot:
        plot_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
    thresh = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)[1]
    angles = quaternion_to_euler_angle_vectorized(object_poses[:, 3:7])

    # find the contour of a single object
    points = []
    points_to = []
    for idx, actions in enumerate(defined_actions):
        actions = actions.copy()
        num_actions = len(actions)
        object_pose = object_poses[idx]

        actions.shape = (-1, 2)
        angle = angles[idx]
        trans = object_pose[0:2]
        sim_x = (trans[0] - WORKSPACE_LIMITS[0][0]) / PIXEL_SIZE
        sim_y = (trans[1] - WORKSPACE_LIMITS[1][0]) / PIXEL_SIZE
        trans[0] = sim_y
        trans[1] = sim_x
        x = actions[:, 0].copy()
        y = actions[:, 1].copy()
        actions[:, 0] = y * np.sin(angle) + x * np.cos(angle)
        actions[:, 1] = y * np.cos(angle) - x * np.sin(angle)
        actions = actions + trans
        actions.shape = (num_actions, 2, 2)
        actions = np.round(actions).astype(np.int32)

        for action in actions:
            point = action[0]
            point_to = action[1]
            should_append = check_valid(point, point_to, thresh)
            if not should_append:
                should_append, point, point_to = global_adjust_new(point, point_to, thresh)
            if should_append:
                points_to.append(point_to)
                points.append(point)

    if plot:
        # loop over the contours
        cnts = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]
        for c in cnts:
            cv2.drawContours(plot_image, [c], -1, (133, 137, 140), 2)

    actions = []
    for pi in range(len(points)):
        if plot:
            # diff_x = points_to[pi][0] - points[pi][0]
            # diff_y = points_to[pi][1] - points[pi][1]
            # diff_norm = math.sqrt(diff_x ** 2 + diff_y ** 2)
            # diff_x /= diff_norm
            # diff_y /= diff_norm
            # point_to = (
            #     int(points[pi][0] + diff_x * PUSH_DISTANCE / PIXEL_SIZE / 2),
            #     int(points[pi][1] + diff_y * PUSH_DISTANCE / PIXEL_SIZE / 2),
            # )
            cv2.arrowedLine(
                plot_image, points[pi], points_to[pi], (255, 0, 0), 2, tipLength=0.2,
            )
        push_start = (points[pi][1], points[pi][0])
        push_end = (points_to[pi][1], points_to[pi][0])
        actions.append([push_start, push_end])
        # push_vector = np.array([points_to[pi][1] - points[pi][1], points_to[pi][0] - points[pi][0]])
        # unit_push = push_vector / np.linalg.norm(push_vector)
        # for ratio in [0.5, 1, 1.5]:
        #     push_end = (
        #         int(round(push_start[0] + unit_push[0] * (ratio * PUSH_DISTANCE) / PIXEL_SIZE)),
        #         int(round(push_start[1] + unit_push[1] * (ratio * PUSH_DISTANCE) / PIXEL_SIZE)),
        #     )
        #     actions.append([push_start, push_end])

    if plot:
        cv2.imwrite("test.png", plot_image)
        # cv2.imshow("test", plot_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    # random actions
    random.shuffle(actions)

    return actions


if __name__ == "__main__":
    # pre define actions for all objects
    generate_sample_actions_for_all()

    # color_image = cv2.imread("actions/images/color_rect.png")
    # mask_image = cv2.imread("actions/images/segm_rect.png", cv2.IMREAD_UNCHANGED)
    # pool = Pool()
    # s = time.time()
    # for i in range(1):
    #     actions = sample_actions_parallel(color_image, mask_image, pool, plot=False)
    # e = time.time()
    # print(e - s)
