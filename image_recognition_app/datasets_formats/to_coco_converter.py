import os
import json

import cv2
import numpy as np
import pandas as pd
from shapely.geometry import Polygon

import os
from random import randint
from PIL import Image, ImageOps

class DatasetConverter:

    @staticmethod
    def mnist_to_coco(source_mnist_path, destination_path, img_data_inverse=False, contours_examples=False):
        mnist_df = pd.read_csv(source_mnist_path)
        coco_json_dict = {"categories": [], "images": [], "annotations": []}

        try:
            os.mkdir(destination_path, 0o755)
        except OSError:
            print(f'Creation of the directory {destination_path} failed.')
        try:
            os.mkdir(destination_path + "/data", 0o755)
        except OSError:
            print(f'Creation of the sub-directory {destination_path + "/data"} failed.')

        for i in range(0, 10):
            coco_json_dict["categories"].append(
                {
                    "id": i,
                    "name": str(i)
                }
            )

        for i, row in mnist_df.iterrows():
            tmp_data = np.array(row)
            tmp_target = tmp_data[0]
            tmp_data = np.delete(tmp_data, 0)
            tmp_data = tmp_data.reshape(28, 28)
            if img_data_inverse:
                tmp_data = 255.0 - tmp_data     # PRIDANÝ KROK - INVERZIA
            cv2.imwrite(f'{destination_path}/data/img{i}.png', tmp_data)
            tmp_data_image = cv2.imread(f'{destination_path}/data/img{i}.png')
            if img_data_inverse:
                tmp_data_image = cv2.bitwise_not(tmp_data_image)

            tmp_height, tmp_width, tmp_channels = tmp_data_image.shape
            coco_json_dict["images"].append(
                {
                    "height": tmp_height,
                    "width": tmp_width,
                    "id": i,
                    "file_name": "img" + str(i) + ".png"
                }
            )

            tmp_data_image = cv2.cvtColor(tmp_data_image, cv2.COLOR_BGR2GRAY)
            tmp_dilate_data_image = cv2.dilate(tmp_data_image, np.ones((3, 3), np.uint8), iterations=1)
            tmp_contours, tmp_hierarchy = cv2.findContours(image=tmp_dilate_data_image, mode=cv2.RETR_EXTERNAL,
                                                           method=cv2.CHAIN_APPROX_NONE)
            if contours_examples:
                tmp_data_image = cv2.cvtColor(tmp_data_image, cv2.COLOR_GRAY2BGR)

            tmp_segmentation_mask_polygon_area = 0
            tmp_segmentation_mask_polygon_list = []
            for tmp_contour in tmp_contours:
                if len(tmp_contour) > 2:
                    if contours_examples:
                        if img_data_inverse:
                            tmp_data_image = cv2.bitwise_not(tmp_data_image)
                        cv2.drawContours(image=tmp_data_image, contours=[tmp_contour], contourIdx=0, color=(0, 0, 255),
                                         thickness=1,
                                         lineType=cv2.LINE_8)
                    tmp_segmentation_mask_polygon = tmp_contour.flatten()
                    tmp_segmentation_mask_polygon_list.append(tmp_segmentation_mask_polygon.tolist())
                    tmp_segmentation_mask_polygon_tuple = tuple(
                        [tuple(t) for t in np.reshape(tmp_segmentation_mask_polygon, (-1, 2))])
                    tmp_segmentation_mask_polygon_area += Polygon(tmp_segmentation_mask_polygon_tuple).area
            if contours_examples:
                cv2.imwrite(f'{destination_path}/data/img_{i}.png', tmp_data_image)
            coco_json_dict["annotations"].append(
                {
                    "id": i,
                    "image_id": i,
                    "category_id": int(tmp_target),
                    "bbox": [0, 0, tmp_width, tmp_height],
                    "segmentation": tmp_segmentation_mask_polygon_list,
                    "area": tmp_segmentation_mask_polygon_area,
                    "iscrowd": 0
                }
            )


        coco_json = json.dumps(coco_json_dict)
        coco_json_file = open(destination_path + "/labels.json", "w")
        coco_json_file.write(coco_json)
        coco_json_file.close()
        print("Conversion finished.")

    @staticmethod
    def ardis_to_coco(source_data_ardis_path, source_labels_ardis_path, destination_path, img_data_inverse=False,
                      contours_examples=False):
        ardis_data_df = pd.read_csv(source_data_ardis_path, sep=" ", header=None)
        ardis_labels_df = pd.read_csv(source_labels_ardis_path, sep="         ", header=None, engine="python")
        ardis_df = pd.concat([ardis_data_df, ardis_labels_df], axis=1)
        coco_json_dict = {"categories": [], "images": [], "annotations": []}

        try:
            os.mkdir(destination_path, 0o755)
        except OSError:
            print(f'Creation of the directory {destination_path} failed.')
        try:
            os.mkdir(destination_path + "/data", 0o755)
        except OSError:
            print(f'Creation of the sub-directory {destination_path + "/data"} failed.')

        for i in range(0, 10):
            coco_json_dict["categories"].append(
                {
                    "id": i,
                    "name": str(i)
                }
            )

        for i, row in ardis_df.iterrows():
            tmp_data = np.array(row)
            tmp_target = np.where(tmp_data[784:] == 1)[0][0]
            tmp_data = tmp_data[:784]
            tmp_data = tmp_data.reshape(28, 28)
            if img_data_inverse:
                tmp_data = 255.0 - tmp_data     # PRIDANÝ KROK - INVERZIA
            cv2.imwrite(f'{destination_path}/data/img{i}.png', tmp_data)
            tmp_data_image = cv2.imread(f'{destination_path}/data/img{i}.png')
            if img_data_inverse:
                tmp_data_image = cv2.bitwise_not(tmp_data_image)

            tmp_height, tmp_width, tmp_channels = tmp_data_image.shape
            coco_json_dict["images"].append(
                {
                    "height": tmp_height,
                    "width": tmp_width,
                    "id": i,
                    "file_name": "img" + str(i) + ".png"
                }
            )

            tmp_data_image = cv2.cvtColor(tmp_data_image, cv2.COLOR_BGR2GRAY)
            tmp_dilate_data_image = cv2.dilate(tmp_data_image, np.ones((3, 3), np.uint8), iterations=1)
            tmp_contours, tmp_hierarchy = cv2.findContours(image=tmp_dilate_data_image, mode=cv2.RETR_EXTERNAL,
                                                           method=cv2.CHAIN_APPROX_NONE)
            if contours_examples:
                tmp_data_image = cv2.cvtColor(tmp_data_image, cv2.COLOR_GRAY2BGR)

            tmp_segmentation_mask_polygon_area = 0
            tmp_segmentation_mask_polygon_list = []
            for tmp_contour in tmp_contours:
                if len(tmp_contour) > 2:
                    if img_data_inverse:
                        tmp_data_image = cv2.bitwise_not(tmp_data_image)
                    if contours_examples:
                            cv2.drawContours(image=tmp_data_image, contours=[tmp_contour], contourIdx=0, color=(0, 0, 255),
                                         thickness=1,
                                         lineType=cv2.LINE_8)
                    tmp_segmentation_mask_polygon = tmp_contour.flatten()
                    tmp_segmentation_mask_polygon_list.append(tmp_segmentation_mask_polygon.tolist())
                    tmp_segmentation_mask_polygon_tuple = tuple(
                        [tuple(t) for t in np.reshape(tmp_segmentation_mask_polygon, (-1, 2))])
                    tmp_segmentation_mask_polygon_area += Polygon(tmp_segmentation_mask_polygon_tuple).area
            if contours_examples:
                cv2.imwrite(f'{destination_path}/data/img_{i}.png', tmp_data_image)
            coco_json_dict["annotations"].append(
                {
                    "id": i,
                    "image_id": i,
                    "category_id": int(tmp_target),
                    "bbox": [0, 0, tmp_width, tmp_height],
                    "segmentation": tmp_segmentation_mask_polygon_list,
                    "area": tmp_segmentation_mask_polygon_area,
                    "iscrowd": 0
                }
            )

        coco_json = json.dumps(coco_json_dict)
        coco_json_file = open(destination_path + "/labels.json", "w")
        coco_json_file.write(coco_json)
        coco_json_file.close()
        print("Conversion finished.")

    @staticmethod
    def ardis_to_yolov5(source_train_data_ardis_path, source_val_data_ardis_path, source_train_labels_ardis_path,
                        source_val_labels_ardis_path, destination_path, img_data_inverse=False):
        ardis_train_data_df = pd.read_csv(source_train_data_ardis_path, sep=" ", header=None)
        ardis_train_labels_df = pd.read_csv(source_train_labels_ardis_path, sep="         ", header=None, engine="python")
        ardis_val_data_df = pd.read_csv(source_val_data_ardis_path, sep=" ", header=None)
        ardis_val_labels_df = pd.read_csv(source_val_labels_ardis_path, sep="         ", header=None, engine="python")
        ardis_train_df = pd.concat([ardis_train_data_df, ardis_train_labels_df], axis=1)
        ardis_val_df = pd.concat([ardis_val_data_df, ardis_val_labels_df], axis=1)

        try:
            os.mkdir(destination_path, 0o755)
        except OSError:
            print(f'Creation of the directory {destination_path} failed.')
        try:
            os.mkdir(destination_path + "/data", 0o755)
        except OSError:
            print(f'Creation of the sub-directory {destination_path + "/data"} failed.')
        try:
            os.mkdir(destination_path + "/labels", 0o755)
        except OSError:
            print(f'Creation of the sub-directory {destination_path + "/labels"} failed.')
        try:
            os.mkdir(destination_path + "/data/train", 0o755)
        except OSError:
            print(f'Creation of the sub-directory {destination_path + "/data/train"} failed.')
        try:
            os.mkdir(destination_path + "/labels/train", 0o755)
        except OSError:
            print(f'Creation of the sub-directory {destination_path + "/labels/train"} failed.')
        try:
            os.mkdir(destination_path + "/data/val", 0o755)
        except OSError:
            print(f'Creation of the sub-directory {destination_path + "/data/val"} failed.')
        try:
            os.mkdir(destination_path + "/labels/val", 0o755)
        except OSError:
            print(f'Creation of the sub-directory {destination_path + "/labels/val"} failed.')

        yaml_string = \
            "train: /data/train/\nval: /data/val/\nnc: 10\nnames: ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']"
        yolo_yaml_file = open(f'{destination_path}/ardis_yolo.yaml', "w")
        yolo_yaml_file.write(yaml_string)
        yolo_yaml_file.close()



        for i, row in ardis_train_df.iterrows():
            tmp_data = np.array(row)
            tmp_target = np.where(tmp_data[784:] == 1)[0][0]
            tmp_data = tmp_data[:784]
            tmp_data = tmp_data.reshape(28, 28)
            if img_data_inverse:
                tmp_data = 255.0 - tmp_data  # PRIDANÝ KROK - INVERZIA
            cv2.imwrite(f'{destination_path}/data/train/img{i}.png', tmp_data)
            tmp_data_image = cv2.imread(f'{destination_path}/data/train/img{i}.png')
            if img_data_inverse:
                tmp_data_image = cv2.bitwise_not(tmp_data_image)

            img_height, img_width, img_channels = tmp_data_image.shape

            tmp_height = 1.0 / img_height
            tmp_width = 1.0 / img_width
            tmp_center_x = 0 + 28 / 2.0
            tmp_center_y = 0 + 28 / 2.0

            center_x = tmp_center_x * tmp_width
            center_y = tmp_center_y * tmp_height
            width = img_width * tmp_width
            height = img_height * tmp_height

            yolo_string = f'{int(tmp_target)} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n'
            yolo_txt_file = open(f'{destination_path}/labels/train/img{i}.txt', "w")
            yolo_txt_file.write(yolo_string)
            yolo_txt_file.close()

        for i, row in ardis_val_df.iterrows():
            tmp_data = np.array(row)
            tmp_target = np.where(tmp_data[784:] == 1)[0][0]
            tmp_data = tmp_data[:784]
            tmp_data = tmp_data.reshape(28, 28)
            if img_data_inverse:
                tmp_data = 255.0 - tmp_data  # PRIDANÝ KROK - INVERZIA
            cv2.imwrite(f'{destination_path}/data/val/img{i}.png', tmp_data)
            tmp_data_image = cv2.imread(f'{destination_path}/data/val/img{i}.png')
            if img_data_inverse:
                tmp_data_image = cv2.bitwise_not(tmp_data_image)

            img_height, img_width, img_channels = tmp_data_image.shape

            tmp_height = 1.0 / img_height
            tmp_width = 1.0 / img_width
            tmp_center_x = 0 + 28 / 2.0
            tmp_center_y = 0 + 28 / 2.0

            center_x = tmp_center_x * tmp_width
            center_y = tmp_center_y * tmp_height
            width = img_width * tmp_width
            height = img_height * tmp_height

            yolo_string = f'{int(tmp_target)} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n'
            yolo_txt_file = open(f'{destination_path}/labels/val/img{i}.txt', "w")
            yolo_txt_file.write(yolo_string)
            yolo_txt_file.close()

        print("Conversion finished.")


if __name__ == '__main__':
   # DatasetConverter.mnist_to_coco("/Users/filipeno1/Downloads/Datasets/Others/MNIST/mnist_test.csv",
   #                                 "/Users/filipeno1/Downloads/Datasets/COCO/MNIST/test_invert_mnist_coco_contours", True, True)
   # DatasetConverter.ardis_to_coco("/Users/filipeno1/Downloads/Datasets/Others/ARDIS/ARDIS_test_2828.csv",
   #                               "/Users/filipeno1/Downloads/Datasets/Others/ARDIS/ARDIS_test_labels.csv",
   #                                "/Users/filipeno1/Downloads/Datasets/COCO/ARDIS/test_ardis_coco",
   #                                False, False)
    DatasetConverter.ardis_to_yolov5("/Users/filipeno1/Downloads/Datasets/Others/ARDIS/ARDIS_train_2828.csv",
                                     "/Users/filipeno1/Downloads/Datasets/Others/ARDIS/ARDIS_test_2828.csv",
                                     "/Users/filipeno1/Downloads/Datasets/Others/ARDIS/ARDIS_train_labels.csv",
                                     "/Users/filipeno1/Downloads/Datasets/Others/ARDIS/ARDIS_test_labels.csv",
                                     "/Users/filipeno1/Downloads/Datasets/YOLO/ARDIS/ardis_yolo", True)

