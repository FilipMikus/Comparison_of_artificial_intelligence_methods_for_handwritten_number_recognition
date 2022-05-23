"""*********************************************************************************************************************
*    Title: Porovnanie metód a nástrojov na rozpoznávanie rukopisov: Shougao
*    Author: Martin Domorák, Jaroslav Ďurfina, Mátyás Kiss, Patrik Majtán, Martin Michale, Tomáš Štrba, Patrícia Török
*    Date: 7. 2. 2022
*    Code version: 2.0
*********************************************************************************************************************"""
import threading

import cv2
import numpy as np


class Model:
    def __init__(self, gui=None):
        self.gui = gui
        self.colored_image_array = None
        self.active_image = None
        self.preprocessed_image_array = None
        self.prev_preprocessed_image_array = None
        self.save_directory = None
        self.selected_images_paths = []
        self.used_filters = {"current": [], "last": []}

    def set_original_image_preview(self):
        self.active_image = self.colored_image_array

    def set_preprocessed_image_preview(self):
        self.active_image = self.preprocessed_image_array

    def undo_image_changes(self):
        if self.prev_preprocessed_image_array is not None:
            tmp_adj_img_array = self.preprocessed_image_array
            self.preprocessed_image_array = self.prev_preprocessed_image_array
            self.prev_preprocessed_image_array = tmp_adj_img_array
            tmp_current_filters = self.used_filters['current'].copy()
            self.used_filters['current'] = self.used_filters['last'].copy()
            self.used_filters['last'] = tmp_current_filters
        if self.gui.image_mode_choice.get() == 2:
            self.active_image = self.preprocessed_image_array

    def reset_image_changes(self):
        self.prev_preprocessed_image_array = self.preprocessed_image_array
        self.used_filters['last'] = self.used_filters['current'].copy()
        self.set_grayscale_img(self.colored_image_array)
        self.used_filters['current'] = self.used_filters['current'][0:1]
        if self.gui.image_mode_choice.get() == 2:
            self.active_image = self.preprocessed_image_array

    def set_colored_img(self, new_img_arr):
        self.colored_image_array = new_img_arr
        self.active_image = self.colored_image_array

    def set_grayscale_img(self, new_colored_img_arr):
        grayscale_function = lambda img_array: cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        self.preprocessed_image_array = grayscale_function(new_colored_img_arr)
        self.used_filters['last'] = self.used_filters['current'].copy()
        self.used_filters['current'].append(grayscale_function)

    def apply_erosion(self):
        self.prev_preprocessed_image_array = self.preprocessed_image_array
        kernel_width = int(self.gui.dilate_erode_kwidth_scale.get())
        kernel_height = int(self.gui.dilate_erode_kheight_scale.get())
        iterations = int(self.gui.dilate_erode_iter_scale.get())
        kernel = np.ones((kernel_height, kernel_width), np.uint8)
        erosion_function = lambda img_array: \
            cv2.erode(img_array, kernel, iterations=iterations)
        self.preprocessed_image_array = erosion_function(self.preprocessed_image_array)
        self.used_filters['last'] = self.used_filters['current'].copy()
        self.used_filters['current'].append(erosion_function)
        self.update_img()
        self.gui.image_mode_choice.set(2)

    def apply_dilatation(self):
        self.prev_preprocessed_image_array = self.preprocessed_image_array
        kernel_width = int(self.gui.dilate_erode_kwidth_scale.get())
        kernel_height = int(self.gui.dilate_erode_kheight_scale.get())
        iterations = int(self.gui.dilate_erode_iter_scale.get())
        kernel = np.ones((kernel_height, kernel_width), np.uint8)
        dilatation_function = lambda img_array: \
            cv2.dilate(img_array, kernel, iterations=iterations)
        self.preprocessed_image_array = dilatation_function(self.preprocessed_image_array)
        self.used_filters['last'] = self.used_filters['current'].copy()
        self.used_filters['current'].append(dilatation_function)
        self.update_img()
        self.gui.image_mode_choice.set(2)

    def update_img(self):
        self.active_image = self.preprocessed_image_array

    def apply_threshold(self):
        # THRESH_BINARY = 0
        # THRESH_BINARY_INV = 1
        # THRESH_TRUNC = 2
        # THRESH_TOZERO = 3
        # THRESH_TOZERO_INV = 4
        # THRESH_OTSU = 8
        # THRESH_TRIANGLE = 16
        self.prev_preprocessed_image_array = self.preprocessed_image_array
        type = self.gui.type_combobox.current()
        max_value = int(self.gui.max_value_scale.get())
        # otsu
        if type == 5:
            type = 8
        # triangle
        elif type == 6:
            type = 16
        # if pixel has higher than threshold value = pixel will be 255 (white)
        if type == 0 or type == 1:
            threshold_function = lambda img_array: \
                cv2.adaptiveThreshold(img_array, max_value,
                                      self.gui.method_combobox.current(), type,
                                      int(self.gui.block_size_scale.get()),
                                      int(self.gui.c_value_scale.get()))
            self.preprocessed_image_array = threshold_function(self.preprocessed_image_array)
        else:
            threshold_function = lambda img_array: \
                cv2.threshold(img_array, int(self.gui.threshold_value_scale.get()),
                              max_value, cv2.THRESH_BINARY + type)
            th, self.preprocessed_image_array = threshold_function(self.preprocessed_image_array)
        self.update_img()
        self.used_filters['last'] = self.used_filters['current'].copy()
        self.used_filters['current'].append(threshold_function)
        self.gui.image_mode_choice.set(2)

    def apply_canny_edge(self):
        self.prev_preprocessed_image_array = self.preprocessed_image_array
        threshold1 = int(self.gui.canny_threshold1_scale.get())
        threshold2 = int(self.gui.canny_threshold2_scale.get())
        canny_function = lambda img_array: cv2.Canny(img_array,
                                                     threshold1, threshold2)
        self.preprocessed_image_array = canny_function(self.preprocessed_image_array)
        self.used_filters['last'] = self.used_filters['current'].copy()
        self.used_filters['current'].append(canny_function)
        self.update_img()
        self.gui.image_mode_choice.set(2)

    def apply_gaussian_blur(self):
        self.prev_preprocessed_image_array = self.preprocessed_image_array
        kernel_width = int(self.gui.gaussian_blur_kwidth_scale.get())
        kernel_height = int(self.gui.gaussian_blur_kheight_scale.get())
        sigma = int(self.gui.gaussian_blur_sigma_scale.get())
        gaussian_blur_function = lambda img_array: \
            cv2.GaussianBlur(img_array, (kernel_width, kernel_height), sigma)
        self.preprocessed_image_array = gaussian_blur_function(self.preprocessed_image_array)
        self.used_filters['last'] = self.used_filters['current'].copy()
        self.used_filters['current'].append(gaussian_blur_function)
        self.update_img()
        self.gui.image_mode_choice.set(2)

    def apply_filters_to_images(self, img_path):
        img = cv2.imread(img_path)
        for filter_function in self.used_filters['current']:
            img = filter_function(img)
            # some thresholding methods return tuple with threshold value
            # we want just the image
            if type(img) is tuple:
                img = img[1]
        return img

    def select_images(self):
        selected_images_paths = self.gui.open_get_selected_images_paths_window()
        if selected_images_paths:
            self.selected_images_paths = selected_images_paths
            self.prev_preprocessed_image_array = None
            self.gui.images_select_text.set(f"Number of selected images: "
                                            f"{len(self.selected_images_paths)}")
            img = cv2.imread(self.selected_images_paths[0])
            self.used_filters['current'] = []
            self.used_filters['last'] = []
            self.set_colored_img(img)
            self.set_grayscale_img(img)
            self.gui.image_mode_choice.set(1)
            self.gui.open_image_preview_window()
            threading.Thread(target=self.gui.update_image_preview_window, daemon=True).start()

    def save_images(self):
        selected_directory = self.gui.open_get_save_directory_window()
        if selected_directory:
            for i in range(0, len(self.selected_images_paths)):
                img = self.apply_filters_to_images(self.selected_images_paths[i])
                cv2.imwrite(selected_directory + "/preprocessed_image_" + str(i) + ".jpg", img)
