import os
import tkinter as tk
from tkinter import filedialog, ttk, StringVar
from tkinter import messagebox
from tkinter.font import Font

import cv2
from ttkthemes import themed_tk
from ttkwidgets import TickScale


class GraphicalUserInterface:
    def __init__(self, model=None, root_width=700, root_height=700):
        self.model = model
        # ******************** ROOT WINDOW ********************
        self.root_width = root_width
        self.root_height = root_height
        self.root_window = themed_tk.ThemedTk()
        self.root_window.set_theme('arc')
        self.root_window.title("Image Preprocessor")
        # ******************** MAIN FRAME ********************
        self.main_frame = ttk.Frame(self.root_window)
        self.menu_frame = ttk.Frame(self.main_frame)
        self.control_frame1 = ttk.Frame(self.main_frame)
        self.control_frame2 = ttk.Frame(self.main_frame)
        # ******************** MENU ********************
        self.images_select_frame = ttk.Frame(self.menu_frame, relief="solid")
        self.images_save_frame = ttk.Frame(self.menu_frame, relief="solid")
        self.images_options_frame = ttk.Frame(self.menu_frame, relief="solid")
        self.images_options_subframe1 = ttk.Frame(self.images_options_frame, relief="solid")
        self.images_options_subframe2 = ttk.Frame(self.images_options_frame, relief="solid")
        self.images_select_text = StringVar()
        self.images_select_text.set("Number of selected images: 0")
        self.images_select_label = ttk.Label(self.images_select_frame,
                                             textvariable=self.images_select_text,
                                             font=Font(family="Helvetica", size=10))
        self.images_select_button = ttk.Button(self.images_select_frame,
                                               text="Select images",
                                               command=self.model.select_images)
        self.images_save_label = ttk.Label(self.images_save_frame,
                                           font=Font(family="Helvetica", size=10),
                                           text="Save images:")
        self.save_as_images_button = ttk.Button(self.images_save_frame,
                                                text="Save as",
                                                command=self.model.save_images)
        self.images_options_label = ttk.Label(self.images_options_subframe1,
                                              font=Font(family="Helvetica", size=10),
                                              text="Image preview mode:")
        self.image_mode_choice = tk.IntVar(None, 1)
        self.original_image_radio = ttk.Radiobutton(self.images_options_subframe1,
                                                    variable=self.image_mode_choice,
                                                    value=1, text="Original",
                                                    command=self.model.set_original_image_preview)
        self.preprocessed_image_radio = ttk.Radiobutton(self.images_options_subframe1,
                                                        variable=self.image_mode_choice,
                                                        value=2, text="Preprocessed",
                                                        command=self.model.set_preprocessed_image_preview)
        self.step_back_button = ttk.Button(self.images_options_subframe2, text="Undo",
                                           command=self.model.undo_image_changes)
        self.reset_button = ttk.Button(self.images_options_subframe2, text="Reset",
                                       command=self.model.reset_image_changes)
        # ******************** IMAGE THRESHOLDING ********************
        self.image_threshold_label = ttk.Label(self.control_frame1,
                                               font=Font(family="Helvetica", size=12, weight="bold"),
                                               text="Image Thresholding")
        self.image_threshold_frame = ttk.Frame(self.control_frame1, relief="solid")
        self.image_threshold_subframe = ttk.Frame(self.image_threshold_frame, relief="solid")
        self.max_value_label = ttk.Label(self.image_threshold_frame, font=Font(family="Helvetica", size=10),
                                         text="Max value")
        self.threshold_value_label = ttk.Label(self.image_threshold_frame, font=Font(family="Helvetica", size=10),
                                               text="Threshold")
        self.block_size_label = ttk.Label(self.image_threshold_frame, font=Font(family="Helvetica", size=10),
                                          text="Block size")
        self.c_value_label = ttk.Label(self.image_threshold_frame, font=Font(family="Helvetica", size=10),
                                       text="C value")
        self.max_value_scale = TickScale(self.image_threshold_frame, from_=0, to=255,
                                         resolution=1, orient=tk.VERTICAL)
        self.max_value_scale.set(255)
        self.threshold_value_scale = TickScale(self.image_threshold_frame, from_=0, to=255,
                                               resolution=1, orient=tk.VERTICAL)
        self.threshold_value_scale.set(127)
        self.block_size_scale = TickScale(self.image_threshold_frame, from_=1, to=80,
                                          resolution=2, orient=tk.VERTICAL)
        self.block_size_scale.set(11)
        self.c_value_scale = TickScale(self.image_threshold_frame, from_=0, to=80, resolution=1,
                                       orient=tk.VERTICAL)
        self.c_value_scale.set(11)
        self.apply_button = ttk.Button(self.image_threshold_frame, text="Apply threshold",
                                       command=self.model.apply_threshold)
        self.method_label = ttk.Label(self.image_threshold_subframe, font=Font(family="Helvetica", size=10),
                                      text="Method")
        self.type_label = ttk.Label(self.image_threshold_subframe, font=Font(family="Helvetica", size=10),
                                    text="Type")
        self.method_choice = tk.StringVar().get()
        self.type_choice = tk.StringVar().get()
        self.method_combobox = ttk.Combobox(self.image_threshold_subframe,
                                            textvariable=self.method_choice,
                                            state="readonly")
        self.type_combobox = ttk.Combobox(self.image_threshold_subframe,
                                          textvariable=self.type_choice,
                                          state="readonly")
        self.method_combobox["values"] = ("Mean C", "Gaussian C")
        self.type_combobox["values"] = ("Binary", "Binary Inverse", "Trunc",
                                        "To Zero", "To Zero Inverse",
                                        "Otsu", "Triangle")
        self.type_combobox.current(1)
        self.method_combobox.current(1)
        # ******************** CANNY EDGE ********************
        self.canny_edge_label = ttk.Label(self.control_frame1, font=Font(family="Helvetica", size=12, weight="bold"),
                                          text="Canny Edge")
        self.canny_edge_frame = ttk.Frame(self.control_frame1, relief="solid")
        self.canny_threshold_label = ttk.Label(self.canny_edge_frame, font=Font(family="Helvetica", size=10),
                                               text="Threshold values")
        self.canny_threshold1_label = ttk.Label(self.canny_edge_frame, font=Font(family="Helvetica", size=10),
                                                text="Threshold 1")
        self.canny_threshold1_scale = TickScale(self.canny_edge_frame, from_=0, to=500,
                                                resolution=1, orient=tk.VERTICAL)
        self.canny_threshold1_scale.set(110)
        self.canny_threshold2_scale = TickScale(self.canny_edge_frame, from_=0, to=500,
                                                resolution=1, orient=tk.VERTICAL)
        self.canny_threshold2_label = ttk.Label(self.canny_edge_frame, font=Font(family="Helvetica", size=10),
                                                text="Threshold 2")
        self.canny_threshold2_scale.set(110)
        self.canny_apply_button = ttk.Button(self.canny_edge_frame, text="Apply canny",
                                             command=self.model.apply_canny_edge)
        # ******************** DILATE/ERODE ********************
        self.dilate_erode_label = ttk.Label(self.control_frame2,
                                            font=Font(family="Helvetica", size=12, weight="bold"),
                                            text="Dilate / Erode")
        self.dilate_erode_frame = ttk.Frame(self.control_frame2, relief="solid")
        self.dilate_erode_subframe = ttk.Frame(self.dilate_erode_frame)
        self.dilate_erode_kernel_label = ttk.Label(self.dilate_erode_frame,
                                                   font=Font(family="Helvetica", size=10),
                                                   text="Kernel")
        self.dilate_erode_kwidth_label = ttk.Label(self.dilate_erode_frame,
                                                   font=Font(family="Helvetica", size=10), text="Width")
        self.dilate_erode_kwidth_scale = TickScale(self.dilate_erode_frame, from_=1, to=15,
                                                   orient=tk.VERTICAL, resolution=1)
        self.dilate_erode_kwidth_scale.set(5)
        self.dilate_erode_kheight_label = ttk.Label(self.dilate_erode_frame,
                                                    font=Font(family="Helvetica", size=10),
                                                    text="Height")
        self.dilate_erode_kheight_scale = TickScale(self.dilate_erode_frame, from_=1, to=15,
                                                    orient=tk.VERTICAL, resolution=1)
        self.dilate_erode_kheight_scale.set(3)
        self.dilate_erode_iter_label = ttk.Label(self.dilate_erode_frame, font=Font(family="Helvetica", size=10),
                                                 text="Iterations")
        self.dilate_erode_iter_scale = TickScale(self.dilate_erode_frame, from_=1, to=15,
                                                 orient=tk.VERTICAL, resolution=2)
        self.erode_apply_button = ttk.Button(self.dilate_erode_subframe,
                                             text="Apply erosion",
                                             command=self.model.apply_erosion)
        self.dilate_apply_button = ttk.Button(self.dilate_erode_subframe,
                                              text="Apply dilation",
                                              command=self.model.apply_dilatation)
        # ******************** GAUSSIAN BLUR ********************
        self.gaussian_blur_label = ttk.Label(self.control_frame2,
                                             font=Font(family="Helvetica", size=12, weight="bold"),
                                             text="Gaussian Blur")
        self.gaussian_blur_frame = ttk.Frame(self.control_frame2, relief="solid")
        self.gaussian_blur_kernel_label = ttk.Label(self.gaussian_blur_frame, font=Font(family="Helvetica", size=10),
                                                    text="Kernel")
        self.gaussian_blur_kwidth_label = ttk.Label(self.gaussian_blur_frame,
                                                    font=Font(family="Helvetica", size=10), text="Width")
        self.gaussian_blur_kwidth_scale = TickScale(self.gaussian_blur_frame, from_=1, to=15,
                                                    orient=tk.VERTICAL, resolution=2)
        self.gaussian_blur_kwidth_scale.set(5)
        self.gaussian_blur_kheight_label = ttk.Label(self.gaussian_blur_frame, font=Font(family="Helvetica", size=10),
                                                     text="Height")
        self.gaussian_blur_kheight_scale = TickScale(self.gaussian_blur_frame, from_=1, to=15,
                                                     orient=tk.VERTICAL, resolution=2)
        self.gaussian_blur_kheight_scale.set(3)
        self.gaussian_blur_sigma_label = ttk.Label(self.gaussian_blur_frame, font=Font(family="Helvetica", size=10),
                                                   text="Sigma")
        self.gaussian_blur_sigma_scale = TickScale(self.gaussian_blur_frame, from_=1, to=20,
                                                   orient=tk.VERTICAL, resolution=0.5)
        self.gaussian_blur_apply_button = ttk.Button(self.gaussian_blur_frame, text="Apply gaussian blur",
                                                     command=self.model.apply_gaussian_blur)

        self.configure_gui_window()
        self.layout_gui_window()

    def configure_root_window(self):
        screen_width = self.root_window.winfo_screenwidth()
        screen_height = self.root_window.winfo_screenheight()
        x = int(screen_width / 2 - self.root_width / 2)
        y = int(screen_height / 2 - self.root_height / 2) - 15
        self.root_window.geometry("{}x{}+{}+{}".format(self.root_width, self.root_height, x, y))
        self.root_window.update_idletasks()

    def layout_root_window(self):
        self.main_frame.pack(fill="both", expand=1, padx=10, pady=10)

    def configure_main_frame(self):
        self.main_frame.grid_rowconfigure(0, weight=2)
        self.main_frame.grid_rowconfigure(1, weight=10)
        self.main_frame.grid_rowconfigure(2, weight=10)
        self.main_frame.grid_columnconfigure(0, weight=1)

    def layout_main_frame(self):
        self.menu_frame.grid(row=0, column=0, sticky="nsew")
        self.control_frame1.grid(row=1, column=0, sticky="nsew")
        self.control_frame2.grid(row=2, column=0, sticky="nsew")

    def configure_menu_frame(self):
        self.menu_frame.grid_rowconfigure(0, weight=1)
        self.menu_frame.grid_columnconfigure(0, weight=1)
        self.menu_frame.grid_columnconfigure(1, weight=2)
        self.menu_frame.grid_columnconfigure(2, weight=1)

    def layout_menu_frame(self):
        self.images_select_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        self.images_save_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 5))
        self.images_options_frame.grid(row=0, column=2, sticky="nsew", padx=(5, 0))

    def configure_control_frame1(self):
        self.control_frame1.grid_rowconfigure(0, weight=1)
        self.control_frame1.grid_rowconfigure(1, weight=10)
        self.control_frame1.grid_columnconfigure(0, weight=4)
        self.control_frame1.grid_columnconfigure(1, weight=2)

    def layout_control_frame1(self):
        self.image_threshold_label.grid(row=0, column=0, pady=(5, 0))
        self.canny_edge_label.grid(row=0, column=1, sticky="", pady=(5, 0))
        self.image_threshold_frame.grid(row=1, column=0, sticky="nsew", padx=(0, 5))
        self.canny_edge_frame.grid(row=1, column=1, sticky="nsew", padx=(5, 0))

    def configure_control_frame2(self):
        self.control_frame2.grid_rowconfigure(0, weight=1)
        self.control_frame2.grid_rowconfigure(1, weight=10)
        self.control_frame2.grid_columnconfigure(0, weight=1)
        self.control_frame2.grid_columnconfigure(1, weight=1)

    def layout_control_frame2(self):
        self.dilate_erode_label.grid(row=0, column=0, pady=(5, 0))
        self.gaussian_blur_label.grid(row=0, column=1, pady=(5, 0))
        self.dilate_erode_frame.grid(row=1, column=0, sticky="nsew", padx=(0, 5))
        self.gaussian_blur_frame.grid(row=1, column=1, sticky="nsew", padx=(5, 0))

    def configure_images_select_frame(self):
        self.images_select_frame.grid_rowconfigure(0, weight=1)
        self.images_select_frame.grid_rowconfigure(1, weight=1)
        self.images_select_frame.grid_columnconfigure(0, weight=1)

    def layout_images_select_frame(self):
        self.images_select_label.grid(row=0, column=0, sticky="", pady=(5, 0))
        self.images_select_button.grid(row=1, column=0, sticky="", pady=5)

    def configure_images_save_frame(self):
        self.images_save_frame.grid_rowconfigure(0, weight=1)
        self.images_save_frame.grid_rowconfigure(1, weight=1)
        self.images_save_frame.grid_columnconfigure(0, weight=1)

    def layout_images_save_frame(self):
        self.images_save_label.grid(row=0, column=0, sticky="", pady=(5, 0))
        self.save_as_images_button.grid(row=1, column=0, sticky="", pady=5)

    def configure_images_options_frame(self):
        self.images_options_frame.grid_rowconfigure(0, weight=1)
        self.images_options_frame.grid_rowconfigure(1, weight=1)
        self.images_options_frame.grid_columnconfigure(0, weight=1)

    def layout_images_options_frame(self):
        self.images_options_subframe1.grid(row=0, column=0, sticky="nsew")
        self.images_options_subframe2.grid(row=1, column=0, sticky="nsew")

    def configure_images_options_subframe1(self):
        self.images_options_subframe1.grid_rowconfigure(0, weight=1)
        self.images_options_subframe1.grid_columnconfigure(0, weight=1)
        self.images_options_subframe1.grid_columnconfigure(1, weight=1)
        self.images_options_subframe1.grid_columnconfigure(2, weight=1)

    def layout_images_options_subframe1(self):
        self.images_options_label.grid(row=0, column=0, padx=(30, 10))
        self.original_image_radio.grid(row=0, column=1)
        self.preprocessed_image_radio.grid(row=0, column=2, padx=(5, 30))

    def configure_images_options_subframe2(self):
        self.images_options_subframe2.grid_rowconfigure(0, weight=1)
        self.images_options_subframe2.grid_columnconfigure(0, weight=1)
        self.images_options_subframe2.grid_columnconfigure(1, weight=1)

    def layout_images_options_subframe2(self):
        self.step_back_button.grid(row=0, column=0, pady=(0, 5), padx=(0, 15), sticky="e")
        self.reset_button.grid(row=0, column=1, pady=(0, 5), padx=(15, 0), sticky="w")

    def configure_image_threshold_frame(self):
        self.image_threshold_frame.grid_rowconfigure(0, weight=1)
        self.image_threshold_frame.grid_rowconfigure(1, weight=1)
        self.image_threshold_frame.grid_rowconfigure(2, weight=1)
        self.image_threshold_frame.grid_rowconfigure(3, weight=1)
        self.image_threshold_frame.grid_rowconfigure(4, weight=1)
        self.image_threshold_frame.grid_columnconfigure(0, weight=1)
        self.image_threshold_frame.grid_columnconfigure(1, weight=1)
        self.image_threshold_frame.grid_columnconfigure(2, weight=1)
        self.image_threshold_frame.grid_columnconfigure(3, weight=1)

    def layout_image_threshold_frame(self):
        self.image_threshold_subframe.grid(row=0, column=0, columnspan=4, rowspan=2, sticky="nsew")
        self.max_value_label.grid(row=2, column=0, sticky="s")
        self.threshold_value_label.grid(row=2, column=1, sticky="s")
        self.block_size_label.grid(row=2, column=2, sticky="s")
        self.c_value_label.grid(row=2, column=3, sticky="s")
        self.max_value_scale.grid(row=3, column=0, pady=(10, 0), sticky="n")
        self.threshold_value_scale.grid(row=3, column=1, pady=(10, 0), sticky="n")
        self.block_size_scale.grid(row=3, column=2, pady=(10, 0), sticky="n")
        self.c_value_scale.grid(row=3, column=3, pady=(10, 0), sticky="n")
        self.apply_button.grid(row=4, column=0, columnspan=4, sticky="", padx=5, pady=(0, 5))

    def configure_image_threshold_subframe(self):
        self.image_threshold_subframe.grid_rowconfigure(0, weight=1)
        self.image_threshold_subframe.grid_rowconfigure(1, weight=1)
        self.image_threshold_subframe.grid_columnconfigure(0, weight=1)
        self.image_threshold_subframe.grid_columnconfigure(1, weight=1)

    def layout_image_threshold_subframe(self):
        self.method_label.grid(row=0, column=0)
        self.type_label.grid(row=1, column=0, sticky="")
        self.method_combobox.grid(row=0, column=1, sticky="w")
        self.type_combobox.grid(row=1, column=1, sticky="w")

    def configure_canny_edge_frame(self):
        self.canny_edge_frame.grid_rowconfigure(0, weight=1)
        self.canny_edge_frame.grid_rowconfigure(1, weight=1)
        self.canny_edge_frame.grid_rowconfigure(2, weight=1)
        self.canny_edge_frame.grid_rowconfigure(3, weight=1)
        self.canny_edge_frame.grid_columnconfigure(0, weight=1)
        self.canny_edge_frame.grid_columnconfigure(1, weight=1)

    def layout_canny_edge_frame(self):
        self.canny_threshold_label.grid(row=0, column=0, columnspan=2, sticky="s")
        self.canny_threshold1_label.grid(row=1, column=0, sticky="s")
        self.canny_threshold2_label.grid(row=1, column=1, sticky="s")
        self.canny_threshold1_scale.grid(row=2, column=0, pady=(5, 0), sticky="")
        self.canny_threshold2_scale.grid(row=2, column=1, pady=(5, 0), sticky="")
        self.canny_apply_button.grid(row=3, column=0, columnspan=2, sticky="", padx=5, pady=(0, 5))

    def configure_dilate_erode_frame(self):
        self.dilate_erode_frame.grid_rowconfigure(0, weight=1)
        self.dilate_erode_frame.grid_rowconfigure(1, weight=1)
        self.dilate_erode_frame.grid_rowconfigure(2, weight=1)
        self.dilate_erode_frame.grid_rowconfigure(3, weight=1)
        self.dilate_erode_frame.grid_columnconfigure(0, weight=1)
        self.dilate_erode_frame.grid_columnconfigure(1, weight=1)
        self.dilate_erode_frame.grid_columnconfigure(2, weight=1)

    def layout_dilate_erode_frame(self):
        self.dilate_erode_kernel_label.grid(row=0, column=0, columnspan=2, sticky="s")
        self.dilate_erode_kwidth_label.grid(row=1, column=0, sticky="s")
        self.dilate_erode_kheight_label.grid(row=1, column=1, sticky="s")
        self.dilate_erode_iter_label.grid(row=1, column=2, sticky="s")
        self.dilate_erode_kwidth_scale.grid(row=2, column=0, pady=(10, 0), sticky="n")
        self.dilate_erode_kheight_scale.grid(row=2, column=1, pady=(10, 0), sticky="n")
        self.dilate_erode_iter_scale.grid(row=2, column=2, pady=(10, 0), sticky="n")
        self.dilate_erode_subframe.grid(row=3, column=0, columnspan=3, sticky="nsew", padx=5, pady=(0, 5))

    def configure_dilate_erode_subframe(self):
        self.dilate_erode_subframe.grid_rowconfigure(0, weight=1)
        self.dilate_erode_subframe.grid_columnconfigure(0, weight=1)
        self.dilate_erode_subframe.grid_columnconfigure(1, weight=1)

    def layout_dilate_erode_subframe(self):
        self.erode_apply_button.grid(row=0, column=0, sticky="e", padx=5, pady=(0, 5))
        self.dilate_apply_button.grid(row=0, column=1, sticky="w", padx=5, pady=(0, 5))

    def configure_gaussian_blur_frame(self):
        self.gaussian_blur_frame.grid_rowconfigure(0, weight=1)
        self.gaussian_blur_frame.grid_rowconfigure(1, weight=1)
        self.gaussian_blur_frame.grid_rowconfigure(2, weight=1)
        self.gaussian_blur_frame.grid_rowconfigure(3, weight=1)
        self.gaussian_blur_frame.grid_columnconfigure(0, weight=1)
        self.gaussian_blur_frame.grid_columnconfigure(1, weight=1)
        self.gaussian_blur_frame.grid_columnconfigure(2, weight=1)

    def layout_gaussian_blur_frame(self):
        self.gaussian_blur_kernel_label.grid(row=0, column=0, columnspan=2, sticky="s")
        self.gaussian_blur_kwidth_label.grid(row=1, column=0, sticky="s")
        self.gaussian_blur_kheight_label.grid(row=1, column=1, sticky="s")
        self.gaussian_blur_sigma_label.grid(row=1, column=2, sticky="s")
        self.gaussian_blur_kwidth_scale.grid(row=2, column=0, pady=(10, 0), sticky="")
        self.gaussian_blur_kheight_scale.grid(row=2, column=1, pady=(10, 0), sticky="")
        self.gaussian_blur_sigma_scale.grid(row=2, column=2, pady=(10, 0), sticky="")
        self.gaussian_blur_apply_button.grid(row=3, column=0, columnspan=3, sticky="", padx=5, pady=(0, 5))

    def configure_gui_window(self):
        self.configure_root_window()
        self.configure_main_frame()
        self.configure_menu_frame()
        self.configure_control_frame1()
        self.configure_control_frame2()
        self.configure_images_select_frame()
        self.configure_images_save_frame()
        self.configure_images_options_frame()
        self.configure_images_options_subframe1()
        self.configure_images_options_subframe2()
        self.configure_image_threshold_frame()
        self.configure_image_threshold_subframe()
        self.configure_canny_edge_frame()
        self.configure_dilate_erode_frame()
        self.configure_dilate_erode_subframe()
        self.configure_gaussian_blur_frame()

    def layout_gui_window(self):
        self.layout_root_window()
        self.layout_main_frame()
        self.layout_menu_frame()
        self.layout_control_frame1()
        self.layout_control_frame2()
        self.layout_images_select_frame()
        self.layout_images_save_frame()
        self.layout_images_options_frame()
        self.layout_images_options_subframe1()
        self.layout_images_options_subframe2()
        self.layout_image_threshold_frame()
        self.layout_image_threshold_subframe()
        self.layout_canny_edge_frame()
        self.layout_dilate_erode_frame()
        self.layout_dilate_erode_subframe()
        self.layout_gaussian_blur_frame()

    def open_get_selected_images_paths_window(self):
        selected_images_paths = list(tk.filedialog.askopenfilenames(
            initialdir=os.path.join(os.path.dirname(os.path.realpath(__file__))),
            title="Select file",
            filetypes=(("jpeg files", "*.jpg"), ("all files", "*.*")))
        )
        return selected_images_paths

    def open_get_save_directory_window(self):
        selected_directory = tk.filedialog.askdirectory(
            initialdir=os.path.join(os.path.dirname(os.path.realpath(__file__))),
            title="Select directory",
        )
        return selected_directory

    def open_image_preview_window(self, width=680, height=700, name="Image"):
        cv2.namedWindow(name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow(name, width, height)

    def update_image_preview_window(self):
        if not self.model.selected_images_paths:
            return
        if cv2.getWindowProperty('Image', 0) >= 0:
            cv2.imshow("Image", self.model.active_image)
            self.root_window.after(100, self.update_image_preview_window)

    def close_gui_window(self):
        if tk.messagebox.askokcancel("Exit", "Do you really want to quit?"):
            self.root_window.destroy()

    def open_gui_window(self):
        self.root_window.protocol("WM_DELETE_WINDOW", self.close_gui_window)
        self.root_window.mainloop()
