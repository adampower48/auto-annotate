import os
import subprocess
import sys
import threading
import tkinter as tk
import traceback
from tkinter import ttk
from tkinter.filedialog import askopenfilename, askdirectory
from types import SimpleNamespace

import cv2 as cv
import numpy as np
import pandas as pd

from auto_annotate import full_run, read_and_clean_kitti, read_kitti

SOURCE_TYPES = ["folder of images", "image", "video", "txt file"]
OUTPUT_TYPES = ["video", "images"]
LOGO_CORNERS = ["TL", "TR", "BL", "BR"]


class App(threading.Thread):
    def __init__(self, master):
        threading.Thread.__init__(self)

        # Setup gui frames
        columns = [tk.Frame(master), tk.Frame(master)]
        columns[0].grid(row=0, column=0)
        columns[1].grid(row=0, column=1)

        required_frame = tk.Frame(columns[0])
        required_frame.grid(row=0, column=0)
        optional_frame = tk.Frame(columns[0])
        optional_frame.grid(row=1, column=0, sticky="W")

        kitti_frame = tk.Frame(columns[1])
        kitti_frame.grid(row=0, column=0)
        preview_frame = tk.Frame(columns[1])
        preview_frame.grid(row=1, column=0)
        start_frame = tk.Frame(columns[1])
        start_frame.grid(row=2, column=0)

        # required files
        self.kitti_file_label = None
        self.out_file_label = None
        self.kitti_browse_button = None
        self.out_browse_button = None
        self.output_type_var = None
        self.output_type_menu = None
        self.image_select_frame_label = None
        self.image_select_frame_entry = None

        self.start_button = None
        self.process_label = None
        self.progress_bar = None

        # kitti summary
        self.kitti_frame = None
        self.kitti_grid = None
        self.kitti_grid2 = None
        self.kitti_frames_text = None
        self.kitti_classes_text = None
        self.kitti_class_nunique_text = None
        self.kitti_class_nframes_text = None
        self.kitti_ids_text = None

        # options
        self.video_framerate_label = None
        self.video_framerate_entry = None

        self.graphs_checkbox_var = None
        self.graphs_checkbox = None

        self.setup_file_select(required_frame)
        self.setup_options_frame(optional_frame)
        self.setup_kitti_preview(kitti_frame)
        self.setup_image_preview(preview_frame)
        self.setup_start_frame(start_frame)

        # temp
        self.tracks = None
        list_frame = tk.Frame(columns[0])
        list_frame.grid(row=2, sticky="W")
        self.setup_tracks(list_frame)


    def setup_file_select(self, parent):
        # Static labels
        tk.Label(parent, text="KITTI Path:", fg="red").grid(row=1, column=0)
        tk.Label(parent, text="Output Path:", fg="red").grid(row=2, column=0)

        self.kitti_file_label = tk.Label(parent, text="file/path/here", width=50)
        self.kitti_file_label.grid(row=1, column=1)

        self.out_file_label = tk.Label(parent, text="file/path/here", width=50)
        self.out_file_label.grid(row=2, column=1)

        self.kitti_browse_button = tk.Button(
            parent, text="Browse", command=self.open_kitti)
        self.kitti_browse_button.grid(row=1, column=2)

        self.out_browse_button = tk.Button(parent, text="Browse",
                                           command=lambda: self.ask_open_filename(self.out_file_label, kind="folder"))
        self.out_browse_button.grid(row=2, column=2)

        # Output Type
        self.output_type_var = tk.StringVar(parent)
        self.output_type_var.set(OUTPUT_TYPES[0])

        self.output_type_menu = tk.OptionMenu(
            parent, self.output_type_var, *OUTPUT_TYPES, command=self.toggle_framerate_entry)
        self.output_type_menu.grid(row=2, column=3)

        # Single image select frame
        self.image_select_frame_label = tk.Label(parent, text="Frame:")

        self.image_select_frame_entry = tk.Entry(parent, width=6, validate="key",
                                                 validatecommand=self.val_cmd(parent, self.validate_integer))
        self.image_select_frame_entry.insert(0, "0")


    def setup_options_frame(self, parent):
        # Header
        tk.Label(parent, text="Options:", padx=10).grid(row=0, column=0, sticky="W")

        # Video framerate
        video_fr_frame = tk.Frame(parent)
        video_fr_frame.grid(row=1, column=0, sticky="W")
        self.video_framerate_label = tk.Label(video_fr_frame, text="Video framerate:")
        self.video_framerate_label.grid(row=0, column=0)

        self.video_framerate_entry = tk.Entry(video_fr_frame, width=4, validate="key",
                                              validatecommand=self.val_cmd(video_fr_frame, self.validate_integer))
        self.video_framerate_entry.grid(row=0, column=1)
        self.video_framerate_entry.insert(0, "15")

        # Graphs
        self.graphs_checkbox_var = tk.IntVar()
        self.graphs_checkbox = tk.Checkbutton(parent, text="Graphs",
                                              variable=self.graphs_checkbox_var)
        self.graphs_checkbox.grid(row=5, column=0, sticky="W")


    def setup_kitti_preview(self, parent):
        kitti_grid2 = tk.Frame(parent)
        kitti_grid2.grid(row=1, column=0)

        # Headings
        tk.Label(kitti_grid2, text="Frames", padx=10).grid(row=0, column=0)
        tk.Label(kitti_grid2, text="Object ids", padx=10).grid(row=1, column=0)
        tk.Label(parent, text="Object Classes", padx=10).grid(row=0, column=1)
        tk.Label(parent, text="Unique IDs", padx=10).grid(row=0, column=2)
        tk.Label(parent, text="Instances", padx=10).grid(row=0, column=3)

        # Details
        self.kitti_frames_text = tk.Label(kitti_grid2, text="#")
        self.kitti_frames_text.grid(row=0, column=1)

        self.kitti_ids_text = tk.Label(kitti_grid2, text="#")
        self.kitti_ids_text.grid(row=1, column=1)

        self.kitti_classes_text = t = tk.Text(
            parent, height=10, width=20)
        t.grid(row=1, column=1)
        t.config(state=tk.DISABLED)

        self.kitti_class_nunique_text = t = tk.Text(parent, height=10, width=8)
        t.grid(row=1, column=2)
        t.config(state=tk.DISABLED)

        self.kitti_class_nframes_text = t = tk.Text(parent, height=10, width=8)
        t.grid(row=1, column=3)
        t.config(state=tk.DISABLED)


    def setup_image_preview(self, parent):
        # todo: image preview
        pass


    def setup_start_frame(self, parent):
        self.start_button = tk.Button(
            parent, text="Start", command=self.start_annotate)
        self.start_button.grid(row=0, column=0)

        self.process_label = tk.Label(parent, text="", fg="green", padx=10)
        self.process_label.grid(row=0, column=1)

        self.progress_bar = ttk.Progressbar(parent, length=100, orient=tk.HORIZONTAL, mode="determinate")
        self.progress_bar.grid(row=0, column=2)


    def setup_tracks(self, parent):

        self.tracks = []

        list_frame = tk.Frame(parent)
        list_frame.grid(row=0, sticky="W")

        # Add track button
        add_button = tk.Button(parent, text="Add Track", command=lambda: add_track(len(self.tracks)))
        add_button.grid(row=1)


        def add_track(i):
            image_frame = tk.Frame(list_frame)
            image_frame.grid(row=i, sticky="W")
            image_frame.list_index = i
            image_frame.nw = SimpleNamespace()  # For accessing values later

            # Separator
            ttk.Separator(image_frame, orient="horizontal").grid(row=0, columnspan=5, sticky="EW")

            # Delete
            tk.Button(image_frame, text="Delete", command=lambda: remove_track(image_frame.list_index)) \
                .grid(row=1, sticky="W")

            # Files
            file_frame = tk.Frame(image_frame)
            file_frame.grid(row=2, columnspan=5, sticky="W")

            tk.Label(file_frame, text="Image src:", fg="red").grid(row=0, column=0)

            image_frame.nw.input_type_var = tk.StringVar()
            image_frame.nw.input_type_var.set(SOURCE_TYPES[0])

            input_type_menu = tk.OptionMenu(file_frame, image_frame.nw.input_type_var, *SOURCE_TYPES)
            input_type_menu.grid(row=0, column=1)

            image_frame.nw.src_label = tk.Label(file_frame, text="")
            image_frame.nw.src_label.grid(row=0, column=3)
            tk.Button(file_frame, text="Browse",
                      command=lambda: self.src_file_dialog(image_frame.nw.input_type_var, image_frame.nw.src_label)
                      ).grid(row=0, column=2)

            ## todo: optional kitti file

            # Options
            options_frame = tk.Frame(image_frame)
            options_frame.grid(row=3, columnspan=5, sticky="W")

            image_frame.nw.bbox_var = tk.IntVar()
            bbox_check = tk.Checkbutton(options_frame, text="Bounding Box", variable=image_frame.nw.bbox_var)
            bbox_check.grid(row=2, column=0, sticky="W")

            # image_frame.nw.graph_var = tk.IntVar()
            # graph_check = tk.Checkbutton(options_frame, text="Graphs", variable=image_frame.nw.graph_var)
            # graph_check.grid(row=2, column=1, sticky="W")

            image_frame.nw.obj_id_var = tk.IntVar()
            obj_id_check = tk.Checkbutton(options_frame, text="Object IDs", variable=image_frame.nw.obj_id_var)
            obj_id_check.grid(row=2, column=2, sticky="W")

            ## Logo
            logo_frame = tk.Frame(image_frame)
            logo_frame.grid(row=4, sticky="W")
            image_frame.nw.logo_var = tk.IntVar()
            logo_check = tk.Checkbutton(logo_frame, text="Logo", variable=image_frame.nw.logo_var,
                                        command=lambda: self.toggle_frame_visibility(logo_toggle_frame,
                                                                                     dict(row=0, column=1),
                                                                                     image_frame.nw.logo_var), )
            logo_check.grid(row=0, column=0, sticky="NW")

            logo_toggle_frame = tk.Frame(logo_frame)

            image_frame.nw.logo_file_label = tk.Label(logo_toggle_frame, text="")
            image_frame.nw.logo_file_label.grid(row=0, column=1, columnspan=5)
            tk.Button(logo_toggle_frame, text="Browse",
                      command=lambda: self.ask_open_filename(image_frame.nw.logo_file_label, kind="file")).grid(
                row=0,
                column=0)

            logo_corner_label = tk.Label(logo_toggle_frame, text="Corner:")
            logo_corner_label.grid(row=1, column=0)
            image_frame.nw.logo_corner_var = tk.StringVar()
            image_frame.nw.logo_corner_var.set(LOGO_CORNERS[0])
            logo_corner_menu = tk.OptionMenu(logo_toggle_frame, image_frame.nw.logo_corner_var, *LOGO_CORNERS)
            logo_corner_menu.grid(row=1, column=1)

            logo_height_label = tk.Label(logo_toggle_frame, text="Height:")
            logo_height_label.grid(row=1, column=2)
            image_frame.nw.logo_height_entry = tk.Entry(logo_toggle_frame, width=6, validate="key",
                                                        validatecommand=self.val_cmd(logo_toggle_frame,
                                                                                     self.validate_integer))
            image_frame.nw.logo_height_entry.insert(0, "100")
            image_frame.nw.logo_height_entry.grid(row=1, column=3)

            logo_width_label = tk.Label(logo_toggle_frame, text="Width:")
            logo_width_label.grid(row=1, column=4)
            image_frame.nw.logo_width_entry = tk.Entry(logo_toggle_frame, width=6, validate="key",
                                                       validatecommand=self.val_cmd(logo_toggle_frame,
                                                                                    self.validate_integer))
            image_frame.nw.logo_width_entry.insert(0, "200")
            image_frame.nw.logo_width_entry.grid(row=1, column=5)

            self.tracks.append(image_frame)


        def remove_track(i):
            self.tracks.pop(i).destroy()

            # Reassign list positions
            for track in self.tracks[i:]:
                track.list_index -= 1
                track.grid(row=track.list_index)

            # Force frame resize
            temp_frame = ttk.Separator(parent)
            temp_frame.grid()
            temp_frame.destroy()


    def toggle_framerate_entry(self, value):
        # Shows/hides the framerate entry box when output type is video/not
        label_cell = {"row": 0, "column": 0}
        entry_cell = {"row": 0, "column": 1}

        if value == "video":
            self.video_framerate_label.grid(**label_cell)
            self.video_framerate_entry.grid(**entry_cell)
        else:
            self.video_framerate_label.grid_forget()
            self.video_framerate_entry.grid_forget()


    def fill_kitti_preview(self, filename):
        try:
            kitti_df = read_kitti(filename)
        except FileNotFoundError:
            print(f"{filename} not found.", file=sys.stderr)
            return
        except pd.errors.ParserError:
            print("Error parsing file", file=sys.stderr)
            return

        frames = len(kitti_df["frame"].unique())
        obj_ids = np.sort(kitti_df["track_id"].unique())

        # Aggregate calcs
        obj_classes_nunique = kitti_df.groupby("type")["track_id"].nunique()
        obj_classes_nframes = kitti_df["type"].value_counts()
        obj_classes = pd.merge(obj_classes_nunique, obj_classes_nframes, left_index=True, right_index=True).rename(
            columns={"track_id": "unique_ids", "type": "total_occurrences"})

        self.kitti_frames_text.config(text=str(frames))
        self.kitti_ids_text.config(text=str(len(obj_ids)))
        self.set_text(self.kitti_classes_text, "\n".join(obj_classes_nunique.index))
        self.set_text(self.kitti_class_nunique_text, "\n".join(obj_classes["unique_ids"].values.astype(str)))
        self.set_text(self.kitti_class_nframes_text, "\n".join(obj_classes["total_occurrences"].values.astype(str)))


    def open_kitti(self):
        filename = askopenfilename(title="Choose a file", filetypes=(("Text files", "*.txt"),))

        if filename:
            self.kitti_file_label.config(text=filename)
            self.fill_kitti_preview(filename)


    def start_annotate(self):
        input_type_map = {
            "image":            "image",
            "folder of images": "img_folder",
            "video":            "video",
            "txt file":         "txt_paths",
        }

        output_type_map = {
            "images": "image",
            "video":  "video",
        }

        kwargs = dict(
            kitti_path=self.kitti_file_label.config()["text"][4],
            output_type=output_type_map[self.output_type_var.get()],
            output_dest=self.out_file_label.config()["text"][4],
            # img_frame=int(self.image_select_frame_entry.get()),
            draw_graphs=self.graphs_checkbox_var.get(),
            callback=lambda i: self.progress_bar.config(value=i + 1)
        )

        if self.output_type_var.get() == "video":
            kwargs["framerate"] = int(self.video_framerate_entry.get())

        tracks = []
        for track in self.tracks:
            tracks.append(SimpleNamespace(
                img_source=track.nw.src_label.config()["text"][4],
                input_type=input_type_map[track.nw.input_type_var.get()],
                draw_bbox=track.nw.bbox_var.get(),
                draw_obj_id=track.nw.obj_id_var.get(),
                draw_logo=track.nw.logo_var.get(),
                logo_path=track.nw.logo_file_label.config()["text"][4],
                logo_corner=track.nw.logo_corner_var.get(),
                logo_size=(int(track.nw.logo_height_entry.get()), int(track.nw.logo_width_entry.get())),
            ))
        kwargs["tracks"] = tracks

        threading.Thread(target=self.run_process_in_background_direct, args=(kwargs,), ).start()


    def run_process_in_background_direct(self, kwargs):
        self.start_button.config(state=tk.DISABLED)
        self.process_label.config(text="")

        try:
            # get number of frames for progress bar
            frames = [int(self.kitti_frames_text.config()["text"][4])]
            for t in kwargs["tracks"]:
                if t.input_type == "img_folder":
                    frames.append(len(os.listdir(t.img_source)))
                elif t.input_type == "txt_paths":
                    with open(t.img_source) as f:
                        frames.append(len([None for _ in f]))
                elif t.input_type == "video":
                    frames.append(int(cv.VideoCapture(t.img_source).get(cv.CAP_PROP_FRAME_COUNT)))

            nframes = min(frames, default=1)

        except FileNotFoundError:
            nframes = 1

        self.progress_bar.config(maximum=nframes)

        try:
            print(kwargs)
            # debug
            # kwargs = {'kitti_path':  '/home/adam/Documents/vidas/data/KITTI/test/0005.txt',
            #           'output_type': 'video', 'output_dest': '/home/adam/Documents/vidas/data/KITTI/test',
            #           'draw_graphs': 1,
            #           'callback':    lambda i: self.progress_bar.config(value=i + 1),
            #           'framerate':   15,
            #           'tracks':      [
            #               SimpleNamespace(draw_bbox=1, draw_logo=0, draw_obj_id=0,
            #                               img_source='/home/adam/Documents/vidas/data/KITTI/mots/0005',
            #                               input_type='img_folder', logo_corner='TL', logo_path='',
            #                               logo_size=(100, 200)),
            #               SimpleNamespace(draw_bbox=1, draw_logo=1, draw_obj_id=1,
            #                               img_source='/home/adam/Documents/vidas/data/KITTI/test/Images/000009.png',
            #                               input_type='image', logo_corner='TL',
            #                               logo_path='/home/adam/Documents/vidas/data/insight_logo.png',
            #                               logo_size=(100, 200)),
            #               SimpleNamespace(draw_bbox=0, draw_logo=0, draw_obj_id=1,
            #                               img_source='/home/adam/Documents/vidas/data/KITTI/semantic_rgb',
            #                               input_type='img_folder', logo_corner='TL',
            #                               logo_path='', logo_size=(100, 200)),
            #               SimpleNamespace(draw_bbox=0, draw_logo=1, draw_obj_id=0,
            #                               img_source='/home/adam/Documents/vidas/data/KITTI/test/Images',
            #                               input_type='img_folder', logo_corner='TR',
            #                               logo_path='/home/adam/Documents/vidas/data/frog.jpg',
            #                               logo_size=(100, 100))]
            #           }

            full_run(**kwargs)
        except:  # GUI keeps running if there is an error in annotate program
            traceback.print_exc(file=sys.stderr)
            self.process_label.config(text=f"ERROR", fg="red")
        else:
            self.process_label.config(text="Finished", fg="green")
        finally:
            self.start_button.config(state=tk.NORMAL)


    def src_file_dialog(self, type_var, file_label):
        if type_var.get() == "folder of images":
            self.ask_open_filename(file_label, "folder")
        elif type_var.get() == "txt file":
            self.ask_open_filename(file_label, "file", filetypes=(("Text files", "*.txt"),))
        else:
            self.ask_open_filename(file_label, "file")


    @staticmethod
    def val_cmd(frame, cmd):
        # wrapper for validation commands
        return frame.register(cmd), '%d', '%i', '%P', '%s', '%S', '%v', '%V', '%W'


    @staticmethod
    def validate_integer(action, index, value_if_allowed, prior_value, text, validation_type, trigger_type,
                         widget_name):
        try:
            int(text)
            return True
        except ValueError:
            return False


    @staticmethod
    def set_text(text_object, text):
        # Sets the text of a tk.Text object
        text_object.config(state=tk.NORMAL)
        text_object.delete(1.0, tk.END)
        text_object.insert(tk.END, text)
        text_object.config(state=tk.DISABLED)


    @staticmethod
    def ask_open_filename(label, kind="file", filetypes=None):
        # type_var: determines the type of dialog to open. "file", "folder"

        if kind == "file":
            if filetypes:
                filename = askopenfilename(title="Choose a file", filetypes=filetypes)
            else:
                filename = askopenfilename(title="Choose a file")
        elif kind == "folder":
            filename = askdirectory(title="Choose a folder")
        else:
            filename = None

        if filename:
            label.config(text=filename)


    @staticmethod
    def toggle_frame_visibility(frame, grid_options, state_var):
        state = state_var.get()

        if state:
            frame.grid(**grid_options)
        else:
            frame.grid_forget()


root = tk.Tk()
root.title("Auto Annotate")

app = App(root)

root.mainloop()

# todo: Add preview image
# todo: Add ability to change layout of images, graphs, etc.
