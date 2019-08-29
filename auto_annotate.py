import argparse
import json
import os
from types import SimpleNamespace

import cv2 as cv
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.style.use("classic")

# command line arguments #
parser = argparse.ArgumentParser(
    description='Create annotated video from images and kitti file.')

parser.add_argument("img_source", type=str, nargs="?",
                    help="Unlabelled image folder, image path, or video path")

parser.add_argument("kitti_path", type=str, nargs="?",
                    help="Path to kitti file")

parser.add_argument("-i", dest="input_type", type=str, nargs="?", default="img_folder", const="img_folder",
                    choices=["img_folder", "image", "video", "txt_paths"], help="Kind of input to the program")

parser.add_argument("-o", dest="output_type", type=str, nargs="?", default="video", const="video",
                    choices=["image", "video"], help="Kind of output from the program")

parser.add_argument("-d", dest="output_dest", type=str, nargs="?", default=".", const=".",
                    help="Destination folder for program output. Default is current working directory")

parser.add_argument("-f", dest="framerate", type=int, nargs="?", default=30, const=30,
                    help="Video framerate")

parser.add_argument("-v", dest="verbose", action="store_true",
                    help="Verbose printing")

parser.add_argument("-l", dest="logo_args", type=str, nargs="*", default=None, const=None,
                    help="Path to logo file, corner, and size")

parser.add_argument("-obj_id", dest="draw_obj_id", action="store_true", help="Draw object IDs")

parser.add_argument("-bbox", dest="draw_bbox", action="store_true", help="Draw bounding boxes")

# parser.add_argument("-g", dest="graphs", type=str, nargs="*", default=None,
#                     help="List of graphs to draw. Choices: classes, total")  # todo: add graph selection

parser.add_argument("-g", dest="draw_graphs", action="store_true",
                    help="Draw graphs")

args = parser.parse_args()

# Logo args
if args.logo_args:
    if len(args.logo_args) not in (0, 4):
        parser.error("Incorrect number of logo arguments. Required: path, corner, height, width")
    else:
        args.logo_path = args.logo_args[0]
        args.logo_corner = args.logo_args[1]
        args.logo_size = tuple(map(int, args.logo_args[2:]))

        if args.logo_corner not in ("TL", "TR", "BL", "BR"):
            parser.error("Invalid logo corner. One of: TL, TR, BL, BR")
else:
    args.logo_path = None
    args.logo_corner = None
    args.logo_size = None


###


### KITTI Reader
def read_and_clean_kitti(filename):
    # Read, filter & rename columns
    df = pd.read_csv(filename, sep=" ", header=None, )
    df = df.iloc[:, IMPORTANT_COLUMNS].rename(columns=COLUMN_MAPPINGS)

    return df


def read_kitti(filename):
    names = ["frame", "track_id", "type", "truncated", "occluded", "alpha", "bbox_left", "bbox_top", "bbox_right",
             "bbox_bottom", "height", "width", "length", "loc_x", "loc_y", "loc_z", "rot_y", "score"]

    return pd.read_csv(filename, sep=" ", header=None, names=names)


def kitti_to_shapes(df, draw_id=True, draw_bbox=True):
    # --- shape config
    cfg = CONFIG["shape_config"]
    text_font = FONTS[cfg["text_font"]]
    text_size = cfg["text_size"]
    text_line_width = cfg["text_line_width"]
    text_y_offset = cfg["text_y_offset"]
    text_x_offset = cfg["text_x_offset"]
    line_width = cfg["line_width"]
    # ---

    for i in range(len(df)):
        frame, object_id, object_type, _, _, _, xmin, ymin, xmax, ymax, _, _, _, _, _, _, _, _ = df.values[i,]
        # cant have decimal pixels for OpenCV
        xmin, ymin, xmax, ymax = map(int, [xmin, ymin, xmax, ymax])

        # Bounding box
        if draw_bbox:
            yield recursive_parse({
                "type":       "rectangle",
                "start":      (xmin, ymin),
                "end":        (xmax, ymax),
                "color":      COLS_BGRA_255[object_type],
                "line_width": line_width,
                "metadata":   {
                    "frame":       frame,
                    "object_id":   object_id,
                    "object_type": object_type,
                }
            })

        # Label
        if draw_id:
            yield recursive_parse({
                "type":       "text",
                "text":       str(object_id),
                "pos":        (xmin + text_x_offset, ymin + text_y_offset),
                "font":       text_font,
                "size":       text_size,
                "color":      COLS_BGRA_255[object_type],
                "line_width": text_line_width,
                "background": cfg["text_background"],
                "bg_color":   cfg["text_background_color"],
                "metadata":   {
                    "frame":       frame,
                    "object_id":   object_id,
                    "object_type": object_type,
                }
            })


###


### I/O Helpers
def make_folder(path):
    try:
        os.makedirs(path)
    except FileExistsError:
        # todo: handle permission errors?
        # print(f"{path} exists")
        pass


def load_image(filename):
    img = cv.imread(filename, cv.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Image not found: {filename}")

    # Mask
    if len(img.shape) == 2:
        img = np.repeat(img[:, :, np.newaxis], 3, axis=2)

    if img.shape[2] == 3:
        # rgb image, add alpha channel
        img = np.concatenate([img, np.ones((*img.shape[:2], 1)) * 255], 2)

    # set transparent to 0 on all colours
    img[img[:, :, 3] == 0] = 0

    return img


def load_images(folder):
    for file in sorted(os.listdir(folder)):
        yield load_image(os.path.join(folder, file))


def load_video(filename):
    vid = cv.VideoCapture(filename)

    while vid.isOpened():
        has_frame, img = vid.read()

        if has_frame:
            if img.shape[2] == 3:
                # rgb image, add alpha channel
                img = np.concatenate([img, np.ones((*img.shape[:2], 1)) * 255], 2)

            yield img
        else:
            break


def save_image(img, filename):
    # Ensure folder exists first
    folder = os.path.join(*os.path.split(filename)[:-1])
    make_folder(folder)

    cv.imwrite(filename, img)


def save_images(imgs, folder, filenames=None):
    """
    Saves multiple images into a folder

    imgs: list of image arrays
    folder: path to folder
    filenames: list of image filenames
    """

    if filenames:
        filenames = [os.path.join(folder, f) for f in filenames]

    else:
        filenames = [os.path.join(folder,
                                  "{:0>5d}.png".format(i)) for i in range(len(imgs))]

    for i, (img, filename) in enumerate(zip(imgs, filenames)):
        save_image(img, filename)

        try:
            args.callback(i)
        except AttributeError:
            pass


def load_json(filename):
    with open(filename) as f:
        return json.load(f)


def save_json(js, filename):
    with open(filename, "w") as f:
        json.dump(js, f)


def split_filename(filename):
    """
    Splits a filename into its path (and name), and its file extension
    eg. "./folder1/folder2/foo.jpg" -> "./folder1/folder2/foo", "jpg"

    """
    parts = filename.split(".")

    if len(parts) < 2:
        return filename, ""

    return ".".join(parts[:-1]), parts[-1]


###

### Image Manipulation

def join_imgs(main, *others, side="R"):
    """
    Joins images side by side into a single image

    :param main: Main image. Other image will be scaled to fit this
    :param others: Other image(s)
    :param side: Side to attach other image. TBLR
    :return: Joined image
    """

    others_resized = []

    if side in "LR":
        for o in others:
            scale = main.shape[0] / o.shape[0]
            new_shape = main.shape[0], int(o.shape[1] * scale)
            others_resized.append(cv.resize(o, new_shape[::-1]))

        axis = 1

    else:
        for o in others:
            scale = main.shape[1] / o.shape[1]
            new_shape = int(o.shape[0] * scale), main.shape[1]
            others_resized.append(cv.resize(o, new_shape[::-1]))

        axis = 0

    if side in "TL":
        return np.concatenate([*others_resized[::-1], main], axis=axis)

    else:
        return np.concatenate([main, *others_resized], axis=axis)


def join_imgs_grid(img_grid):
    """
    Joins images in a grid

    :param img_grid: 2D array of images
    :return: Single joined image
    """

    rows = (join_imgs(*row, side="R") for row in img_grid)
    return join_imgs(*rows, side="B")


def overlay_image(bg_img, fg_img, fg_alpha=1):
    """
    overlays one image ontop of another. Works with transparent images.

    fg_alpha: weighting of the foreground to background images

    """

    return np.where(fg_img[:, :, 3:4] != 0,
                    #                                                 semi-transparent area
                    fg_img * fg_alpha + bg_img * \
                    (1 - fg_alpha + (1 - fg_img[:, :, 3:4] / 255) * fg_alpha),
                    bg_img
                    )


def scale_and_pad(img, new_size, new_pos, pad_size=None):
    """
    scales down an image, and pads it with transparent background

    new_size: dimensions the image is scaled down to (must be smaller than pad_size)
    new_pos: where to put the small image
    pad_size: size of new image (including padding). Default is input image size
    """

    if pad_size is None:
        pad_size = img.shape

    new_img = np.zeros((*pad_size, 4))
    small = cv.resize(img, new_size[::-1])

    ## Clipping out of bounds
    # top
    if new_pos[0] < 0:
        small = small[-new_pos[0]:, :]
        new_pos = (0, new_pos[1])
    # left
    if new_pos[1] < 0:
        small = small[:, -new_pos[1]:]
        new_pos = (new_pos[0], 0)
    # bottom
    if new_pos[0] + new_size[0] > pad_size[0]:
        small = small[:pad_size[0] - new_pos[0], :]
    # right
    if new_pos[1] + new_size[1] > pad_size[1]:
        small = small[:, :pad_size[1] - new_pos[1]]

    new_img[new_pos[0]:new_pos[0] + small.shape[0], new_pos[1]:new_pos[1] + small.shape[1], :] = small

    return new_img


def mask_to_rgba(mask, class_vals=None, colors=None):
    """
    Turns a greyscale mask with classes, represented by each pixel value, into an rgba mask with each class colored.

    :param mask: shape: (height, width)
    :param class_vals: list of classes
    :param colors: list of rbga color tuples
    :return: image
    """

    if class_vals is None:
        class_vals = np.unique(mask)[1:]  # dont include 0

    if colors is None:
        colors = [tuple(int(val * 255) for val in matplotlib.colors.to_rgba(f"C{cls}")) for cls in class_vals]

    img = np.zeros((*mask.shape, 4))
    for cls, col in zip(class_vals, colors):
        img = np.where(
            mask[:, :, np.newaxis] == cls,
            col,
            img
        )

    return img


def draw_shape(img, shape, in_place=False):
    """
    draws a shape on the given image

    img: numpy array of image
    shape: shape parameter dictionary
    in_place: draw on the image in-place

    """
    # for k, v in shape.items():
    #     print(k, v, type(v))

    if not in_place:
        img = np.copy(img)

    if shape["type"] == "rectangle":
        cv.rectangle(img, shape["start"], shape["end"],
                     shape["color"], shape["line_width"])

    elif shape["type"] == "circle":
        cv.circle(img, shape["centre"], shape["radius"],
                  shape["color"], shape["line_width"])

    elif shape["type"] == "poly":
        cv.polylines(img, np.int32([shape["points"]]), shape["closed"],
                     shape["color"], shape["line_width"])

    elif shape["type"] == "text":
        # Background box for text
        if shape["background"]:
            width, height = cv.getTextSize(shape["text"], shape["font"], shape["size"], shape["line_width"])[0]
            start = shape["pos"][0] - 1, shape["pos"][1] + 1  # 1 px padding
            end = (shape["pos"][0] + width + 1, shape["pos"][1] - height - 1)  # +width, -height, weird

            cv.rectangle(img, start, end, shape["bg_color"], cv.FILLED)

        cv.putText(img, shape["text"], shape["pos"], shape["font"], shape["size"],
                   shape["color"], shape["line_width"], cv.LINE_AA)

    elif shape["type"] == "overlayed_img":
        ov_img = scale_and_pad(shape["img"], shape["size"], shape["pos"], img.shape[:2])
        img = overlay_image(img, ov_img, shape["opacity"])

    return img


def annotate_image(img, shapes):
    annotated = img
    for shape in shapes:
        annotated = draw_shape(annotated, shape)

    return annotated


def annotate_images(img_generator, shapes_list):
    for i, (img, shapes) in enumerate(zip(img_generator, shapes_list)):
        yield annotate_image(img, shapes)


def logo_to_shape(img, logo_size, bg_size, corner="TL"):
    # todo: Display logo at start/end of the video.
    #       Maybe have them independent of each other

    if corner == "TL":
        pos = (0, 0)
    elif corner == "TR":
        pos = (0, bg_size[1] - logo_size[1])
    elif corner == "BL":
        pos = (bg_size[0] - logo_size[0], 0)
    elif corner == "BR":
        pos = (bg_size[0] - logo_size[0], bg_size[1] - logo_size[1])
    else:
        pos = (0, 0)

    return {
        "type":    "overlayed_img",
        "img":     img,
        "size":    logo_size,
        "pos":     pos,
        "opacity": 1,
    }


def draw_logo(img, logo_img, logo_size, logo_corner):
    shape = logo_to_shape(logo_img, logo_size, img.shape[:2], logo_corner)
    return draw_shape(img, shape)


###


### Plotting

def fig_to_img(fig):
    """
    Turns a matplotlib figure into a BGRA image

    adapted from: http://www.icare.univ-lille1.fr/tutorials/convert_a_matplotlib_figure


    :param fig: matplotlib figure
    :return: 3D numpy array of image
    """

    # draw the renderer
    fig.canvas.draw()

    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8).reshape(h, w, 4)

    # ARGB -> BGRA
    return buf[:, :, ::-1]


def rolling_line_animation(data, xrange, title=None):
    """
    Creates images for a line graph animation.


    :param data: pd.Series of height values
    :param xrange: Range of values to be in the plot at once
    :param title: Plot title
    :return: list of images
    """

    fig = plt.figure()
    ax = plt.axes(xlim=(0, xrange - 1), ylim=(data.min(), data.max()), xticks=[], title=title)
    line, = ax.plot([], [])


    def init():
        line.set_data([], [])


    def animate(i):
        interval = data[max(0, i - xrange + 1):i + 1]
        interval = np.concatenate([[np.nan] * (xrange - len(interval)), interval.values])
        line.set_data(range(len(interval)), interval)


    return plot_to_imgs(fig, init, animate, len(data))


def rolling_bar_animation(data, window, title=None):
    """
    Creates images for a bar graph animation.

    :param data: pd.DataFrame with a category, timestamp (or similar) to group on. Each row is an instance
    :param window: Number of timesteps for rolling calculation
    :param title: Plot title
    :return: list of images
    """

    category = "type"
    timestamp = "frame"

    # Count
    grouped = data.groupby([timestamp, category]).count().iloc[:, 0]
    # Crosstab
    pivot = grouped.reset_index().pivot_table(values="track_id", index="frame", columns=category, fill_value=0)
    # Rolling window
    rolling = pivot.rolling(window, min_periods=1).mean()

    classes = pivot.columns

    fig = plt.figure()
    plt.subplot(ylim=(0, 10), title=title)
    bars = plt.bar(range(len(classes)), [0] * len(classes), tick_label=classes,
                   color=[COLS_RGBA_1[c] for c in classes])


    def init():
        for bar in bars:
            bar.set_height(0)


    def animate(i):
        for cls, bar in zip(classes, bars):
            bar.set_height(rolling.loc[i, cls])


    return plot_to_imgs(fig, init, animate, len(pivot))


def birdseye_animation(data, title=None):
    """
    Creates images for a birds eye view of objects

    :param data:
    :param title:
    :return:
    """

    data = data[data["height"] > 0]
    base_img = np.ones((
        int((data["loc_x"].max() - data["loc_x"].min() + data["width"].max()) * 10),  # 0.1m precision
        int((data["loc_z"].max() - data["loc_z"].min() + data["length"].max()) * 10),
        4)) * 255

    fig = plt.figure()
    plt.subplot(title=title)


    def init():
        pass


    def animate(i):
        img = np.copy(base_img)

        for i in data[data["frame"] == i].index:
            x0 = int((data.loc[i, "loc_x"] - data.loc[i, "width"] / 2 - data["loc_x"].min()) * 10)
            y0 = int((data.loc[i, "loc_z"] - data.loc[i, "length"] / 2 - data["loc_z"].min()) * 10)
            x1 = int((data.loc[i, "loc_x"] + data.loc[i, "width"] / 2 - data["loc_x"].min()) * 10)
            y1 = int((data.loc[i, "loc_z"] + data.loc[i, "length"] / 2 - data["loc_z"].min()) * 10)

            cv.rectangle(img, (x0, y0), (x1, y1), COLS_BGRA_255[data.loc[i, "type"]], -1)

        plt.imshow(img.astype(int)[::-1, :, 2::-1])


    return plot_to_imgs(fig, init, animate, len(data.groupby("frame")))


def plot_to_imgs(fig, init_func, animate_func, frames):
    """
    Takes a plot structure and generates images over timesteps

    :param fig: plot figure
    :param init_func: function called once to initialise the plot
    :param animate_func: function called at every timestep to update the plot
    :param frames: number of timesteps
    :return: generator of images
    """

    init_func()

    for i in range(frames):
        animate_func(i)
        yield fig_to_img(fig)


###


def recursive_parse(obj):
    """
    Turns nested lists into tuples.
    openCV cannot take lists [] for some arguments instead of tuples ()
    """

    if type(obj) in (list, tuple):
        return tuple(recursive_parse(o) for o in obj)

    if type(obj) in (dict,):
        return {k: recursive_parse(v) for k, v in obj.items()}

    # numpy arrays
    if hasattr(obj, "dtype"):
        if np.prod(obj.shape) == 1:
            return obj.item()

        return recursive_parse(obj.tolist())

    return obj


def input_to_images(path, kind="img_folder"):
    """
    Handles input for the program

    path: path to file/folder
    kind: "img_folder", "image", "video", "txt_paths
    """

    if kind == "img_folder":
        imgs = load_images(path)
        filenames = (split_filename(p)[0] for p in sorted(os.listdir(path)))

    elif kind == "image":
        imgs = iter([load_image(path)])
        filenames = iter([split_filename(os.path.split(path)[-1])[0]])

    elif kind == "video":
        imgs = load_video(path)
        nframes = int(cv.VideoCapture(path).get(cv.CAP_PROP_FRAME_COUNT))
        filenames = ("{:0>5d}".format(i) for i in range(nframes))

    elif kind == "txt_paths":
        # todo: allow mixed list of files/folders
        # todo: allow indexed lists of files for out of order paths

        with open(path) as f:
            paths = [line.strip() for line in f]
            filenames = [split_filename(os.path.split(p)[-1])[0] for p in paths]

        imgs = map(load_image, paths)
        filenames = iter(filenames)

    else:
        imgs, filenames = None, None

    return imgs, filenames


def configure_colors(kitti_df, color_by="type"):
    """
    # Sets up colour space for drawing annotations

    kitti_df: DataFrame of kitti information
    color_by: The column that determines the colours of the drawings
        - options: "object_type", "object_id"

    """

    global COLS_RGBA_1, COLS_BGRA_255

    unique_obj_types = kitti_df[color_by].unique()

    COLS_RGBA_1 = {}
    COLS_BGRA_255 = {}
    for i, t in enumerate(unique_obj_types):
        # Ci for matplotlib default colours
        rgba_1 = matplotlib.colors.to_rgba(f"C{i}")
        COLS_RGBA_1[t] = rgba_1
        # Convert to [0,255) range
        rgba_255 = tuple(int(val * 255) for val in rgba_1)
        bgra_255 = (*rgba_255[2::-1], rgba_255[3])
        COLS_BGRA_255[t] = bgra_255


def images_to_video(imgs_folder=None, imgs=None, dest_folder=None, framerate=15, video_filename=None):
    """
    Takes a folder/list of images and creates a video

    imgs_folder: path to source folder
    imgs: list of source images. If this is given, imgs_folder is ignored
    dest_folder: path to destination folder
    framerate: number of frames per second
    video_filename: name of output video, without extension


    todo: make other formats/encoders available

    """
    assert imgs_folder is not None or imgs is not None
    assert imgs_folder is not None or video_filename is not None

    if dest_folder is not None:
        # Use given dest_folder
        pass
    elif imgs_folder is not None:
        # Use images parent folder
        dest_folder = os.path.join(*os.path.split(imgs_folder)[:-1])
    else:
        # Current working directory
        dest_folder = ""

    if imgs is None:
        imgs = [cv.imread(os.path.join(imgs_folder, filename))
                for filename in sorted(os.listdir(imgs_folder))]

    if video_filename is None:
        video_filename = os.path.split(imgs_folder)[-1] + ".mp4"
    else:
        video_filename += ".mp4"

    # Check folder exists
    if not os.path.isdir(dest_folder):
        raise NotADirectoryError(f"'{dest_folder}' does not exist")

    video_path = os.path.join(dest_folder, video_filename)

    writer = None
    for i, img in enumerate(imgs):
        img = np.uint8(img)[:, :, :3]

        if writer is None:
            height, width, channels = img.shape
            writer = cv.VideoWriter(video_path, cv.VideoWriter_fourcc(*"mp4v"), framerate, (width, height))

        writer.write(img)

        if args.verbose:
            print(f"Frame: {i}")

        try:
            args.callback(i)
        except AttributeError:
            pass

    writer.release()


def do_track(namespace, kitti):
    """
    Fully sets up one video track with annotations

    :param namespace: Settings for this track
    :param kitti: kitti info
    :return: annotated images for this track
    """

    t_imgs, _ = input_to_images(namespace.img_source, namespace.input_type)
    if namespace.input_type == "image":
        # infinite generator for static image
        im = next(t_imgs)
        t_imgs = (np.copy(im) for _ in iter(int, 1))

    t_shapes = (kitti_to_shapes(group, draw_id=namespace.draw_obj_id, draw_bbox=namespace.draw_bbox)
                for _, group in kitti.groupby("frame"))
    t_annotated = annotate_images(t_imgs, t_shapes)

    if namespace.draw_logo:
        t_logo = load_image(namespace.logo_path)
        t_annotated = (draw_logo(img, t_logo, namespace.logo_size, namespace.logo_corner) for img in t_annotated)

    return t_annotated


def full_run(kitti_path, output_type, output_dest, framerate=15, draw_graphs=False, callback=None, tracks=()):
    """
    full run of program from start to finish.
    See commandline help for argument details
    Everything with generators for better memory handling
    todo: come up with a good name for this

    :param kitti_path: String path to kitti file
    :param output_type: String output type: "video", "images"
    :param output_dest: String path to output folder
    :param framerate: int video framerate (for output_type=="video")
    :param draw_graphs: boolean draw graphs on video
    :param callback: function(i) called after each frame is processed
    :param tracks: Namespace with properties:
            draw_bbox: boolean draw bounding boxes on track
            draw_obj_id: boolean draw object IDs on track
            img_source: String path to image/video/folder
            input_type: String input type: "image", "video", "img_folder"
            draw_logo: boolean draw logo on track
            logo_path: String path to logo
            logo_corner: String corner to draw logo: "TL", "TR", "BL", "BR"
            logo_size: int tuple (height, width) size of logo

    """
    # setup callback
    if callback:
        args.callback = callback

    if len(tracks) == 0:
        raise Exception("Must have at least 1 track.")

    kitti = read_kitti(kitti_path)
    configure_colors(kitti)

    # Annotate tracks
    track_imgs = [do_track(t, kitti) for t in tracks]
    annotated_imgs = (join_imgs(*t_img, side="B") for t_img in zip(*track_imgs))

    # Take 1st non-image track for filenames, or number of frames as a last resort
    main_track = next((t for t in tracks if t.input_type != "image"), None)
    if main_track:
        _, filenames = input_to_images(main_track.img_source, main_track.input_type)
    else:
        filenames = ("{:0>5d}".format(i) for i in range(len(kitti.groupby("frame"))))

    # Add graphs on side
    if draw_graphs:
        obj_class_imgs = rolling_bar_animation(kitti, framerate, "Objects by class")
        obj_count_imgs = rolling_line_animation(kitti.groupby("frame").count()["track_id"], framerate,
                                                "Total Objects")
        birdseye_imgs = birdseye_animation(kitti, "Birds Eye View")

        # Stack graphs
        graphs = (join_imgs_grid([[cls], [cnt], [b]]) for cls, cnt, b in zip(
            obj_class_imgs, obj_count_imgs, birdseye_imgs))

        # Attach to tracks
        annotated_imgs = (join_imgs(img, graph, side="R") for img, graph in
                          zip(annotated_imgs, graphs))

    # Save
    if output_type == "video":
        filename = split_filename(os.path.split(kitti_path)[-1])[0]

        images_to_video(imgs=annotated_imgs, dest_folder=output_dest,
                        framerate=framerate, video_filename=filename)

    else:
        filenames = [f + ".png" for f in filenames]
        save_images(annotated_imgs, output_dest, filenames)


def full_run_cmd():
    """
    full run of program from start to finish using command line arguments.
    Only able to do 1 track with this.
    """

    full_run(args.kitti_path, args.output_type, args.output_dest, args.framerate, args.draw_graphs, callback=None,
             tracks=[
                 SimpleNamespace(
                     draw_bbox=args.draw_bbox, draw_logo=(args.logo_path is not None),
                     draw_obj_id=args.draw_obj_id,
                     img_source=args.img_source,
                     input_type=args.input_type, logo_corner=args.logo_corner,
                     logo_path=args.logo_path,
                     logo_size=args.logo_size
                 )
             ])


# CONFIG
CONFIG = load_json("./auto_annotate_config.json")
COLS_BGRA_255 = None
COLS_RGBA_1 = None


# Fonts
class FontDict(dict):
    def __missing__(self, key):
        return cv.FONT_HERSHEY_SIMPLEX


FONTS = FontDict({
    "HERSHEY_SIMPLEX":        cv.FONT_HERSHEY_SIMPLEX,
    "HERSHEY_PLAIN":          cv.FONT_HERSHEY_PLAIN,
    "HERSHEY_DUPLEX":         cv.FONT_HERSHEY_DUPLEX,
    "HERSHEY_COMPLEX":        cv.FONT_HERSHEY_COMPLEX,
    "HERSHEY_TRIPLEX":        cv.FONT_HERSHEY_TRIPLEX,
    "HERSHEY_COMPLEX_SMALL":  cv.FONT_HERSHEY_COMPLEX_SMALL,
    "HERSHEY_SCRIPT_SIMPLEX": cv.FONT_HERSHEY_SCRIPT_SIMPLEX,
    "HERSHEY_SCRIPT_COMPLEX": cv.FONT_HERSHEY_SCRIPT_COMPLEX,
})

# kitti
COLUMN_MAPPINGS = dict(CONFIG["kitti_columns_map"])
IMPORTANT_COLUMNS = sorted(list(COLUMN_MAPPINGS.keys()))

# Logo
CORNERS = {
    "TL": (0, 0),
    "TR": (0, 1),
    "BL": (1, 0),
    "BR": (1, 1),
}

###


if __name__ == '__main__':
    full_run_cmd()
