# KITTI Auto Annotate & Visualisation
Visualises KITTI annotations for vehicle detection & tracking.

# Usage
Can be used through the GUI, command line, or as a package.
#### GUI
Run `python annotate_gui.py`
- Requires at least "KITTI Path", "Output Path" & 1 track
- Click "Add Track" to add a video/image feed
    - Each track requires at least "Image Src"
    
#### Command Line
Run `python auto_annotate.py --help` for argument info

#### Package
Example:
```
from auto_annotate import full_run
from types import SimpleNamespace

kwargs = {
    'kitti_path':  '/home/adam/Documents/KITTI/test/0005.txt', 
    'output_type': 'video',
    'output_dest': '/home/adam/Documents/', 
    'draw_graphs': 0, 
    'callback': None, 
    'framerate': 15,
    'tracks':      [
        SimpleNamespace(draw_bbox=0, draw_logo=0, draw_obj_id=0,
                        img_source='/home/adam/Documents/0010.mp4',
                        input_type='video', logo_corner='TL', logo_path='', logo_size=(100, 200))
    ]
}

full_run(**kwargs)

```