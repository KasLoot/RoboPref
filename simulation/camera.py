"""A live canvas showing what the scene's cameras see, side by side.

Tk and Pillow only, so no dependency beyond what the project already has. The
window is driven from the simulation loop -- call `update` each step and it
redraws at its own frame rate, ignoring the rest.

    canvas = CameraCanvas(model)
    while not canvas.closed:
        mujoco.mj_step(model, data)
        canvas.update(data)
    canvas.close()
"""

import time
import tkinter as tk

import mujoco
import numpy as np
from PIL import Image, ImageTk

CAMERAS = ("third_person", "wrist_cam")


class CameraCanvas:
    """Offscreen-renders each named camera into a Tk window.

    ``height``/``width`` must fit the model's offscreen framebuffer, which the
    XML sets via ``<visual><global offwidth offheight/></visual>`` (640x480 by
    default). Rendering both feeds costs ~0.6 ms, so ``fps`` is what bounds the
    cost, not the renderer.
    """

    def __init__(self, model, cameras=CAMERAS, height=240, width=320, fps=30.0):
        max_h, max_w = model.vis.global_.offheight, model.vis.global_.offwidth
        if height > max_h or width > max_w:
            raise ValueError(f"{height}x{width} exceeds offscreen framebuffer {max_h}x{max_w}")

        self.cameras = tuple(cameras)
        self.renderer = mujoco.Renderer(model, height, width)
        self._interval = 1.0 / fps
        self._last = -np.inf
        self._closed = False

        self.root = tk.Tk()
        self.root.title("ARX L5 cameras")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self._panels = {}
        self._photos = {}  # Tk drops images that nothing references
        for column, name in enumerate(self.cameras):
            frame = tk.Frame(self.root)
            frame.grid(row=0, column=column, padx=4, pady=4)
            tk.Label(frame, text=name, font=("TkDefaultFont", 10)).pack()
            panel = tk.Label(frame)
            panel.pack()
            self._panels[name] = panel

    @property
    def closed(self):
        return self._closed

    def _on_close(self):
        self._closed = True

    def update(self, data, force=False):
        """Redraw if the frame interval has elapsed. Cheap to call every step."""
        if self._closed:
            return
        now = time.time()
        if not force and now - self._last < self._interval:
            return
        self._last = now

        for name in self.cameras:
            self.renderer.update_scene(data, camera=name)
            photo = ImageTk.PhotoImage(Image.fromarray(self.renderer.render()))
            self._photos[name] = photo
            self._panels[name].configure(image=photo)

        try:
            self.root.update()
        except tk.TclError:  # window went away between the check and the redraw
            self._closed = True

    def capture(self, data):
        """Render every camera once and return ``{name: uint8 HxWx3}``.

        For recording, independent of the window's frame rate.
        """
        frames = {}
        for name in self.cameras:
            self.renderer.update_scene(data, camera=name)
            frames[name] = self.renderer.render().copy()
        return frames

    def close(self):
        self._closed = True
        try:
            self.root.destroy()
        except tk.TclError:
            pass
        self.renderer.close()
