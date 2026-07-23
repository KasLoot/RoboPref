from __future__ import annotations

import os
import platform
import tempfile
from pathlib import Path
from typing import Any, Iterable

from PIL import Image


WIDTH = 640
HEIGHT = 480
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
ARCHIVED_SCENE_XML = (
    REPOSITORY_ROOT
    / "_old_1"
    / "simulation"
    / "assets"
    / "robots"
    / "arx_l5"
    / "scene.xml"
)

COLOURS = {
    "red": (0.85, 0.10, 0.10, 1.0),
    "green": (0.10, 0.70, 0.20, 1.0),
    "blue": (0.10, 0.25, 0.85, 1.0),
    "yellow": (0.95, 0.85, 0.10, 1.0),
    "orange": (0.95, 0.45, 0.08, 1.0),
    "brown": (0.50, 0.23, 0.08, 1.0),
    "black": (0.04, 0.05, 0.06, 1.0),
    "white": (0.92, 0.92, 0.90, 1.0),
    "grey": (0.45, 0.48, 0.52, 1.0),
    "gray": (0.45, 0.48, 0.52, 1.0),
    "silver": (0.65, 0.68, 0.72, 1.0),
    "purple": (0.55, 0.20, 0.72, 1.0),
}


def _prepare_gl_backend() -> None:
    system = platform.system()
    if system == "Windows":
        os.environ.setdefault("MUJOCO_GL", "wgl")
    elif system == "Linux":
        os.environ.setdefault("MUJOCO_GL", "egl")
    elif system == "Darwin":
        os.environ.setdefault("MUJOCO_GL", "cgl")


def _attributes(state: Any) -> dict[str, Any]:
    raw = getattr(state, "attributes", ())
    return dict(raw) if not isinstance(raw, dict) else dict(raw)


def _state_values(state: Any) -> dict[str, Any]:
    raw = getattr(state, "state", ())
    return dict(raw) if not isinstance(raw, dict) else dict(raw)


def _look_at_xyaxes(position: Iterable[float], target: Iterable[float]) -> list[float]:
    import numpy as np

    position_array = np.asarray(tuple(position), dtype=float)
    target_array = np.asarray(tuple(target), dtype=float)
    forward = target_array - position_array
    forward /= np.linalg.norm(forward)
    camera_z = -forward
    camera_x = np.cross([0.0, 0.0, 1.0], camera_z)
    camera_x /= np.linalg.norm(camera_x)
    camera_y = np.cross(camera_z, camera_x)
    return [*camera_x, *camera_y]


def _add_box(
    spec: Any,
    *,
    name: str,
    position: tuple[float, float, float],
    half_size: tuple[float, float, float],
    rgba: tuple[float, float, float, float],
    quaternion_wxyz: tuple[float, float, float, float] | None = None,
) -> Any:
    import mujoco

    body = spec.worldbody.add_body(
        name=name,
        pos=list(position),
        **({"quat": list(quaternion_wxyz)} if quaternion_wxyz else {}),
    )
    body.add_geom(
        name=f"{name}_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=list(half_size),
        rgba=list(rgba),
        contype=0,
        conaffinity=0,
    )
    return body


def _add_workspace(spec: Any, family: str) -> None:
    import mujoco

    table = spec.worldbody.add_body(name="benchmark_table", pos=[0.27, 0.0, -0.018])
    table.add_geom(
        name="benchmark_table_top",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.19, 0.225, 0.018],
        rgba=[0.52, 0.35, 0.20, 1.0],
        friction=[1.0, 0.01, 0.001],
    )

    if family in {"block_stack", "category_sort"}:
        left_colour = (
            [0.92, 0.92, 0.92, 1.0]
            if family == "block_stack"
            else [0.78, 0.86, 0.96, 1.0]
        )
        right_colour = (
            [0.07, 0.07, 0.07, 1.0]
            if family == "block_stack"
            else [0.96, 0.86, 0.70, 1.0]
        )
        for name, y, colour in (
            ("left_region", -0.125, left_colour),
            ("right_region", 0.125, right_colour),
        ):
            region = spec.worldbody.add_body(name=name, pos=[0.27, y, 0.002])
            region.add_geom(
                name=f"{name}_geom",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[0.135, 0.09, 0.002],
                rgba=colour,
                contype=0,
                conaffinity=0,
            )
    else:
        placemat = spec.worldbody.add_body(
            name="placemat", pos=[0.27, 0.0, 0.002]
        )
        placemat.add_geom(
            name="placemat_geom",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[0.15, 0.19, 0.002],
            rgba=[0.82, 0.76, 0.66, 1.0],
            contype=0,
            conaffinity=0,
        )


def _add_object(spec: Any, state: Any) -> None:
    import mujoco

    attributes = _attributes(state)
    state_values = _state_values(state)
    object_type = str(getattr(state, "object_type", "object")).lower()
    object_id = str(getattr(state, "object_id"))
    position = tuple(float(value) for value in getattr(state, "position_m"))
    size = tuple(float(value) for value in getattr(state, "size_m"))
    xyzw = tuple(float(value) for value in getattr(state, "orientation_xyzw"))
    quaternion_wxyz = (xyzw[3], xyzw[0], xyzw[1], xyzw[2])
    colour_name = str(
        attributes.get("color", attributes.get("colour", "grey"))
    ).lower()
    rgba = COLOURS.get(colour_name, COLOURS["grey"])

    body = spec.worldbody.add_body(
        name=object_id,
        pos=list(position),
        quat=list(quaternion_wxyz),
    )

    if object_type in {"plate", "cup", "bowl"}:
        radius = max(size[0], size[1]) / 2
        body.add_geom(
            name=f"{object_id}_geom",
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=[radius, size[2] / 2],
            rgba=list(rgba),
            contype=0,
            conaffinity=0,
        )
        if object_type == "plate":
            body.add_geom(
                name=f"{object_id}_centre",
                type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                size=[radius * 0.62, size[2] * 0.56],
                pos=[0, 0, size[2] * 0.52],
                rgba=[0.70, 0.76, 0.80, 1.0],
                contype=0,
                conaffinity=0,
            )
        return

    body.add_geom(
        name=f"{object_id}_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[dimension / 2 for dimension in size],
        rgba=list(rgba),
        contype=0,
        conaffinity=0,
    )

    if object_type in {"phone", "tablet", "laptop", "electronic"}:
        body.add_geom(
            name=f"{object_id}_screen",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[
                max(0.002, size[0] * 0.39),
                max(0.002, size[1] * 0.39),
                0.0005,
            ],
            pos=[0.0, 0.0, size[2] / 2 + 0.0006],
            rgba=[0.08, 0.30, 0.42, 1.0],
            contype=0,
            conaffinity=0,
        )
        if object_type == "laptop":
            body.add_geom(
                name=f"{object_id}_hinge",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[
                    max(0.002, size[0] * 0.42),
                    max(0.001, size[1] * 0.035),
                    max(0.001, size[2] * 0.12),
                ],
                pos=[0.0, size[1] * 0.42, size[2] / 2 + 0.001],
                rgba=[0.18, 0.20, 0.23, 1.0],
                contype=0,
                conaffinity=0,
            )
    elif object_type in {"book", "notebook", "magazine"}:
        body.add_geom(
            name=f"{object_id}_pages",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[
                max(0.002, size[0] * 0.47),
                max(0.002, size[1] * 0.44),
                max(0.001, size[2] * 0.32),
            ],
            rgba=[0.90, 0.84, 0.68, 1.0],
            contype=0,
            conaffinity=0,
        )
        if object_type == "magazine":
            body.add_geom(
                name=f"{object_id}_cover_stripe",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[
                    max(0.002, size[0] * 0.32),
                    max(0.001, size[1] * 0.055),
                    0.0007,
                ],
                pos=[0.0, -size[1] * 0.25, size[2] / 2 + 0.0008],
                rgba=[0.76, 0.18, 0.15, 1.0],
                contype=0,
                conaffinity=0,
            )
    elif object_type == "knife":
        # The near-miss endpoint differs only by a 180-degree knife rotation.
        # Model the local +Y blade and -Y handle asymmetrically so that direction
        # is observable in the rendered evidence (a centred box is rotationally
        # symmetric and made success/near-miss images identical).
        body.add_geom(
            name=f"{object_id}_blade",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[
                max(0.001, size[0] * 0.50),
                max(0.004, size[1] * 0.25),
                size[2] * 0.56,
            ],
            pos=[0.0, size[1] * 0.25, size[2] * 0.10],
            rgba=[0.88, 0.91, 0.94, 1.0],
            contype=0,
            conaffinity=0,
        )
        body.add_geom(
            name=f"{object_id}_handle",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[
                max(0.001, size[0] * 0.31),
                max(0.004, size[1] * 0.22),
                size[2] * 0.60,
            ],
            pos=[0.0, -size[1] * 0.28, size[2] * 0.12],
            rgba=[0.20, 0.23, 0.27, 1.0],
            contype=0,
            conaffinity=0,
        )
    elif object_type in {"fork", "spoon", "utensil"}:
        # A bright handle marking makes the thin primitive legible from above.
        body.add_geom(
            name=f"{object_id}_handle",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[
                max(0.001, size[0] * 0.14),
                max(0.001, size[1] * 0.46),
                size[2] * 0.55,
            ],
            rgba=[0.78, 0.80, 0.83, 1.0],
            contype=0,
            conaffinity=0,
        )

    if state_values.get("unstable") or state_values.get("unsafe"):
        body.add_geom(
            name=f"{object_id}_warning_marker",
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.004, 0.004, 0.004],
            pos=[0.0, 0.0, size[2] * 0.8],
            rgba=[0.95, 0.55, 0.02, 1.0],
            contype=0,
            conaffinity=0,
        )


def _build_model(specification: Any, objects: Iterable[Any]) -> Any:
    _prepare_gl_backend()
    import mujoco

    if not ARCHIVED_SCENE_XML.is_file():
        raise FileNotFoundError(
            f"Archived ARX L5 scene is missing: {ARCHIVED_SCENE_XML}"
        )
    model_spec = mujoco.MjSpec.from_file(str(ARCHIVED_SCENE_XML))
    camera = next(
        camera for camera in model_spec.cameras if camera.name == "third_person"
    )
    camera.pos = [0.83, 0.0, 0.58]
    camera.alt.xyaxes = _look_at_xyaxes(camera.pos, [0.27, 0.0, 0.025])
    try:
        material = next(
            material for material in model_spec.materials if material.name == "black_mat"
        )
        material.rgba = [0.52, 0.52, 0.54, 1.0]
    except StopIteration:
        pass

    _add_workspace(model_spec, str(specification.family))
    for state in objects:
        _add_object(model_spec, state)
    return model_spec.compile()


def _render(specification: Any, objects: Iterable[Any], *, occluded: bool) -> Image.Image:
    _prepare_gl_backend()
    import mujoco

    model = _build_model(specification, objects)
    data = mujoco.MjData(model)
    home_key = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_KEY,
        "home",
    )
    if home_key >= 0:
        mujoco.mj_resetDataKeyframe(model, data, home_key)
    else:
        mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)
    try:
        renderer.update_scene(data, camera="third_person")
        pixels = renderer.render().copy()
    finally:
        renderer.close()
    image = Image.fromarray(pixels, mode="RGB")
    if occluded:
        from PIL import ImageDraw

        draw = ImageDraw.Draw(image, "RGBA")
        draw.rounded_rectangle(
            (48, 138, 592, 460),
            radius=28,
            fill=(112, 116, 122, 255),
            outline=(72, 76, 82, 255),
            width=5,
        )
    return image


def _atomic_save(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.stem}.", suffix=".png"
    )
    os.close(file_descriptor)
    temporary_path = Path(temporary_name)
    try:
        image.save(temporary_path, format="PNG", optimize=False)
        os.replace(temporary_path, path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def render_pair_mujoco(
    specification: Any, initial_path: str | Path, final_path: str | Path
) -> None:
    """Render privileged endpoint states in the archived ARX L5 MuJoCo scene.

    This is a snapshot renderer, not a claim that the arm executed the transition.
    The archived controller remains useful for future trajectory rollouts.
    """

    initial_path = Path(initial_path)
    final_path = Path(final_path)
    initial = _render(specification, specification.initial_objects, occluded=False)
    final = _render(
        specification,
        specification.final_objects,
        occluded=getattr(specification, "observation_status", "") == "occluded",
    )
    _atomic_save(initial, initial_path)
    _atomic_save(final, final_path)
