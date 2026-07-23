from __future__ import annotations

import os
import platform
import tempfile
from pathlib import Path
from typing import Any, Iterable

from PIL import Image


WIDTH = 640
HEIGHT = 480
THIRD_PERSON_CAMERA_POSITION = (0.97, 0.0, 0.66)
THIRD_PERSON_CAMERA_TARGET = (0.27, 0.0, 0.035)
THIRD_PERSON_CAMERA_FOVY = 41.0
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

MATERIAL = {
    "wood": "benchmark_oak_wood",
    "yellow_mat": "benchmark_woven_mustard",
    "teal_mat": "benchmark_teal_mat",
    "cream_mat": "benchmark_cream_mat",
    "charcoal_mat": "benchmark_charcoal_mat",
    "mat_border": "benchmark_mat_border",
    "ceramic": "benchmark_ivory_ceramic",
    "ceramic_shadow": "benchmark_ceramic_shadow",
    "metal": "benchmark_brushed_steel",
    "dark_metal": "benchmark_dark_metal",
    "screen": "benchmark_screen_glass",
    "paper": "benchmark_paper",
    "book": "benchmark_book_cover",
    "magazine": "benchmark_magazine_cover",
    "laptop": "benchmark_laptop_shell",
    "tablet": "benchmark_tablet_shell",
    "napkin": "benchmark_napkin_cloth",
    "red": "benchmark_block_red",
    "green": "benchmark_block_green",
    "blue": "benchmark_block_blue",
    "generic": "benchmark_generic",
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


def _wood_texture_data(width: int = 128, height: int = 64) -> list[int]:
    """Create deterministic oak grain without requiring an external image asset."""

    import math

    pixels: list[int] = []
    for y in range(height):
        for x in range(width):
            warped_x = x + 5.0 * math.sin(y * 0.21) + 2.0 * math.sin(y * 0.63)
            grain = (
                0.58 * math.sin(warped_x * 0.22)
                + 0.28 * math.sin(warped_x * 0.61 + y * 0.05)
                + 0.14 * math.sin(warped_x * 1.37)
            )
            plank = -10.0 if y % 21 in {0, 1} else 0.0
            knot_distance = ((x - 91.0) / 13.0) ** 2 + ((y - 36.0) / 7.0) ** 2
            knot = -24.0 * math.exp(-knot_distance)
            variation = grain * 14.0 + plank + knot
            pixels.extend(
                (
                    max(0, min(255, int(126 + variation))),
                    max(0, min(255, int(70 + variation * 0.55))),
                    max(0, min(255, int(30 + variation * 0.28))),
                )
            )
    return pixels


def _woven_texture_data(width: int = 48, height: int = 48) -> list[int]:
    """Create a subtle mustard textile weave for the placemat."""

    pixels: list[int] = []
    for y in range(height):
        for x in range(width):
            warp = 10 if x % 4 == 0 else -3 if x % 4 == 2 else 2
            weft = 8 if y % 4 == 0 else -2 if y % 4 == 2 else 1
            pixels.extend(
                (
                    max(0, min(255, 204 + warp + weft)),
                    max(0, min(255, 159 + int(warp * 0.45) + weft)),
                    max(0, min(255, 44 + int(weft * 0.25))),
                )
            )
    return pixels


def _add_benchmark_materials(spec: Any) -> None:
    import mujoco

    oak_texture = spec.add_texture(
        name="benchmark_oak_texture",
        type=mujoco.mjtTexture.mjTEXTURE_2D,
        width=128,
        height=64,
        nchannel=3,
    )
    oak_texture.data = bytes(_wood_texture_data())
    woven_texture = spec.add_texture(
        name="benchmark_woven_texture",
        type=mujoco.mjtTexture.mjTEXTURE_2D,
        width=48,
        height=48,
        nchannel=3,
    )
    # Assigning bytes after construction works across the supported MuJoCo
    # MjSpec bindings (3.3.7+); older bindings reject list data in add_texture.
    woven_texture.data = bytes(_woven_texture_data())

    def material(
        name: str,
        rgba: tuple[float, float, float, float],
        *,
        roughness: float,
        specular: float,
        shininess: float,
        reflectance: float,
        metallic: float = 0.0,
        texture: str | None = None,
        texrepeat: tuple[float, float] = (1.0, 1.0),
    ) -> None:
        value = spec.add_material(
            name=name,
            rgba=list(rgba),
            roughness=roughness,
            specular=specular,
            shininess=shininess,
            reflectance=reflectance,
            metallic=metallic,
        )
        if texture:
            # MuJoCo texture role 1 is RGB/albedo; role 0 is the generic
            # user-defined slot and is not sampled by the classic renderer.
            value.textures[1] = texture
            value.texuniform = 1
            value.texrepeat = list(texrepeat)

    material(
        MATERIAL["wood"],
        (0.95, 0.84, 0.70, 1.0),
        texture="benchmark_oak_texture",
        texrepeat=(3.2, 5.0),
        roughness=0.42,
        specular=0.32,
        shininess=0.36,
        reflectance=0.12,
    )
    material(
        MATERIAL["yellow_mat"],
        (0.92, 0.68, 0.16, 1.0),
        texture="benchmark_woven_texture",
        texrepeat=(8.0, 10.0),
        roughness=0.74,
        specular=0.13,
        shininess=0.12,
        reflectance=0.025,
    )
    material(
        MATERIAL["teal_mat"],
        (0.055, 0.28, 0.31, 1.0),
        roughness=0.68,
        specular=0.16,
        shininess=0.15,
        reflectance=0.035,
    )
    material(
        MATERIAL["cream_mat"],
        (0.69, 0.62, 0.48, 1.0),
        roughness=0.63,
        specular=0.18,
        shininess=0.18,
        reflectance=0.04,
    )
    material(
        MATERIAL["charcoal_mat"],
        (0.075, 0.085, 0.095, 1.0),
        roughness=0.58,
        specular=0.22,
        shininess=0.22,
        reflectance=0.055,
    )
    material(
        MATERIAL["mat_border"],
        (0.075, 0.055, 0.035, 1.0),
        roughness=0.48,
        specular=0.22,
        shininess=0.20,
        reflectance=0.06,
    )
    material(
        MATERIAL["ceramic"],
        (0.83, 0.87, 0.88, 1.0),
        roughness=0.20,
        specular=0.72,
        shininess=0.82,
        reflectance=0.18,
    )
    material(
        MATERIAL["ceramic_shadow"],
        (0.25, 0.32, 0.34, 1.0),
        roughness=0.30,
        specular=0.50,
        shininess=0.62,
        reflectance=0.10,
    )
    material(
        MATERIAL["metal"],
        (0.62, 0.68, 0.72, 1.0),
        roughness=0.16,
        specular=0.92,
        shininess=0.92,
        reflectance=0.28,
        metallic=0.86,
    )
    material(
        MATERIAL["dark_metal"],
        (0.085, 0.105, 0.13, 1.0),
        roughness=0.25,
        specular=0.75,
        shininess=0.76,
        reflectance=0.18,
        metallic=0.72,
    )
    material(
        MATERIAL["screen"],
        (0.025, 0.16, 0.22, 1.0),
        roughness=0.08,
        specular=0.95,
        shininess=0.98,
        reflectance=0.24,
        metallic=0.12,
    )
    material(
        MATERIAL["paper"],
        (0.76, 0.70, 0.56, 1.0),
        roughness=0.84,
        specular=0.08,
        shininess=0.06,
        reflectance=0.01,
    )
    material(
        MATERIAL["book"],
        (0.38, 0.075, 0.055, 1.0),
        roughness=0.48,
        specular=0.24,
        shininess=0.24,
        reflectance=0.045,
    )
    material(
        MATERIAL["magazine"],
        (0.13, 0.20, 0.48, 1.0),
        roughness=0.42,
        specular=0.30,
        shininess=0.32,
        reflectance=0.055,
    )
    material(
        MATERIAL["laptop"],
        (0.15, 0.18, 0.22, 1.0),
        roughness=0.24,
        specular=0.72,
        shininess=0.74,
        reflectance=0.16,
        metallic=0.72,
    )
    material(
        MATERIAL["tablet"],
        (0.055, 0.065, 0.08, 1.0),
        roughness=0.22,
        specular=0.72,
        shininess=0.78,
        reflectance=0.16,
        metallic=0.45,
    )
    material(
        MATERIAL["napkin"],
        (0.08, 0.31, 0.32, 1.0),
        roughness=0.88,
        specular=0.05,
        shininess=0.04,
        reflectance=0.0,
    )
    for colour, rgba in (
        ("red", (0.70, 0.035, 0.025, 1.0)),
        ("green", (0.025, 0.48, 0.13, 1.0)),
        ("blue", (0.025, 0.18, 0.72, 1.0)),
    ):
        material(
            MATERIAL[colour],
            rgba,
            roughness=0.30,
            specular=0.48,
            shininess=0.48,
            reflectance=0.085,
        )
    material(
        MATERIAL["generic"],
        (0.31, 0.36, 0.40, 1.0),
        roughness=0.42,
        specular=0.36,
        shininess=0.38,
        reflectance=0.075,
    )


def _configure_visuals(spec: Any) -> None:
    import mujoco

    spec.visual.headlight.active = 1
    spec.visual.headlight.ambient = [0.068, 0.073, 0.082]
    spec.visual.headlight.diffuse = [0.23, 0.24, 0.26]
    spec.visual.headlight.specular = [0.32, 0.34, 0.38]
    spec.visual.quality.shadowsize = 4096
    spec.visual.quality.offsamples = 4
    spec.visual.quality.numslices = 48
    spec.visual.quality.numstacks = 24
    spec.visual.quality.numquads = 8
    spec.visual.map.shadowscale = 0.85
    spec.visual.map.shadowclip = 1.6
    spec.visual.global_.fovy = 41.0
    spec.visual.rgba.haze = [0.055, 0.065, 0.08, 1.0]

    for light in spec.lights:
        light.active = 0
    spec.worldbody.add_light(
        name="benchmark_key_light",
        type=mujoco.mjtLightType.mjLIGHT_SPOT,
        pos=[0.72, -0.48, 0.88],
        dir=[-0.43, 0.48, -0.82],
        castshadow=1,
        bulbradius=0.055,
        cutoff=55.0,
        exponent=8.0,
        ambient=[0.015, 0.012, 0.010],
        diffuse=[0.96, 0.89, 0.80],
        specular=[0.96, 0.89, 0.80],
    )
    spec.worldbody.add_light(
        name="benchmark_fill_light",
        type=mujoco.mjtLightType.mjLIGHT_SPOT,
        pos=[0.08, 0.52, 0.62],
        dir=[0.25, -0.68, -0.69],
        castshadow=0,
        bulbradius=0.10,
        cutoff=62.0,
        exponent=5.0,
        ambient=[0.012, 0.016, 0.022],
        diffuse=[0.34, 0.45, 0.59],
        specular=[0.29, 0.39, 0.52],
    )
    spec.worldbody.add_light(
        name="benchmark_rim_light",
        type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
        pos=[-0.35, -0.15, 0.72],
        dir=[0.42, 0.12, -0.90],
        castshadow=0,
        ambient=[0.0, 0.0, 0.0],
        diffuse=[0.16, 0.19, 0.24],
        specular=[0.30, 0.34, 0.42],
    )

    for texture in spec.textures:
        if texture.name == "groundplane":
            texture.rgb1 = [0.17, 0.19, 0.22]
            texture.rgb2 = [0.105, 0.12, 0.14]
            texture.markrgb = [0.29, 0.32, 0.36]
        elif not texture.name:
            texture.rgb1 = [0.18, 0.22, 0.28]
            texture.rgb2 = [0.025, 0.03, 0.045]
    for material in spec.materials:
        if material.name == "groundplane":
            material.roughness = 0.54
            material.specular = 0.26
            material.shininess = 0.28
            material.reflectance = 0.10
    for geom in spec.geoms:
        if geom.name == "floor":
            geom.pos = [0.0, 0.0, -0.275]


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

    table = spec.worldbody.add_body(
        name="benchmark_table",
        pos=[0.27, 0.0, -0.022],
    )
    table.add_geom(
        name="benchmark_table_top",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.225, 0.26, 0.022],
        material=MATERIAL["wood"],
        friction=[1.0, 0.01, 0.001],
    )
    table.add_geom(
        name="benchmark_table_surface",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        pos=[0.0, 0.0, 0.0222],
        size=[0.222, 0.257, 0.001],
        material=MATERIAL["wood"],
        contype=0,
        conaffinity=0,
    )
    pedestal = spec.worldbody.add_body(
        name="benchmark_robot_pedestal",
        pos=[0.0, 0.0, -0.137],
    )
    pedestal.add_geom(
        name="benchmark_robot_pedestal_geom",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        # MjSpec releases before 3.10 require all geom size assignments to
        # provide the full mjtNum[3], even when the geom uses fewer entries.
        size=[0.082, 0.137, 0.0],
        material=MATERIAL["dark_metal"],
        contype=0,
        conaffinity=0,
    )
    for index, (x, y) in enumerate(
        (
            (-0.185, -0.215),
            (-0.185, 0.215),
            (0.185, -0.215),
            (0.185, 0.215),
        ),
        start=1,
    ):
        table.add_geom(
            name=f"benchmark_table_leg_{index}",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[x, y, -0.135],
            size=[0.018, 0.018, 0.115],
            material=MATERIAL["wood"],
            contype=0,
            conaffinity=0,
        )

    if family in {"block_stack", "category_sort"}:
        left_material = (
            MATERIAL["cream_mat"]
            if family == "block_stack"
            else MATERIAL["yellow_mat"]
        )
        right_material = (
            MATERIAL["charcoal_mat"]
            if family == "block_stack"
            else MATERIAL["teal_mat"]
        )
        for name, y, surface_material in (
            ("left_region", -0.125, left_material),
            ("right_region", 0.125, right_material),
        ):
            region = spec.worldbody.add_body(
                name=name,
                # Keep the visible textile surface close to z=0 so thin
                # magazines and blocks do not appear sunk into the mat.
                pos=[0.27, y, 0.00025],
            )
            region.add_geom(
                name=f"{name}_border",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[0.139, 0.094, 0.0005],
                material=MATERIAL["mat_border"],
                contype=0,
                conaffinity=0,
            )
            region.add_geom(
                name=f"{name}_geom",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=[0.0, 0.0, 0.00055],
                size=[0.134, 0.089, 0.00018],
                material=surface_material,
                contype=0,
                conaffinity=0,
            )
    else:
        placemat = spec.worldbody.add_body(
            name="placemat",
            pos=[0.27, 0.0, 0.00025],
        )
        placemat.add_geom(
            name="placemat_border",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[0.158, 0.198, 0.0005],
            material=MATERIAL["mat_border"],
            contype=0,
            conaffinity=0,
        )
        placemat.add_geom(
            name="placemat_geom",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[0.0, 0.0, 0.00055],
            size=[0.153, 0.193, 0.00018],
            material=MATERIAL["yellow_mat"],
            contype=0,
            conaffinity=0,
        )


def _add_object(spec: Any, state: Any) -> None:
    import mujoco

    attributes = _attributes(state)
    object_type = str(getattr(state, "object_type", "object")).lower()
    object_id = str(getattr(state, "object_id"))
    position = tuple(float(value) for value in getattr(state, "position_m"))
    size = tuple(float(value) for value in getattr(state, "size_m"))
    xyzw = tuple(float(value) for value in getattr(state, "orientation_xyzw"))
    quaternion_wxyz = (xyzw[3], xyzw[0], xyzw[1], xyzw[2])
    colour_name = str(
        attributes.get("color", attributes.get("colour", "grey"))
    ).lower()

    body = spec.worldbody.add_body(
        name=object_id,
        pos=list(position),
        quat=list(quaternion_wxyz),
    )

    if object_type == "plate":
        radius = max(size[0], size[1]) / 2
        body.add_geom(
            name=f"{object_id}_rim",
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=[radius, size[2] / 2, 0.0],
            material=MATERIAL["ceramic_shadow"],
            contype=0,
            conaffinity=0,
        )
        body.add_geom(
            name=f"{object_id}_well",
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=[radius * 0.79, max(0.001, size[2] * 0.18), 0.0],
            pos=[0.0, 0.0, size[2] * 0.38],
            material=MATERIAL["ceramic"],
            contype=0,
            conaffinity=0,
        )
    elif object_type in {"cup", "bowl"}:
        radius = max(size[0], size[1]) / 2
        body.add_geom(
            name=f"{object_id}_body",
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=[radius, size[2] / 2, 0.0],
            material=MATERIAL["ceramic"],
            contype=0,
            conaffinity=0,
        )
        body.add_geom(
            name=f"{object_id}_rim",
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=[
                radius * 1.03,
                max(0.001, size[2] * 0.035),
                0.0,
            ],
            pos=[0.0, 0.0, size[2] * 0.50],
            material=MATERIAL["ceramic"],
            contype=0,
            conaffinity=0,
        )
        body.add_geom(
            name=f"{object_id}_opening",
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=[
                radius * 0.73,
                max(0.0007, size[2] * 0.018),
                0.0,
            ],
            pos=[0.0, 0.0, size[2] * 0.54],
            material=MATERIAL["ceramic_shadow"],
            contype=0,
            conaffinity=0,
        )
        if object_type == "cup":
            handle_x = radius * 1.52
            handle_radius = max(0.0020, radius * 0.13)
            lower_z = -size[2] * 0.22
            upper_z = size[2] * 0.22
            body.add_geom(
                name=f"{object_id}_handle_outer",
                type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                fromto=[
                    radius * 0.88,
                    0.0,
                    lower_z,
                    handle_x,
                    0.0,
                    lower_z,
                ],
                size=[handle_radius, 0.0, 0.0],
                material=MATERIAL["ceramic"],
                contype=0,
                conaffinity=0,
            )
            body.add_geom(
                name=f"{object_id}_handle_curve",
                type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                fromto=[
                    handle_x,
                    0.0,
                    lower_z,
                    handle_x,
                    0.0,
                    upper_z,
                ],
                size=[handle_radius, 0.0, 0.0],
                material=MATERIAL["ceramic"],
                contype=0,
                conaffinity=0,
            )
            body.add_geom(
                name=f"{object_id}_handle_inner",
                type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                fromto=[
                    handle_x,
                    0.0,
                    upper_z,
                    radius * 0.88,
                    0.0,
                    upper_z,
                ],
                size=[handle_radius, 0.0, 0.0],
                material=MATERIAL["ceramic"],
                contype=0,
                conaffinity=0,
            )
    elif object_type in {"cube", "block"}:
        block_material = MATERIAL.get(colour_name, MATERIAL["generic"])
        body.add_geom(
            name=f"{object_id}_geom",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[dimension / 2 for dimension in size],
            material=block_material,
            contype=0,
            conaffinity=0,
        )
        highlight = tuple(
            min(1.0, channel * 1.18 + 0.035)
            for channel in COLOURS.get(colour_name, COLOURS["grey"])[:3]
        )
        body.add_geom(
            name=f"{object_id}_top_face",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[size[0] * 0.42, size[1] * 0.42, 0.00045],
            pos=[0.0, 0.0, size[2] / 2 + 0.00045],
            rgba=[*highlight, 1.0],
            contype=0,
            conaffinity=0,
        )
    elif object_type == "laptop":
        body.add_geom(
            name=f"{object_id}_base",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[dimension / 2 for dimension in size],
            material=MATERIAL["laptop"],
            contype=0,
            conaffinity=0,
        )
        body.add_geom(
            name=f"{object_id}_keyboard",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[size[0] * 0.34, size[1] * 0.27, 0.0006],
            pos=[0.0, -size[1] * 0.10, size[2] / 2 + 0.0007],
            material=MATERIAL["dark_metal"],
            contype=0,
            conaffinity=0,
        )
        hinge = body.add_body(
            name=f"{object_id}_lid",
            pos=[0.0, size[1] * 0.43, size[2] * 0.38],
            quat=[0.843, -0.537, 0.0, 0.0],
        )
        hinge.add_geom(
            name=f"{object_id}_lid_shell",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[size[0] * 0.47, size[1] * 0.36, 0.0014],
            pos=[0.0, -size[1] * 0.36, 0.0],
            material=MATERIAL["laptop"],
            contype=0,
            conaffinity=0,
        )
        hinge.add_geom(
            name=f"{object_id}_screen",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[size[0] * 0.41, size[1] * 0.30, 0.00065],
            pos=[0.0, -size[1] * 0.36, -0.0016],
            material=MATERIAL["screen"],
            contype=0,
            conaffinity=0,
        )
        body.add_geom(
            name=f"{object_id}_hinge",
            type=mujoco.mjtGeom.mjGEOM_CAPSULE,
            fromto=[
                -size[0] * 0.42,
                size[1] * 0.43,
                size[2] * 0.42,
                size[0] * 0.42,
                size[1] * 0.43,
                size[2] * 0.42,
            ],
            size=[max(0.001, size[2] * 0.12), 0.0, 0.0],
            material=MATERIAL["dark_metal"],
            contype=0,
            conaffinity=0,
        )
    elif object_type in {"phone", "tablet", "electronic"}:
        shell_material = (
            MATERIAL["tablet"]
            if object_type in {"tablet", "electronic"}
            else MATERIAL["dark_metal"]
        )
        body.add_geom(
            name=f"{object_id}_bezel",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[dimension / 2 for dimension in size],
            material=shell_material,
            contype=0,
            conaffinity=0,
        )
        body.add_geom(
            name=f"{object_id}_screen",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[
                max(0.002, size[0] * 0.40),
                max(0.002, size[1] * 0.40),
                0.00055,
            ],
            pos=[0.0, 0.0, size[2] / 2 + 0.00065],
            material=MATERIAL["screen"],
            contype=0,
            conaffinity=0,
        )
    elif object_type in {"book", "notebook", "magazine"}:
        cover_material = (
            MATERIAL["magazine"]
            if object_type == "magazine"
            else MATERIAL["book"]
        )
        body.add_geom(
            name=f"{object_id}_cover",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[dimension / 2 for dimension in size],
            material=cover_material,
            contype=0,
            conaffinity=0,
        )
        body.add_geom(
            name=f"{object_id}_pages",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[
                max(0.002, size[0] * 0.46),
                max(0.002, size[1] * 0.43),
                max(0.0007, size[2] * 0.27),
            ],
            pos=[0.0, 0.0, size[2] * 0.05],
            material=MATERIAL["paper"],
            contype=0,
            conaffinity=0,
        )
        body.add_geom(
            name=f"{object_id}_spine",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[
                max(0.001, size[0] * 0.055),
                size[1] * 0.47,
                max(0.0008, size[2] * 0.34),
            ],
            pos=[-size[0] * 0.43, 0.0, size[2] * 0.16],
            material=cover_material,
            contype=0,
            conaffinity=0,
        )
        if object_type == "magazine":
            body.add_geom(
                name=f"{object_id}_cover_stripe",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[
                    max(0.002, size[0] * 0.31),
                    max(0.001, size[1] * 0.045),
                    0.0006,
                ],
                pos=[0.0, -size[1] * 0.25, size[2] / 2 + 0.0007],
                rgba=[0.92, 0.53, 0.08, 1.0],
                contype=0,
                conaffinity=0,
            )
    elif object_type == "knife":
        body.add_geom(
            name=f"{object_id}_blade",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[
                max(0.001, size[0] * 0.50),
                max(0.004, size[1] * 0.25),
                size[2] * 0.56,
            ],
            pos=[0.0, size[1] * 0.25, size[2] * 0.10],
            material=MATERIAL["metal"],
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
            material=MATERIAL["dark_metal"],
            contype=0,
            conaffinity=0,
        )
        body.add_geom(
            name=f"{object_id}_blade_tip",
            type=mujoco.mjtGeom.mjGEOM_ELLIPSOID,
            size=[
                max(0.001, size[0] * 0.48),
                max(0.002, size[1] * 0.07),
                size[2] * 0.52,
            ],
            pos=[0.0, size[1] * 0.50, size[2] * 0.10],
            material=MATERIAL["metal"],
            contype=0,
            conaffinity=0,
        )
    elif object_type in {"fork", "spoon", "utensil"}:
        body.add_geom(
            name=f"{object_id}_handle",
            type=mujoco.mjtGeom.mjGEOM_CAPSULE,
            fromto=[
                0.0,
                -size[1] * 0.48,
                0.0,
                0.0,
                size[1] * 0.18,
                0.0,
            ],
            size=[max(0.0012, size[0] * 0.20), 0.0, 0.0],
            material=MATERIAL["metal"],
            contype=0,
            conaffinity=0,
        )
        body.add_geom(
            name=f"{object_id}_head",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[size[0] * 0.46, size[1] * 0.10, size[2] * 0.34],
            pos=[0.0, size[1] * 0.26, 0.0],
            material=MATERIAL["metal"],
            contype=0,
            conaffinity=0,
        )
        if object_type == "fork":
            for tine, x_offset in enumerate((-0.36, -0.12, 0.12, 0.36), start=1):
                body.add_geom(
                    name=f"{object_id}_tine_{tine}",
                    type=mujoco.mjtGeom.mjGEOM_BOX,
                    size=[
                        max(0.00065, size[0] * 0.070),
                        size[1] * 0.15,
                        size[2] * 0.25,
                    ],
                    pos=[
                        size[0] * x_offset,
                        size[1] * 0.46,
                        0.0,
                    ],
                    material=MATERIAL["metal"],
                    contype=0,
                    conaffinity=0,
                )
    elif object_type == "napkin":
        body.add_geom(
            name=f"{object_id}_cloth",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[dimension / 2 for dimension in size],
            material=MATERIAL["napkin"],
            contype=0,
            conaffinity=0,
        )
        body.add_geom(
            name=f"{object_id}_fold",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[size[0] * 0.33, 0.00075, 0.0005],
            pos=[0.0, 0.0, size[2] / 2 + 0.00055],
            euler=[0.0, 0.0, 0.72],
            rgba=[0.17, 0.48, 0.48, 1.0],
            contype=0,
            conaffinity=0,
        )
    else:
        body.add_geom(
            name=f"{object_id}_geom",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[dimension / 2 for dimension in size],
            material=MATERIAL["generic"],
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
    _configure_visuals(model_spec)
    _add_benchmark_materials(model_spec)
    camera = next(
        camera for camera in model_spec.cameras if camera.name == "third_person"
    )
    # Mount the observation camera directly across the table from the robot.
    # Keeping it on y=0 produces a head-on view while the workspace-centred
    # target keeps both task regions and the robot in frame.
    camera.pos = list(THIRD_PERSON_CAMERA_POSITION)
    camera.fovy = THIRD_PERSON_CAMERA_FOVY
    camera.alt.xyaxes = _look_at_xyaxes(
        camera.pos,
        THIRD_PERSON_CAMERA_TARGET,
    )
    try:
        material = next(
            material for material in model_spec.materials if material.name == "black_mat"
        )
        material.rgba = [0.22, 0.25, 0.29, 1.0]
        material.roughness = 0.28
        material.specular = 0.62
        material.shininess = 0.66
        material.reflectance = 0.10
        material.metallic = 0.55
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
