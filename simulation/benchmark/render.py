"""Deterministic endpoint rendering for the PrefMem simulation benchmark.

The renderer deliberately has no knowledge of task targets or outcome labels.
An initial image is a function of the scene identity and ``initial_objects``;
a final image is a function of the same scene identity and ``final_objects``.
This keeps target/outcome metadata out of the pixels and makes counterfactual
episodes that share an initial scene byte-for-byte comparable.

The public functions accept either dataclass-like objects or dictionaries.
Only Pillow is required.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import tempfile
from collections.abc import Mapping, Sequence
from typing import Any

from PIL import Image, ImageDraw, ImageFilter


IMAGE_SIZE = (640, 480)

_MISSING = object()

_NAMED_COLOURS: dict[str, tuple[int, int, int]] = {
    "red": (210, 54, 56),
    "green": (54, 158, 91),
    "blue": (51, 105, 195),
    "yellow": (232, 184, 55),
    "orange": (224, 126, 49),
    "purple": (132, 83, 176),
    "pink": (222, 117, 158),
    "brown": (135, 87, 58),
    "black": (38, 42, 48),
    "white": (236, 238, 236),
    "grey": (122, 129, 136),
    "gray": (122, 129, 136),
    "silver": (166, 172, 178),
    "beige": (207, 186, 145),
}


@dataclass(frozen=True)
class _VisualObject:
    object_id: str
    kind: str
    label: str
    colour: tuple[int, int, int]
    position: tuple[float, float, float] | None
    region: str
    orientation: float
    attrs: Mapping[str, Any]


def render_initial(spec: Any, path: str | os.PathLike[str]) -> Path:
    """Render the initial endpoint to a deterministic 640x480 RGB PNG."""

    return render_observation(spec, path, view="initial")


def render_final(spec: Any, path: str | os.PathLike[str]) -> Path:
    """Render the final endpoint to a deterministic 640x480 RGB PNG."""

    return render_observation(spec, path, view="final")


def render_pair(
    spec: Any,
    initial_path: str | os.PathLike[str],
    final_path: str | os.PathLike[str],
) -> tuple[Path, Path]:
    """Render both endpoint observations and return their paths."""

    initial = render_initial(spec, initial_path)
    final = render_final(spec, final_path)
    return initial, final


def render_observation(
    spec: Any,
    path: str | os.PathLike[str],
    *,
    view: str = "initial",
) -> Path:
    """Render one endpoint.

    ``view`` must be ``"initial"`` or ``"final"``.  The initial random
    generator is seeded only with ``family``, ``scene_variant`` and ``seed``.
    No target or outcome field is ever read.
    """

    normalised_view = str(view).strip().lower()
    if normalised_view not in {"initial", "final"}:
        raise ValueError("view must be 'initial' or 'final'")

    family = _normalise_family(
        _first_value(spec, "family", "task_family", "scene.family", default="block_stack")
    )
    objects, state_was_supplied = _extract_objects(spec, normalised_view)
    if normalised_view == "final" and _is_observation_occluded(spec):
        objects.append(_observation_occluder())
    rng = random.Random(_scene_seed(spec, family))

    # A missing initial state is rendered as a stable illustrative scene.  A
    # missing/empty final state remains empty: final pixels must not be inferred
    # from target or outcome metadata.
    if normalised_view == "initial" and not state_was_supplied:
        objects = _default_initial_objects(family)

    image = _draw_background(family)
    if family == "block_stack":
        _render_blocks(image, objects, normalised_view, rng)
    elif family == "category_sort":
        _render_category_sort(image, objects, normalised_view, rng)
    elif family == "place_setting":
        _render_place_setting(image, objects, normalised_view, rng)
    else:
        _render_generic(image, objects, normalised_view, rng)

    output = Path(path)
    _atomic_save_png(image.convert("RGB"), output)
    return output


# ---------------------------------------------------------------------------
# Input adaptation and determinism


def _lookup(source: Any, name: str, default: Any = _MISSING) -> Any:
    if source is None:
        return default
    if isinstance(source, Mapping):
        if name in source:
            return source[name]
        folded = name.casefold()
        for key, value in source.items():
            if str(key).casefold() == folded:
                return value
        return default
    try:
        return getattr(source, name)
    except (AttributeError, TypeError):
        return default


def _path_value(source: Any, path: str, default: Any = _MISSING) -> Any:
    current = source
    for part in path.split("."):
        current = _lookup(current, part, _MISSING)
        if current is _MISSING:
            return default
    return current


def _first_value(source: Any, *paths: str, default: Any = None) -> Any:
    for path in paths:
        value = _path_value(source, path, _MISSING)
        if value is not _MISSING and value is not None:
            return value
    return default


def _normalise_family(value: Any) -> str:
    token = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "blocks": "block_stack",
        "block": "block_stack",
        "stack": "block_stack",
        "stacking": "block_stack",
        "ordered_stack": "block_stack",
        "semantic_sort": "category_sort",
        "sorting": "category_sort",
        "category_assignment": "category_sort",
        "tidy_table": "category_sort",
        "table_setting": "place_setting",
        "setting": "place_setting",
        "place": "place_setting",
    }
    return aliases.get(token, token or "block_stack")


def _scene_seed(spec: Any, family: str) -> int:
    """Create a stable seed without touching target or outcome fields."""

    variant = _first_value(
        spec,
        "scene_variant",
        "scene.variant",
        "scene_variant_id",
        default="default",
    )
    seed = _first_value(spec, "seed", "scene.seed", "scene_seed", default=0)
    canonical = json.dumps(
        {
            "family": family,
            "scene_variant": _json_scalar(variant),
            "seed": _json_scalar(seed),
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def _json_scalar(value: Any) -> str | int | float | bool | None:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _extract_objects(spec: Any, view: str) -> tuple[list[_VisualObject], bool]:
    paths = (
        (
            "initial_objects",
            "scene.initial_objects",
            "ground_truth.initial_objects",
            "initial_state.objects",
            "states.initial.objects",
        )
        if view == "initial"
        else (
            "final_objects",
            "ground_truth.final_objects",
            "final_state.objects",
            "states.final.objects",
            "result.final_objects",
        )
    )
    raw: Any = _MISSING
    for path in paths:
        raw = _path_value(spec, path, _MISSING)
        if raw is not _MISSING:
            break
    if raw is _MISSING:
        return [], False

    if raw is None:
        return [], True
    if isinstance(raw, Mapping):
        nested = _first_value(raw, "objects", "items", default=_MISSING)
        if nested is not _MISSING:
            raw = nested

    entries: list[tuple[str | None, Any]]
    if isinstance(raw, Mapping):
        entries = [(str(key), value) for key, value in raw.items()]
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
        entries = [(None, value) for value in raw]
    else:
        entries = [(None, raw)]

    normalised = [
        _normalise_object(item, fallback_id=fallback_id or f"object_{index:02d}")
        for index, (fallback_id, item) in enumerate(entries)
    ]
    normalised.sort(key=lambda item: item.object_id.casefold())
    return normalised, True


def _pairs_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return {str(key): item for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        result: dict[str, Any] = {}
        for entry in value:
            if (
                isinstance(entry, Sequence)
                and not isinstance(entry, (str, bytes, bytearray))
                and len(entry) == 2
            ):
                result[str(entry[0])] = entry[1]
        return result
    return {}


def _normalise_object(item: Any, fallback_id: str) -> _VisualObject:
    object_id = str(
        _first_value(item, "object_id", "id", "name", default=fallback_id)
    )

    attrs: dict[str, Any] = {}
    for attrs_name in ("metadata", "attributes", "attrs"):
        attrs_value = _path_value(item, attrs_name, _MISSING)
        attrs.update(_pairs_mapping(attrs_value))
    # Dynamic state contains placement facts such as zone and stack_level.
    attrs.update(_pairs_mapping(_path_value(item, "state", _MISSING)))

    raw_kind = str(
        _first_value(
            item,
            "kind",
            "object_type",
            "type",
            "category",
            "class_name",
            default="object",
        )
    ).strip().lower().replace("-", "_").replace(" ", "_")
    role = str(attrs.get("role", "")).strip().lower().replace("-", "_").replace(" ", "_")
    if raw_kind in {"tabletop_item", "place_setting_item", "item", "object"} and role:
        kind = role
    elif raw_kind in {"cube", "wooden_cube", "coloured_cube", "colored_cube"}:
        kind = "block"
    else:
        kind = raw_kind
    label = str(
        _first_value(
            item,
            "label",
            "display_name",
            "name",
            default=role or object_id,
        )
    )

    position_m = _first_value(item, "position_m", "pose.position_m", default=_MISSING)
    if position_m is not _MISSING:
        position_source = position_m
        attrs.setdefault("coordinate_space", "metres")
    else:
        position_source = _first_value(
            item, "position", "pos", "xyz", "pose.position", default=None
        )
    size_m = _first_value(item, "size_m", "dimensions_m", default=_MISSING)
    if size_m is not _MISSING:
        attrs.setdefault("size_m", size_m)
    position = _parse_position(position_source, item)
    region = str(
        _first_value(
            item,
            "region",
            "location",
            "slot",
            default=attrs.get("region", attrs.get("zone", "")),
        )
    ).strip().lower().replace("-", "_").replace(" ", "_")
    orientation_xyzw = _first_value(
        item,
        "orientation_xyzw",
        "pose.orientation_xyzw",
        default=_MISSING,
    )
    if orientation_xyzw is not _MISSING:
        orientation = _parse_orientation(orientation_xyzw, quaternion_order="xyzw")
    else:
        orientation = _parse_orientation(
            _first_value(
                item,
                "orientation",
                "yaw",
                "angle",
                "pose.orientation",
                "attrs.orientation",
                default=0.0,
            )
        )

    explicit_colour = _first_value(
        item,
        "color",
        "colour",
        default=attrs.get("color", attrs.get("colour")),
    )
    colour = _parse_colour(explicit_colour, object_id, label, kind)
    return _VisualObject(
        object_id=object_id,
        kind=kind,
        label=label,
        colour=colour,
        position=position,
        region=region,
        orientation=orientation,
        attrs=attrs,
    )


def _parse_position(value: Any, item: Any) -> tuple[float, float, float] | None:
    if isinstance(value, Mapping) or (
        value is not None and not isinstance(value, (str, bytes, Sequence))
    ):
        x = _first_value(value, "x", "0", default=_MISSING)
        y = _first_value(value, "y", "1", default=_MISSING)
        z = _first_value(value, "z", "2", default=0.0)
        if x is not _MISSING and y is not _MISSING:
            try:
                return float(x), float(y), float(z)
            except (TypeError, ValueError):
                pass
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if len(value) >= 2:
            try:
                return (
                    float(value[0]),
                    float(value[1]),
                    float(value[2]) if len(value) >= 3 else 0.0,
                )
            except (TypeError, ValueError):
                pass

    x = _first_value(item, "x", "pose.x", default=_MISSING)
    y = _first_value(item, "y", "pose.y", default=_MISSING)
    z = _first_value(item, "z", "pose.z", default=0.0)
    if x is not _MISSING and y is not _MISSING:
        try:
            return float(x), float(y), float(z)
        except (TypeError, ValueError):
            return None
    return None


def _parse_orientation(value: Any, *, quaternion_order: str = "wxyz") -> float:
    if isinstance(value, Mapping):
        value = _first_value(value, "yaw", "z", "angle", default=0.0)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        # A quaternion is accepted, but only its planar yaw is relevant here.
        if len(value) == 4:
            try:
                components = tuple(float(component) for component in value)
                if quaternion_order == "xyzw":
                    x, y, z, w = components
                else:
                    w, x, y, z = components
                radians = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
                return math.degrees(radians)
            except (TypeError, ValueError):
                return 0.0
        value = value[-1] if value else 0.0
    try:
        angle = float(value)
    except (TypeError, ValueError):
        return 0.0
    if 0.0 < abs(angle) <= 2.0 * math.pi:
        return math.degrees(angle)
    return angle


def _parse_colour(
    value: Any,
    object_id: str,
    label: str,
    kind: str,
) -> tuple[int, int, int]:
    if isinstance(value, str):
        token = value.strip().lower()
        if token in _NAMED_COLOURS:
            return _NAMED_COLOURS[token]
        if token.startswith("#") and len(token) in {4, 7}:
            try:
                if len(token) == 4:
                    return tuple(int(char * 2, 16) for char in token[1:])  # type: ignore[return-value]
                return tuple(int(token[index : index + 2], 16) for index in (1, 3, 5))  # type: ignore[return-value]
            except ValueError:
                pass
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if len(value) >= 3:
            try:
                channels = tuple(float(channel) for channel in value[:3])
                if all(0.0 <= channel <= 1.0 for channel in channels):
                    channels = tuple(channel * 255.0 for channel in channels)
                return tuple(_clamp_channel(channel) for channel in channels)  # type: ignore[return-value]
            except (TypeError, ValueError):
                pass

    searchable = f"{object_id} {label} {kind}".lower()
    for name, colour in _NAMED_COLOURS.items():
        if name in searchable:
            return colour
    kind_defaults = {
        "book": (178, 77, 62),
        "phone": (53, 59, 66),
        "smartphone": (53, 59, 66),
        "tablet": (67, 74, 84),
        "laptop": (113, 123, 132),
        "plate": (224, 228, 225),
        "bowl": (222, 226, 223),
        "cup": (218, 222, 220),
        "mug": (218, 222, 220),
        "fork": (177, 182, 184),
        "knife": (177, 182, 184),
        "spoon": (177, 182, 184),
        "napkin": (181, 72, 75),
        "banana": (229, 194, 61),
        "sweet_potato": (155, 82, 56),
        "occluder": (103, 111, 118),
    }
    if kind in kind_defaults:
        return kind_defaults[kind]
    digest = hashlib.sha256(f"{object_id}|{kind}".encode("utf-8")).digest()
    return 70 + digest[0] % 130, 70 + digest[1] % 130, 70 + digest[2] % 130


def _clamp_channel(value: float) -> int:
    return max(0, min(255, int(round(value))))


def _default_initial_objects(family: str) -> list[_VisualObject]:
    if family == "block_stack":
        raw = [
            {"object_id": "red_block", "kind": "block", "color": "red"},
            {"object_id": "green_block", "kind": "block", "color": "green"},
            {"object_id": "blue_block", "kind": "block", "color": "blue"},
        ]
    elif family == "category_sort":
        raw = [
            {"object_id": "red_book", "kind": "book", "color": "red"},
            {"object_id": "blue_book", "kind": "book", "color": "blue"},
            {"object_id": "phone", "kind": "phone"},
            {"object_id": "laptop", "kind": "laptop"},
        ]
    elif family == "place_setting":
        raw = [
            {"object_id": "plate", "kind": "plate"},
            {"object_id": "fork", "kind": "fork"},
            {"object_id": "knife", "kind": "knife"},
            {"object_id": "cup", "kind": "cup"},
            {"object_id": "napkin", "kind": "napkin", "color": "red"},
        ]
    else:
        raw = [
            {"object_id": "object_a", "kind": "object", "color": "blue"},
            {"object_id": "object_b", "kind": "object", "color": "orange"},
            {"object_id": "object_c", "kind": "object", "color": "green"},
        ]
    return [
        _normalise_object(item, fallback_id=f"object_{index:02d}")
        for index, item in enumerate(raw)
    ]


def _is_observation_occluded(spec: Any) -> bool:
    status = str(
        _first_value(
            spec,
            "observation_status",
            "observability.status",
            "observation.status",
            "ground_truth.observability.status",
            default="",
        )
    ).strip().lower().replace("-", "_").replace(" ", "_")
    occluded_ids = _first_value(
        spec,
        "occluded_object_ids",
        "observability.occluded_object_ids",
        "ground_truth.observability.occluded_object_ids",
        default=(),
    )
    return status in {
        "occluded",
        "partially_occluded",
        "unobservable",
        "not_observable",
    } or bool(occluded_ids)


def _observation_occluder() -> _VisualObject:
    return _VisualObject(
        object_id="__physical_foreground_panel__",
        kind="foreground_panel",
        label="foreground panel",
        colour=(111, 119, 126),
        position=None,
        region="foreground",
        orientation=0.0,
        attrs={
            "occluder": True,
            "alpha": 255,
            "pixel_width": 540,
            "pixel_height": 315,
        },
    )


# ---------------------------------------------------------------------------
# Background and coordinate mapping


def _draw_background(family: str) -> Image.Image:
    image = Image.new("RGB", IMAGE_SIZE, (94, 102, 112))
    draw = ImageDraw.Draw(image)

    # Cool studio wall and a warm oak work surface.  The grain is generated
    # procedurally so the fallback renderer remains asset-free and deterministic.
    for y in range(0, 145):
        shade = 112 - (y * 22 // 145)
        draw.line((0, y, 639, y), fill=(shade - 8, shade - 2, shade + 8))
    for y in range(145, 480):
        depth = y - 145
        step = depth * 30 // 335
        draw.line(
            (0, y, 639, y),
            fill=(145 + step, 85 + step * 3 // 4, 42 + step // 2),
        )
    draw.line((0, 145, 639, 145), fill=(61, 48, 38), width=4)

    # Long, slightly irregular lines read as oak grain without adding any
    # semantic clue about the task or target.
    for row, grain_y in enumerate(range(157, 480, 13)):
        points: list[tuple[int, int]] = []
        for x in range(-10, 651, 8):
            wave = 2.4 * math.sin(x * 0.035 + row * 0.71)
            wave += 1.2 * math.sin(x * 0.093 - row * 0.27)
            points.append((x, int(round(grain_y + wave))))
        grain_colour = (104 + row % 9, 57 + row % 7, 28 + row % 4)
        draw.line(points, fill=grain_colour, width=1)
        if row % 4 == 1:
            draw.line(
                [(x, y + 2) for x, y in points],
                fill=(183, 119, 61),
                width=1,
            )

    # Plank seams and a broad, soft key-light reflection add scale and depth.
    draw.line((0, 306, 639, 306), fill=(91, 49, 27), width=2)
    draw.line((0, 309, 639, 309), fill=(190, 125, 67), width=1)
    draw.line((112, 145, 112, 480), fill=(102, 56, 30), width=2)
    draw.line((525, 145, 525, 480), fill=(102, 56, 30), width=2)
    reflection = Image.new("RGBA", IMAGE_SIZE, (0, 0, 0, 0))
    reflection_draw = ImageDraw.Draw(reflection)
    reflection_draw.ellipse(
        (-210, 118, 430, 570),
        fill=(255, 214, 158, 29),
    )
    reflection = reflection.filter(ImageFilter.GaussianBlur(radius=34))
    image.paste(reflection, (0, 0), reflection)
    draw = ImageDraw.Draw(image)

    if family == "category_sort":
        _draw_sorting_mats(draw)
    elif family == "place_setting":
        draw.rounded_rectangle(
            (98, 174, 554, 446),
            radius=20,
            fill=(64, 40, 23),
        )
        draw.rounded_rectangle(
            (92, 166, 548, 438),
            radius=18,
            fill=(211, 160, 43),
            outline=(91, 58, 25),
            width=4,
        )
        for x in range(108, 535, 8):
            draw.line((x, 179, x, 425), fill=(224, 181, 67), width=1)
        for y in range(181, 426, 8):
            draw.line((106, y, 534, y), fill=(181, 127, 29), width=1)
        draw.arc(
            (105, 174, 535, 425),
            204,
            320,
            fill=(225, 181, 69),
            width=1,
        )
    elif family == "block_stack":
        draw.ellipse((170, 402, 470, 427), fill=(91, 51, 28))
    return image


def _draw_sorting_mats(draw: ImageDraw.ImageDraw) -> None:
    draw.rounded_rectangle(
        (58, 194, 302, 434),
        radius=18,
        fill=(62, 39, 23),
    )
    draw.rounded_rectangle(
        (52, 186, 296, 426),
        radius=18,
        fill=(214, 162, 43),
        outline=(88, 56, 26),
        width=4,
    )
    draw.rounded_rectangle(
        (350, 194, 594, 434),
        radius=18,
        fill=(62, 39, 23),
    )
    draw.rounded_rectangle(
        (344, 186, 588, 426),
        radius=18,
        fill=(35, 100, 105),
        outline=(17, 51, 55),
        width=4,
    )
    # Woven lines make the two physical sorting areas visibly distinct from
    # the glossy table while remaining semantically unlabelled.
    for x in range(68, 284, 8):
        draw.line((x, 202, x, 410), fill=(226, 182, 69), width=1)
    for y in range(204, 411, 8):
        draw.line((67, y, 281, y), fill=(182, 126, 27), width=1)
    for x in range(360, 576, 8):
        draw.line((x, 202, x, 410), fill=(47, 119, 123), width=1)
    for y in range(204, 411, 8):
        draw.line((359, y, 573, y), fill=(24, 78, 83), width=1)
    draw.arc((61, 197, 287, 416), 205, 315, fill=(226, 183, 72), width=1)
    draw.arc((353, 197, 579, 416), 205, 315, fill=(55, 123, 127), width=1)


def _position_space(obj: _VisualObject) -> str:
    value = str(
        _first_value(
            obj.attrs,
            "coordinate_space",
            "position_space",
            "frame",
            default="",
        )
    ).strip().lower()
    return value


def _map_xy(
    obj: _VisualObject,
    *,
    normalised_bounds: tuple[float, float, float, float],
    physical_scale: tuple[float, float],
    physical_origin: tuple[float, float],
) -> tuple[float, float] | None:
    if obj.position is None:
        screen_position = _first_value(
            obj.attrs, "screen_position", "pixel_position", default=None
        )
        parsed = _parse_position(screen_position, {})
        if parsed is None:
            return None
        return parsed[0], parsed[1]

    x, y, _ = obj.position
    space = _position_space(obj)
    if space in {"pixel", "pixels", "screen", "image"} or max(abs(x), abs(y)) > 2.0:
        return x, y
    if space in {"m", "metre", "metres", "meter", "meters", "world_m"}:
        # Archived ARX workspaces use forward x≈0.18..0.34 m and lateral
        # y≈-0.16..0.16 m.  The camera sees lateral y horizontally and forward
        # x as tabletop depth.
        screen_x = 320.0 + y * 1_250.0
        screen_y = 205.0 + (x - 0.16) * 900.0
        return (
            max(72.0, min(568.0, screen_x)),
            max(185.0, min(400.0, screen_y)),
        )

    left, top, right, bottom = normalised_bounds
    if space in {"normalised", "normalized", "unit", "relative"} or (
        0.0 <= x <= 1.0 and 0.0 <= y <= 1.0
    ):
        return left + x * (right - left), top + y * (bottom - top)

    return (
        physical_origin[0] + x * physical_scale[0],
        physical_origin[1] + y * physical_scale[1],
    )


def _region_token(obj: _VisualObject) -> str:
    if obj.region:
        return obj.region
    return str(_first_value(obj.attrs, "region", "location", "slot", default="")).lower()


def _stable_fraction(text: str, salt: str = "") -> float:
    digest = hashlib.sha256(f"{salt}|{text}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") / 0xFFFFFFFF


def _shade(colour: tuple[int, int, int], factor: float) -> tuple[int, int, int]:
    return tuple(_clamp_channel(channel * factor) for channel in colour)  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Block stacking


def _render_blocks(
    image: Image.Image,
    objects: list[_VisualObject],
    view: str,
    rng: random.Random,
) -> None:
    draw = ImageDraw.Draw(image)
    visible = [obj for obj in objects if not _is_hidden(obj)]
    occluders = [obj for obj in visible if _is_occluder(obj)]
    blocks = [obj for obj in visible if not _is_occluder(obj)]

    fallback_positions = _scattered_positions(len(blocks), rng, (165, 300, 475, 360))
    z_values = sorted(
        {
            round(obj.position[2], 6)
            for obj in blocks
            if obj.position is not None
        }
    )
    placements: list[tuple[float, float, int, _VisualObject]] = []
    for index, obj in enumerate(blocks):
        mapped = _map_xy(
            obj,
            normalised_bounds=(120, 285, 520, 365),
            physical_scale=(620, 270),
            physical_origin=(320, 343),
        )
        if mapped is None:
            mapped = fallback_positions[index]
        x, y = mapped
        level_value = _first_value(
            obj.attrs, "stack_level", "level", "height_index", default=_MISSING
        )
        if level_value is not _MISSING:
            try:
                level = max(0, int(level_value))
            except (TypeError, ValueError):
                level = 0
        elif obj.position is not None and len(z_values) > 1:
            level = z_values.index(round(obj.position[2], 6))
        else:
            level = 0
        y -= level * 49
        placements.append((x, y, level, obj))

    # Lower blocks and physically farther objects are drawn first.
    placements.sort(key=lambda entry: (entry[2], entry[1], entry[3].object_id.casefold()))
    for x, y, _, obj in placements:
        draw.ellipse((x - 33, y + 21, x + 35, y + 34), fill=(135, 117, 95))
        _draw_block(draw, x, y, obj.colour)

    _draw_occluders(image, occluders)


def _scattered_positions(
    count: int,
    rng: random.Random,
    bounds: tuple[int, int, int, int],
) -> list[tuple[float, float]]:
    left, top, right, bottom = bounds
    canonical = [
        (left, bottom - 8),
        ((left + right) / 2, top + 5),
        (right, bottom - 15),
        (left + 75, top + 18),
        (right - 78, top + 25),
        ((left + right) / 2, bottom - 3),
    ]
    rng.shuffle(canonical)
    output: list[tuple[float, float]] = []
    for index in range(count):
        if index < len(canonical):
            x, y = canonical[index]
        else:
            columns = max(1, int(math.ceil(math.sqrt(count))))
            x = left + (index % columns) * (right - left) / max(1, columns - 1)
            y = top + (index // columns) * 62
        output.append((x + rng.randint(-9, 9), y + rng.randint(-5, 5)))
    return output


def _draw_block(
    draw: ImageDraw.ImageDraw,
    x: float,
    y: float,
    colour: tuple[int, int, int],
) -> None:
    x = int(round(x))
    y = int(round(y))
    half_width = 34
    height = 43
    depth = 10
    draw.rectangle(
        (x - half_width, y - height // 2, x + half_width, y + height // 2),
        fill=colour,
        outline=_shade(colour, 0.55),
        width=3,
    )
    draw.polygon(
        [
            (x - half_width, y - height // 2),
            (x - half_width + depth, y - height // 2 - depth),
            (x + half_width + depth, y - height // 2 - depth),
            (x + half_width, y - height // 2),
        ],
        fill=_shade(colour, 1.18),
        outline=_shade(colour, 0.60),
    )
    draw.polygon(
        [
            (x + half_width, y - height // 2),
            (x + half_width + depth, y - height // 2 - depth),
            (x + half_width + depth, y + height // 2 - depth),
            (x + half_width, y + height // 2),
        ],
        fill=_shade(colour, 0.76),
        outline=_shade(colour, 0.55),
    )
    draw.line(
        (x - half_width + 8, y + height // 2 - 7, x + half_width - 8, y + height // 2 - 7),
        fill=_shade(colour, 0.88),
        width=2,
    )


# ---------------------------------------------------------------------------
# Semantic category sorting


def _render_category_sort(
    image: Image.Image,
    objects: list[_VisualObject],
    view: str,
    rng: random.Random,
) -> None:
    visible = [obj for obj in objects if not _is_hidden(obj)]
    occluders = [obj for obj in visible if _is_occluder(obj)]
    items = [obj for obj in visible if not _is_occluder(obj)]

    fallback = _scattered_positions(len(items), rng, (155, 245, 485, 370))
    by_region: dict[str, list[_VisualObject]] = {}
    for obj in items:
        region = _canonical_side(_region_token(obj))
        if region:
            by_region.setdefault(region, []).append(obj)
    for region_items in by_region.values():
        region_items.sort(key=lambda obj: obj.object_id.casefold())

    placements: list[tuple[float, float, _VisualObject]] = []
    for index, obj in enumerate(items):
        mapped = _map_xy(
            obj,
            normalised_bounds=(70, 205, 570, 410),
            physical_scale=(700, 390),
            physical_origin=(320, 300),
        )
        region = _canonical_side(_region_token(obj))
        if mapped is None and region:
            slot_index = by_region[region].index(obj)
            mapped = _category_region_position(region, slot_index)
        if mapped is None:
            mapped = fallback[index]
        placements.append((mapped[0], mapped[1], obj))

    placements.sort(key=lambda entry: (entry[1], entry[2].object_id.casefold()))
    for x, y, obj in placements:
        _draw_category_object(image, x, y, obj)

    _draw_occluders(image, occluders)


def _canonical_side(region: str) -> str:
    token = region.lower()
    if token in {
        "left",
        "left_bin",
        "left_mat",
        "bin_a",
        "tray_a",
        "region_a",
        "a",
    }:
        return "left"
    if token in {
        "right",
        "right_bin",
        "right_mat",
        "bin_b",
        "tray_b",
        "region_b",
        "b",
    }:
        return "right"
    if token in {"centre", "center", "middle", "unassigned", "workspace"}:
        return "centre"
    return ""


def _category_region_position(region: str, index: int) -> tuple[float, float]:
    slots = {
        "left": [(120, 265), (225, 265), (120, 360), (225, 360)],
        "right": [(412, 265), (517, 265), (412, 360), (517, 360)],
        "centre": [(320, 275), (320, 355), (285, 315), (355, 315)],
    }
    values = slots.get(region, slots["centre"])
    return values[index % len(values)]


def _draw_category_object(
    image: Image.Image,
    x: float,
    y: float,
    obj: _VisualObject,
) -> None:
    kind = obj.kind
    searchable = f"{obj.object_id} {obj.label} {kind}".lower()
    if any(token in searchable for token in ("book", "magazine", "journal")):
        sprite = _book_sprite(obj.colour)
    elif "laptop" in searchable or "computer" in searchable or kind in {"pc", "notebook"}:
        sprite = _laptop_sprite(obj.colour)
    elif "phone" in searchable or "tablet" in searchable:
        sprite = _phone_sprite(obj.colour, tablet=("tablet" in searchable))
    elif "mouse" in searchable:
        sprite = _mouse_sprite(obj.colour)
    elif "camera" in searchable:
        sprite = _camera_sprite(obj.colour)
    else:
        sprite = _generic_sprite(obj.colour)
    _paste_rotated(image, sprite, x, y, obj.orientation)


def _book_sprite(colour: tuple[int, int, int]) -> Image.Image:
    sprite = Image.new("RGBA", (96, 76), (0, 0, 0, 0))
    draw = ImageDraw.Draw(sprite)
    draw.rounded_rectangle((13, 12, 82, 65), radius=5, fill=(96, 80, 65, 75))
    draw.rounded_rectangle(
        (10, 8, 78, 60),
        radius=4,
        fill=colour + (255,),
        outline=_shade(colour, 0.55) + (255,),
        width=3,
    )
    draw.rectangle((17, 10, 23, 58), fill=_shade(colour, 0.70) + (255,))
    draw.line((31, 18, 65, 18), fill=_shade(colour, 1.25) + (255,), width=3)
    draw.line((31, 25, 58, 25), fill=_shade(colour, 1.16) + (255,), width=2)
    draw.line((31, 48, 68, 48), fill=_shade(colour, 0.78) + (255,), width=2)
    return sprite


def _phone_sprite(colour: tuple[int, int, int], *, tablet: bool) -> Image.Image:
    width, height = ((84, 72) if tablet else (66, 82))
    sprite = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(sprite)
    box = (12, 7, width - 12, height - 7)
    draw.rounded_rectangle(box, radius=7, fill=colour + (255,), outline=(25, 29, 33, 255), width=3)
    draw.rounded_rectangle(
        (box[0] + 5, box[1] + 7, box[2] - 5, box[3] - 8),
        radius=3,
        fill=(70, 112, 132, 255),
    )
    draw.line(
        (box[0] + 9, box[1] + 12, box[2] - 8, box[1] + 12),
        fill=(137, 183, 196, 255),
        width=2,
    )
    draw.ellipse(
        ((width // 2) - 2, box[3] - 5, (width // 2) + 2, box[3] - 1),
        fill=(177, 180, 181, 255),
    )
    return sprite


def _laptop_sprite(colour: tuple[int, int, int]) -> Image.Image:
    sprite = Image.new("RGBA", (112, 86), (0, 0, 0, 0))
    draw = ImageDraw.Draw(sprite)
    draw.rounded_rectangle(
        (20, 7, 91, 57),
        radius=4,
        fill=_shade(colour, 0.70) + (255,),
        outline=(48, 53, 58, 255),
        width=3,
    )
    draw.rectangle((26, 13, 85, 51), fill=(74, 121, 139, 255))
    draw.polygon(
        [(15, 58), (96, 58), (107, 76), (6, 76)],
        fill=_shade(colour, 1.12) + (255,),
        outline=(66, 69, 71, 255),
    )
    draw.rectangle((43, 62, 69, 69), fill=_shade(colour, 0.87) + (255,))
    return sprite


def _mouse_sprite(colour: tuple[int, int, int]) -> Image.Image:
    sprite = Image.new("RGBA", (70, 62), (0, 0, 0, 0))
    draw = ImageDraw.Draw(sprite)
    draw.ellipse((14, 7, 56, 56), fill=colour + (255,), outline=_shade(colour, 0.55) + (255,), width=3)
    draw.line((35, 10, 35, 31), fill=_shade(colour, 0.62) + (255,), width=2)
    draw.rounded_rectangle((32, 14, 38, 25), radius=2, fill=(190, 192, 190, 255))
    return sprite


def _camera_sprite(colour: tuple[int, int, int]) -> Image.Image:
    sprite = Image.new("RGBA", (84, 66), (0, 0, 0, 0))
    draw = ImageDraw.Draw(sprite)
    draw.rounded_rectangle((8, 17, 76, 57), radius=6, fill=colour + (255,), outline=(39, 42, 45, 255), width=3)
    draw.rectangle((20, 9, 42, 19), fill=_shade(colour, 0.85) + (255,))
    draw.ellipse((31, 22, 62, 53), fill=(31, 39, 47, 255), outline=(182, 187, 188, 255), width=3)
    draw.ellipse((39, 30, 54, 45), fill=(71, 119, 143, 255))
    return sprite


def _generic_sprite(colour: tuple[int, int, int]) -> Image.Image:
    sprite = Image.new("RGBA", (72, 64), (0, 0, 0, 0))
    draw = ImageDraw.Draw(sprite)
    draw.ellipse((8, 45, 64, 58), fill=(90, 75, 61, 80))
    draw.rounded_rectangle((10, 8, 62, 51), radius=8, fill=colour + (255,), outline=_shade(colour, 0.55) + (255,), width=3)
    draw.line((20, 18, 50, 18), fill=_shade(colour, 1.18) + (255,), width=3)
    return sprite


# ---------------------------------------------------------------------------
# Place setting


def _render_place_setting(
    image: Image.Image,
    objects: list[_VisualObject],
    view: str,
    rng: random.Random,
) -> None:
    visible = [obj for obj in objects if not _is_hidden(obj)]
    occluders = [obj for obj in visible if _is_occluder(obj)]
    items = [obj for obj in visible if not _is_occluder(obj)]
    fallback = _scattered_positions(len(items), rng, (155, 205, 485, 385))

    by_region: dict[str, list[_VisualObject]] = {}
    for obj in items:
        region = _canonical_place_region(_region_token(obj))
        if region:
            by_region.setdefault(region, []).append(obj)
    for region_items in by_region.values():
        region_items.sort(key=lambda obj: obj.object_id.casefold())

    placements: list[tuple[float, float, _VisualObject]] = []
    for index, obj in enumerate(items):
        mapped = _map_xy(
            obj,
            normalised_bounds=(115, 185, 525, 420),
            physical_scale=(660, 390),
            physical_origin=(320, 300),
        )
        region = _canonical_place_region(_region_token(obj))
        if mapped is None and region:
            mapped = _place_region_position(region, by_region[region].index(obj))
        if mapped is None:
            mapped = fallback[index]
        placements.append((mapped[0], mapped[1], obj))

    # Napkins/flatware first, raised vessels and food later.
    depth_order = {
        "napkin": 0,
        "fork": 1,
        "knife": 1,
        "spoon": 1,
        "plate": 2,
        "bowl": 3,
        "banana": 4,
        "sweet_potato": 4,
        "cup": 5,
        "mug": 5,
    }
    placements.sort(
        key=lambda entry: (
            depth_order.get(entry[2].kind, 2),
            entry[1],
            entry[2].object_id.casefold(),
        )
    )
    for x, y, obj in placements:
        _draw_place_object(image, x, y, obj)

    _draw_occluders(image, occluders)


def _canonical_place_region(region: str) -> str:
    aliases = {
        "center": "centre",
        "middle": "centre",
        "plate_center": "centre",
        "left": "left",
        "left_of_plate": "left",
        "west": "left",
        "right": "right",
        "right_of_plate": "right",
        "east": "right",
        "upper_left": "upper_left",
        "top_left": "upper_left",
        "north_west": "upper_left",
        "upper_right": "upper_right",
        "top_right": "upper_right",
        "north_east": "upper_right",
        "lower_left": "lower_left",
        "bottom_left": "lower_left",
        "lower_right": "lower_right",
        "bottom_right": "lower_right",
        "on_plate": "on_plate",
        "inside_plate": "on_plate",
        "in_plate": "on_plate",
        "on": "on_plate",
    }
    return aliases.get(region, region if region in aliases.values() else "")


def _place_region_position(region: str, index: int) -> tuple[float, float]:
    slots = {
        "centre": [(320, 300), (320, 300), (320, 315)],
        "on_plate": [(320, 295), (305, 300), (336, 300)],
        "left": [(218, 300), (185, 300), (235, 350)],
        "right": [(422, 300), (455, 300), (405, 350)],
        "upper_left": [(205, 205), (250, 215)],
        "upper_right": [(432, 207), (390, 215)],
        "lower_left": [(210, 385), (255, 390)],
        "lower_right": [(430, 385), (385, 390)],
    }
    values = slots.get(region, slots["centre"])
    return values[index % len(values)]


def _draw_place_object(
    image: Image.Image,
    x: float,
    y: float,
    obj: _VisualObject,
) -> None:
    kind = obj.kind
    searchable = f"{obj.object_id} {obj.label} {kind}".lower()
    if "plate" in searchable:
        sprite = _plate_sprite(obj.colour)
    elif "bowl" in searchable:
        sprite = _bowl_sprite(obj.colour)
    elif "fork" in searchable:
        sprite = _fork_sprite(obj.colour)
    elif "knife" in searchable:
        sprite = _knife_sprite(obj.colour)
    elif "spoon" in searchable:
        sprite = _spoon_sprite(obj.colour)
    elif "cup" in searchable or "mug" in searchable:
        sprite = _cup_sprite(obj.colour)
    elif "napkin" in searchable:
        sprite = _napkin_sprite(obj.colour)
    elif "banana" in searchable:
        sprite = _banana_sprite(obj.colour)
    elif "potato" in searchable:
        sprite = _potato_sprite(obj.colour)
    else:
        sprite = _generic_sprite(obj.colour)
    _paste_rotated(image, sprite, x, y, obj.orientation)


def _plate_sprite(colour: tuple[int, int, int]) -> Image.Image:
    sprite = Image.new("RGBA", (150, 126), (0, 0, 0, 0))
    draw = ImageDraw.Draw(sprite)
    draw.ellipse((10, 18, 140, 115), fill=(96, 81, 63, 50))
    draw.ellipse((8, 8, 138, 105), fill=colour + (255,), outline=(135, 143, 145, 255), width=3)
    draw.ellipse((27, 23, 119, 91), fill=_shade(colour, 0.97) + (255,), outline=(190, 196, 196, 255), width=3)
    draw.arc((17, 15, 129, 98), 205, 330, fill=(251, 251, 247, 255), width=3)
    return sprite


def _bowl_sprite(colour: tuple[int, int, int]) -> Image.Image:
    sprite = Image.new("RGBA", (120, 92), (0, 0, 0, 0))
    draw = ImageDraw.Draw(sprite)
    draw.ellipse((12, 12, 108, 67), fill=colour + (255,), outline=(128, 136, 139, 255), width=3)
    draw.ellipse((23, 19, 97, 54), fill=(176, 197, 198, 255), outline=(113, 124, 128, 255), width=2)
    draw.arc((22, 30, 98, 81), 0, 180, fill=_shade(colour, 0.75) + (255,), width=5)
    return sprite


def _fork_sprite(colour: tuple[int, int, int]) -> Image.Image:
    sprite = Image.new("RGBA", (54, 130), (0, 0, 0, 0))
    draw = ImageDraw.Draw(sprite)
    draw.rounded_rectangle((23, 35, 31, 121), radius=4, fill=colour + (255,), outline=(118, 123, 124, 255))
    draw.rectangle((13, 22, 41, 42), fill=colour + (255,))
    for x in (14, 22, 31, 39):
        draw.rounded_rectangle((x, 7, x + 4, 32), radius=2, fill=colour + (255,))
    draw.line((17, 40, 37, 40), fill=(123, 128, 128, 255), width=2)
    return sprite


def _knife_sprite(colour: tuple[int, int, int]) -> Image.Image:
    sprite = Image.new("RGBA", (56, 132), (0, 0, 0, 0))
    draw = ImageDraw.Draw(sprite)
    draw.rounded_rectangle((22, 72, 35, 124), radius=5, fill=(75, 78, 79, 255))
    draw.polygon(
        [(17, 9), (36, 9), (36, 79), (23, 79), (17, 61)],
        fill=colour + (255,),
        outline=(119, 125, 126, 255),
    )
    draw.line((20, 16, 20, 58), fill=(230, 233, 231, 255), width=2)
    return sprite


def _spoon_sprite(colour: tuple[int, int, int]) -> Image.Image:
    sprite = Image.new("RGBA", (58, 132), (0, 0, 0, 0))
    draw = ImageDraw.Draw(sprite)
    draw.rounded_rectangle((25, 51, 33, 124), radius=4, fill=colour + (255,))
    draw.ellipse((13, 7, 45, 61), fill=colour + (255,), outline=(117, 123, 124, 255), width=2)
    draw.arc((18, 13, 40, 51), 180, 355, fill=(230, 232, 230, 255), width=2)
    return sprite


def _cup_sprite(colour: tuple[int, int, int]) -> Image.Image:
    sprite = Image.new("RGBA", (104, 92), (0, 0, 0, 0))
    draw = ImageDraw.Draw(sprite)
    draw.ellipse((14, 10, 80, 72), fill=(105, 89, 69, 45))
    draw.ellipse((65, 26, 97, 61), outline=_shade(colour, 0.70) + (255,), width=8)
    draw.ellipse((10, 7, 78, 69), fill=colour + (255,), outline=(133, 141, 142, 255), width=3)
    draw.ellipse((18, 14, 70, 57), fill=(97, 64, 43, 255), outline=(168, 174, 173, 255), width=3)
    draw.arc((23, 18, 65, 51), 190, 325, fill=(154, 111, 76, 255), width=3)
    return sprite


def _napkin_sprite(colour: tuple[int, int, int]) -> Image.Image:
    sprite = Image.new("RGBA", (92, 88), (0, 0, 0, 0))
    draw = ImageDraw.Draw(sprite)
    draw.polygon(
        [(12, 12), (78, 9), (82, 74), (16, 79)],
        fill=colour + (255,),
        outline=_shade(colour, 0.63) + (255,),
    )
    draw.line((13, 13, 81, 74), fill=_shade(colour, 1.20) + (255,), width=3)
    draw.line((46, 11, 48, 76), fill=_shade(colour, 0.87) + (255,), width=2)
    return sprite


def _banana_sprite(colour: tuple[int, int, int]) -> Image.Image:
    sprite = Image.new("RGBA", (118, 86), (0, 0, 0, 0))
    draw = ImageDraw.Draw(sprite)
    draw.arc((8, -4, 109, 75), 12, 155, fill=_shade(colour, 0.65) + (255,), width=24)
    draw.arc((8, -7, 109, 72), 12, 155, fill=colour + (255,), width=17)
    draw.ellipse((96, 33, 105, 43), fill=(92, 74, 39, 255))
    return sprite


def _potato_sprite(colour: tuple[int, int, int]) -> Image.Image:
    sprite = Image.new("RGBA", (104, 80), (0, 0, 0, 0))
    draw = ImageDraw.Draw(sprite)
    draw.ellipse((10, 13, 94, 67), fill=colour + (255,), outline=_shade(colour, 0.60) + (255,), width=3)
    for x, y in ((29, 28), (48, 50), (67, 31), (78, 49)):
        draw.ellipse((x, y, x + 5, y + 4), fill=_shade(colour, 0.72) + (255,))
    return sprite


# ---------------------------------------------------------------------------
# Generic objects, visibility, and output


def _render_generic(
    image: Image.Image,
    objects: list[_VisualObject],
    view: str,
    rng: random.Random,
) -> None:
    visible = [obj for obj in objects if not _is_hidden(obj)]
    occluders = [obj for obj in visible if _is_occluder(obj)]
    items = [obj for obj in visible if not _is_occluder(obj)]
    fallback = _scattered_positions(len(items), rng, (145, 240, 495, 380))
    for index, obj in enumerate(items):
        mapped = _map_xy(
            obj,
            normalised_bounds=(90, 190, 550, 420),
            physical_scale=(650, 380),
            physical_origin=(320, 300),
        )
        x, y = mapped or fallback[index]
        _paste_rotated(image, _generic_sprite(obj.colour), x, y, obj.orientation)
    _draw_occluders(image, occluders)


def _is_hidden(obj: _VisualObject) -> bool:
    visible = _first_value(obj.attrs, "visible", "is_visible", default=True)
    if isinstance(visible, str):
        visible = visible.strip().lower() not in {"false", "no", "0", "hidden"}
    return not bool(visible) or obj.region in {"hidden", "removed", "out_of_view"}


def _is_occluder(obj: _VisualObject) -> bool:
    searchable = f"{obj.object_id} {obj.kind} {obj.label}".lower()
    return (
        obj.kind in {"occluder", "foreground_panel", "robot_arm", "barrier"}
        or "occluder" in searchable
        or bool(_first_value(obj.attrs, "occluder", "is_occluder", default=False))
    )


def _draw_occluders(image: Image.Image, objects: list[_VisualObject]) -> None:
    if not objects:
        return
    overlay = Image.new("RGBA", IMAGE_SIZE, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for index, obj in enumerate(objects):
        mapped = _map_xy(
            obj,
            normalised_bounds=(80, 170, 560, 430),
            physical_scale=(620, 360),
            physical_origin=(320, 300),
        )
        x, y = mapped or (320.0, 305.0)
        width = int(
            _first_value(obj.attrs, "width", "pixel_width", default=300 + 35 * index)
        )
        height = int(_first_value(obj.attrs, "height", "pixel_height", default=150))
        alpha = int(_first_value(obj.attrs, "alpha", "opacity", default=238))
        alpha = max(80, min(255, alpha))
        colour = obj.colour

        if obj.kind == "robot_arm" or "robot_arm" in obj.object_id.lower():
            # A physical foreground arm blocks the workspace while remaining
            # recognisable as an observation limitation, not an outcome label.
            joints = [(72, 421), (185, 337), (330, 294), (485, 242)]
            for start, end in zip(joints, joints[1:]):
                draw.line((*start, *end), fill=colour + (alpha,), width=42)
                draw.line((*start, *end), fill=_shade(colour, 1.17) + (alpha,), width=24)
            for joint_x, joint_y in joints:
                draw.ellipse(
                    (joint_x - 27, joint_y - 27, joint_x + 27, joint_y + 27),
                    fill=_shade(colour, 0.72) + (alpha,),
                    outline=(57, 61, 65, alpha),
                    width=4,
                )
            draw.rounded_rectangle(
                (470, 217, 535, 270),
                radius=9,
                fill=(52, 58, 62, alpha),
                outline=(31, 35, 38, alpha),
                width=3,
            )
        else:
            box = (
                int(x - width / 2),
                int(y - height / 2),
                int(x + width / 2),
                int(y + height / 2),
            )
            draw.rounded_rectangle(
                box,
                radius=16,
                fill=colour + (alpha,),
                outline=_shade(colour, 0.55) + (alpha,),
                width=5,
            )
            # Rivets make the neutral foreground shape look physical.
            for rivet_x, rivet_y in (
                (box[0] + 16, box[1] + 16),
                (box[2] - 16, box[1] + 16),
                (box[0] + 16, box[3] - 16),
                (box[2] - 16, box[3] - 16),
            ):
                draw.ellipse(
                    (rivet_x - 4, rivet_y - 4, rivet_x + 4, rivet_y + 4),
                    fill=_shade(colour, 1.28) + (alpha,),
                )
    image.paste(overlay, (0, 0), overlay)


def _paste_rotated(
    image: Image.Image,
    sprite: Image.Image,
    x: float,
    y: float,
    angle: float,
) -> None:
    if abs(angle) > 0.01:
        sprite = sprite.rotate(
            -angle,
            resample=Image.Resampling.BICUBIC,
            expand=True,
        )
    left = int(round(x - sprite.width / 2))
    top = int(round(y - sprite.height / 2))

    # A blurred, directional contact shadow gives flat fallback sprites a
    # stable relationship to the work surface.  It is derived solely from the
    # sprite silhouette, so it cannot leak target/outcome metadata.
    alpha = sprite.getchannel("A")
    shadow_alpha = alpha.point(lambda value: value * 82 // 255)
    shadow_alpha = shadow_alpha.filter(ImageFilter.GaussianBlur(radius=5))
    shadow = Image.new("RGBA", sprite.size, (27, 18, 12, 0))
    shadow.putalpha(shadow_alpha)
    image.paste(shadow, (left + 7, top + 9), shadow)
    image.paste(sprite, (left, top), sprite)


def _atomic_save_png(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        image.save(
            temporary,
            format="PNG",
            optimize=False,
            compress_level=9,
        )
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


__all__ = [
    "IMAGE_SIZE",
    "render_final",
    "render_initial",
    "render_observation",
    "render_pair",
]
