from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dataset.episode import DatasetEpisode, DatasetEpisodeError


OPAQUE_SCENARIO_ID = re.compile(r"^ep-[0-9a-f]{24}$")


class BenchmarkEpisodeError(DatasetEpisodeError):
    pass


@dataclass(frozen=True, slots=True)
class BenchmarkEpisode:
    """A generated endpoint packet with an explicit oracle boundary.

    ``manifest`` is for the out-of-band benchmark scorer.  Callers constructing
    model prompts must use :meth:`model_context`, which deliberately excludes the
    target, expected outcome, simulator state, and filesystem paths.
    """

    episode: DatasetEpisode
    scenario_id: str
    manifest: dict[str, Any]

    @classmethod
    def from_path(
        cls, path: str | Path, workspace_root: str | Path
    ) -> BenchmarkEpisode:
        episode = DatasetEpisode.from_path(path, workspace_root)
        manifest_path = episode.directory / "manifest.json"
        if not manifest_path.is_file():
            raise BenchmarkEpisodeError(
                f"Benchmark episode has no manifest.json: {episode.directory}"
            )
        try:
            with manifest_path.open(encoding="utf-8") as stream:
                manifest = json.load(stream)
        except (OSError, json.JSONDecodeError) as error:
            raise BenchmarkEpisodeError(
                f"Benchmark manifest is not valid JSON: {manifest_path}"
            ) from error
        if not isinstance(manifest, dict):
            raise BenchmarkEpisodeError(
                f"Benchmark manifest must contain an object: {manifest_path}"
            )
        scenario_id = str(manifest.get("scenario_id", ""))
        if scenario_id != episode.directory.name:
            raise BenchmarkEpisodeError(
                "Benchmark scenario_id must match its opaque episode directory."
            )
        if not OPAQUE_SCENARIO_ID.fullmatch(scenario_id):
            raise BenchmarkEpisodeError(
                f"Benchmark scenario_id is not canonical: {scenario_id!r}"
            )
        frames = manifest.get("frames")
        frame_names = (
            {str(value) for value in frames.values()}
            if isinstance(frames, dict)
            else set()
        )
        if frame_names != {"1.png", "2.png"}:
            raise BenchmarkEpisodeError(
                "Benchmark manifest must bind initial/final frames to 1.png and 2.png."
            )
        return cls(episode=episode, scenario_id=scenario_id, manifest=manifest)

    @property
    def instruction(self) -> str:
        """The human task query stored by the offline scenario author."""

        target = self.manifest.get("target")
        return str(target.get("instruction", "")) if isinstance(target, dict) else ""

    def model_context(self) -> dict[str, Any]:
        """Return the only manifest-derived context safe to expose to an agent."""

        # Scenario IDs deliberately distinguish target/outcome counterfactuals, so
        # even an opaque scenario ID would be a model-visible side channel.  The
        # observation ID is instead a function of the initial pixels only.  Every
        # counterfactual sibling with the same initial scene therefore receives
        # exactly the same context.
        observation_digest = hashlib.sha256(
            self.episode.initial_frame.read_bytes()
        ).hexdigest()[:24]
        return {
            "observation_id": f"obs-{observation_digest}",
            "available_modalities": ["initial_rgb"],
        }

    def oracle(self) -> dict[str, Any]:
        """Return a shallow copy for an out-of-band scorer, never a model prompt."""

        return dict(self.manifest)
