"""Typed configuration, loaded once at the CLI boundary.

Settings are read here from the environment, never scattered as
``os.environ`` lookups through the business logic. Grouped frozen dataclasses
keep each subsystem's needs explicit, so a caller is handed only the slice it
uses (e.g. extraction gets ``ModelConfig`` / ``ImageConfig``, not the whole
config). ``load_config()`` is the single entry point and owns
``load_dotenv()``; importing this module has no side effects.
"""

import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True)
class DatabaseConfig:
    path: str

    @property
    def url(self) -> str:
        return f"sqlite:///{self.path}"


@dataclass(frozen=True)
class PravdaConfig:
    url: str


@dataclass(frozen=True)
class ModelConfig:
    name: str


@dataclass(frozen=True)
class ImageConfig:
    tile_size: int
    tile_overlap: float


@dataclass(frozen=True)
class PathsConfig:
    input_base_path: str
    output_base_path: str


@dataclass(frozen=True)
class Config:
    database: DatabaseConfig
    pravda: PravdaConfig
    model: ModelConfig
    image: ImageConfig
    paths: PathsConfig


def load_config() -> Config:
    """Load and validate all settings from the environment (and ``.env``)."""
    load_dotenv()
    return Config(
        database=DatabaseConfig(path=os.environ["KOLKHOZ_DB"]),
        pravda=PravdaConfig(url=os.environ["PRAVDA_URL"]),
        model=ModelConfig(name=os.environ["OPENAI_MODEL"]),
        image=ImageConfig(
            tile_size=int(os.environ["IMAGE_TILE_SIZE"]),
            tile_overlap=float(os.environ["IMAGE_TILE_OVERLAP"]),
        ),
        paths=PathsConfig(
            input_base_path=os.environ["INPUT_BASE_PATH"],
            output_base_path=os.environ["OUTPUT_BASE_PATH"],
        ),
    )
