from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class PaperMargins:
    top: float
    bottom: float
    left: float
    right: float


SizeId = Literal[
    "small-1-inch",
    "1-inch",
    "large-1-inch",
    "small-2-inch",
    "2-inch",
    "3-inch",
    "id-card",
]


@dataclass(frozen=True)
class SizeOption:
    id: SizeId
    label: str
    dimensions: str
    aspect_ratio: float
    physical_width: float  # mm
    physical_height: float  # mm


SIZE_OPTIONS: list[SizeOption] = [
    SizeOption("small-1-inch", "Small 1 Inch", "22×32mm", 22 / 32, 22, 32),
    SizeOption("1-inch", "1 Inch", "25×35mm", 25 / 35, 25, 35),
    SizeOption("large-1-inch", "Large 1 Inch", "33×48mm", 33 / 48, 33, 48),
    SizeOption("small-2-inch", "Small 2 Inch", "35×45mm", 35 / 45, 35, 45),
    SizeOption("2-inch", "2 Inch", "35×53mm", 35 / 53, 35, 53),
    SizeOption("3-inch", "3 Inch", "35×52mm", 35 / 52, 35, 52),
    SizeOption("id-card", "China ID Card", "26×32mm", 26 / 32, 26, 32),
]

SIZE_OPTIONS_BY_ID: dict[str, SizeOption] = {s.id: s for s in SIZE_OPTIONS}
