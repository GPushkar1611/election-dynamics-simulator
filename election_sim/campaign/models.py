"""Campaign action schema and JSON loading for the first campaign milestone.

Uniformly targeted ordinary and misleading actions are supported here.
Additional targeting policies will build on the same action model.
"""

from dataclasses import dataclass
import json
import math
from pathlib import Path


@dataclass(frozen=True)
class CampaignAction:
    action_id: str
    candidate_id: str
    start_day: int
    duration: int
    intensity: float
    reach_fraction: float
    message_type: str = "ordinary"
    target_policy: str = "uniform"
    message_position: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.action_id, str) or not self.action_id.strip():
            raise ValueError("campaign action_id must be a non-empty string")
        if not isinstance(self.candidate_id, str) or not self.candidate_id.strip():
            raise ValueError("campaign candidate_id must be a non-empty string")
        for name in ("start_day", "duration"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"campaign {name} must be a positive integer")
        for name in ("intensity", "reach_fraction"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
                raise ValueError(f"campaign {name} must be finite")
            if not 0 <= value <= 1:
                raise ValueError(f"campaign {name} must be in [0, 1]")
        if self.message_type not in {"ordinary", "misleading"}:
            raise ValueError("campaign message_type must be ordinary or misleading")
        if self.target_policy != "uniform":
            raise ValueError("campaign implementation currently supports only target_policy='uniform'")
        if self.message_position is not None:
            if (not isinstance(self.message_position, (int, float)) or isinstance(self.message_position, bool)
                    or not math.isfinite(self.message_position) or not -1 <= self.message_position <= 1):
                raise ValueError("campaign message_position must be finite and in [-1, 1]")
        if self.message_type == "ordinary" and self.message_position is not None:
            raise ValueError("ordinary campaigns derive their message from the candidate platform; omit message_position")
        if self.message_type == "misleading" and self.message_position is None:
            raise ValueError("misleading campaigns require message_position")

    def active_on(self, day: int) -> bool:
        return self.start_day <= day < self.start_day + self.duration


def load_actions(path: str | Path, days: int, candidate_ids: set[str]) -> list[CampaignAction]:
    """Load and validate a JSON file shaped as {"actions": [ ... ]}."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or set(raw) != {"actions"} or not isinstance(raw["actions"], list):
        raise ValueError("campaign file must be a JSON object with exactly an 'actions' list")
    actions = [CampaignAction(**item) for item in raw["actions"]]
    ids = [action.action_id for action in actions]
    if len(ids) != len(set(ids)):
        raise ValueError("campaign action_id values must be unique")
    for action in actions:
        if action.candidate_id not in candidate_ids:
            raise ValueError(f"campaign action {action.action_id} references unknown candidate {action.candidate_id}")
        if action.start_day + action.duration - 1 > days:
            raise ValueError(f"campaign action {action.action_id} extends beyond day {days}")
    return actions
