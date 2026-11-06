"""Campaign action schema and JSON loading (plan sections 2.3 and 3.3).

Two message types are supported, and the type is the action's explicit
truth label:

- ordinary: a truthful message that pulls recipients' opinions toward the
  sponsoring candidate's real platform.
- smear: a fabricated negative claim about another candidate. It does not
  move opinions; it lowers recipients' utility for the target candidate.

``target_policy`` decides who receives an action. Only "uniform" exists
today; new policies are added in ``effects.select_recipients``.
"""

from dataclasses import dataclass
import json
import math
from pathlib import Path

MESSAGE_TYPES = {"ordinary", "smear"}
TARGET_POLICIES = {"uniform"}


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
    target_candidate_id: str | None = None

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
        if self.message_type not in MESSAGE_TYPES:
            raise ValueError(f"campaign message_type must be one of {sorted(MESSAGE_TYPES)}")
        if self.target_policy not in TARGET_POLICIES:
            raise ValueError(f"campaign target_policy must be one of {sorted(TARGET_POLICIES)}")
        if self.message_type == "ordinary" and self.target_candidate_id is not None:
            raise ValueError("ordinary campaigns promote their own candidate; omit target_candidate_id")
        if self.message_type == "smear":
            if not isinstance(self.target_candidate_id, str) or not self.target_candidate_id.strip():
                raise ValueError("smear campaigns require target_candidate_id")
            if self.target_candidate_id == self.candidate_id:
                raise ValueError("a smear campaign cannot target its own candidate")

    def active_on(self, day: int) -> bool:
        return self.start_day <= day < self.start_day + self.duration


def check_actions(actions: list[CampaignAction], days: int, candidate_ids: set[str]) -> list[CampaignAction]:
    """Validate actions against a run and return them in canonical action_id order."""
    ids = [action.action_id for action in actions]
    if len(ids) != len(set(ids)):
        raise ValueError("campaign action_id values must be unique")
    for action in actions:
        for candidate in (action.candidate_id, action.target_candidate_id):
            if candidate is not None and candidate not in candidate_ids:
                raise ValueError(f"campaign action {action.action_id} references unknown candidate {candidate}")
        if action.start_day + action.duration - 1 > days:
            raise ValueError(f"campaign action {action.action_id} extends beyond day {days}")
    return sorted(actions, key=lambda action: action.action_id)


def load_actions(path: str | Path, days: int, candidate_ids: set[str]) -> list[CampaignAction]:
    """Load and validate a JSON file shaped as {"actions": [ ... ]}."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or set(raw) != {"actions"} or not isinstance(raw["actions"], list):
        raise ValueError("campaign file must be a JSON object with exactly an 'actions' list")
    return check_actions([CampaignAction(**item) for item in raw["actions"]], days, candidate_ids)
