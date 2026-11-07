"""Candidate agents: on each decision day they read a poll and pick one campaign action.

Agents never see voters' true opinions, only polls (see poll.py). Each
agent has a budget of voter contacts (recipients x active days) for the whole
campaign. The simulation skips any action the agent can no longer afford.

Strategies:
- greedy: an ordinary message to the group where the candidate polls worst.
- go_negative: smear the poll leader when trailing it by more than
  `smear_gap`; otherwise act like greedy.
- rebut: when the candidate's share fell by more than `drop_alert` since its
  previous poll (e.g. after a smear), send a broad message using `policy`;
  otherwise act like greedy.

Setting `distortion` > 0 makes the agent's own messages misleading instead
of ordinary.
"""

from dataclasses import dataclass
import json
from pathlib import Path

from ..campaign.models import TARGET_POLICIES, CampaignAction

STRATEGIES = {"greedy", "go_negative", "rebut"}


@dataclass(frozen=True)
class CandidateAgent:
    candidate_id: str
    strategy: str = "greedy"
    budget: int = 3000            # total voter contacts: recipients x active days
    start_day: int = 1            # first decision day
    decide_every: int = 3         # days between decisions; each action lasts this long
    reach_fraction: float = 0.05  # share of the constituency each action asks to reach
    intensity: float = 0.5
    distortion: float = 0.0       # > 0 makes own messages misleading
    policy: str = "uniform"       # targeting for smears and rebuttals (not "group")
    smear_gap: float = 0.10       # go_negative: smear when this far behind the leader
    drop_alert: float = 0.02      # rebut: react when own share fell this much

    def __post_init__(self) -> None:
        if not isinstance(self.candidate_id, str) or not self.candidate_id.strip():
            raise ValueError("agent candidate_id must be a non-empty string")
        if self.strategy not in STRATEGIES:
            raise ValueError(f"agent strategy must be one of {sorted(STRATEGIES)}")
        for name, minimum in (("budget", 0), ("start_day", 1), ("decide_every", 1)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"agent {name} must be an integer >= {minimum}")
        for name in ("reach_fraction", "intensity", "distortion", "smear_gap", "drop_alert"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
                raise ValueError(f"agent {name} must be a number in [0, 1]")
        if self.policy not in TARGET_POLICIES - {"group"}:
            raise ValueError(f"agent policy must be one of {sorted(TARGET_POLICIES - {'group'})}")

    def decides_on(self, day: int) -> bool:
        return day >= self.start_day and (day - self.start_day) % self.decide_every == 0

    def decide(self, day: int, days: int, poll: dict, previous_poll: dict | None = None) -> CampaignAction:
        """Choose today's action from the poll (and the agent's previous poll)."""
        shares = poll["shares"]
        own = shares[self.candidate_id]
        leader = max(shares, key=shares.get)
        duration = min(self.decide_every, days - day + 1)
        if (self.strategy == "go_negative" and leader != self.candidate_id
                and shares[leader] - own > self.smear_gap):
            return self._action(day, duration, "smear", self.policy, target_candidate_id=leader)
        if (self.strategy == "rebut" and previous_poll is not None
                and previous_poll["shares"][self.candidate_id] - own > self.drop_alert):
            return self._action(day, duration, self._own_message, self.policy)
        groups = poll["group_shares"]
        if not groups:
            return self._action(day, duration, self._own_message, "uniform")
        weakest = min(groups, key=lambda group: groups[group][self.candidate_id])
        return self._action(day, duration, self._own_message, "group", target_group=weakest)

    @property
    def _own_message(self) -> str:
        return "misleading" if self.distortion > 0 else "ordinary"

    def _action(self, day: int, duration: int, message_type: str, policy: str, **target) -> CampaignAction:
        return CampaignAction(
            action_id=f"{self.candidate_id}_day{day:02d}_{message_type}",
            candidate_id=self.candidate_id, start_day=day, duration=duration,
            intensity=self.intensity, reach_fraction=self.reach_fraction,
            message_type=message_type, target_policy=policy,
            distortion=self.distortion if message_type == "misleading" else 0.0, **target)


def check_agents(agents: list[CandidateAgent], candidate_ids: set[str]) -> list[CandidateAgent]:
    """At most one agent per candidate, and only for candidates on this ballot."""
    seen = set()
    for agent in agents:
        if agent.candidate_id not in candidate_ids:
            raise ValueError(f"agent for unknown candidate {agent.candidate_id}")
        if agent.candidate_id in seen:
            raise ValueError(f"more than one agent for candidate {agent.candidate_id}")
        seen.add(agent.candidate_id)
    return sorted(agents, key=lambda agent: agent.candidate_id)


def load_agents(path: str | Path, candidate_ids: set[str]) -> list[CandidateAgent]:
    """Load a JSON file shaped as {"agents": [ {CandidateAgent fields}, ... ]}."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or set(raw) != {"agents"} or not isinstance(raw["agents"], list):
        raise ValueError("agent file must be a JSON object with exactly an 'agents' list")
    return check_agents([CandidateAgent(**item) for item in raw["agents"]], candidate_ids)
