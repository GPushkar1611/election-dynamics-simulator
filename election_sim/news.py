"""Seeded news scheduling; ordinary shocks reuse the existing event compiler."""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import math
import numpy as np
import pandas as pd
from .events import prepare_events


@dataclass(frozen=True)
class NewsAgent:
    schedule: object = None  # DataFrame or CSV path; None uses data["events"]
    random_events_per_day: float = 0.0

    def __post_init__(self):
        if not math.isfinite(self.random_events_per_day) or self.random_events_per_day < 0:
            raise ValueError("random_events_per_day must be finite and nonnegative")

    def prepare(self, fallback, voters, candidate_ids, seed, config):
        rows = (fallback.copy() if self.schedule is None else
                self.schedule.copy() if isinstance(self.schedule, pd.DataFrame) else
                pd.read_csv(Path(self.schedule)))
        rng = np.random.default_rng([*seed, 100])
        generated = []
        for day in range(1, config.days + 1):
            for index in range(rng.poisson(self.random_events_per_day)):
                generated.append(dict(event_id=f"random-news-{day}-{index}", event_day=day,
                    duration=1, intensity=float(rng.uniform(.1, 1)), credibility=float(rng.uniform(.5, 1)),
                    target_group="General", issue="general", candidate_id=None,
                    candidate_effect=None, source_type="News", event_type="random_news"))
        if generated:
            rows = pd.concat([rows, pd.DataFrame(generated)], ignore_index=True)
        special = {}
        for row in rows.to_dict("records"):
            kind = row.get("event_type")
            if kind not in {"fact_check", "poll_release"}:
                continue
            candidate = row.get("candidate_id")
            if kind == "fact_check" and candidate not in candidate_ids:
                raise ValueError("fact_check requires a local candidate_id identifying the smear claim")
            if kind == "poll_release" and row.get("target_group") != "General":
                raise ValueError("poll_release must target General (public announcement)")
            if row.get("structural_effect", "none") not in {None, "none"} and pd.notna(row.get("structural_effect")):
                raise ValueError("announcements cannot have structural effects")
            special[str(row["event_id"])] = kind
        # Keep audience/scope/timing/dependency validation in one place.
        compiled = prepare_events(rows, voters, candidate_ids, seed, config.days,
            config.constituency_id, config.event_memory_half_life_days,
            config.event_reminder_probability)
        for event in compiled:
            event["news_kind"] = special.get(event["event_id"])
            salt = int(hashlib.sha256(event["event_id"].encode()).hexdigest()[:8], 16)
            event["news_rng"] = np.random.default_rng([*seed, salt, 101])
        return compiled

