"""Synchronous regional social dynamics with independent local ballots and agents."""
from dataclasses import asdict, dataclass, replace
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
from constituency_loader.validation import validate_data
from .agents import CandidateAgent
from .campaign.models import CampaignAction
from .config import SimulationConfig
from .dynamics import social_step
from .network import build_incoming
from .simulation import _simulate_steps, _fingerprint


def load_region(data_dir, days=30, events_file=None):
    """Read and validate all four CSVs once; simulate_region selects the region."""
    root = Path(data_dir)
    data = {name: pd.read_csv(root / filename) for name, filename in (
        ("voters", "voters.csv"), ("candidates", "candidates.csv"),
        ("social_network", "social_network.csv"))}
    data["events"] = pd.read_csv(events_file or root / "political_events.csv")
    validate_data(data["voters"], data["social_network"], data["candidates"], data["events"], days)
    return data


@dataclass
class RegionResult:
    constituencies: dict
    candidate_results: pd.DataFrame
    turnout: pd.DataFrame
    seats: pd.DataFrame
    daily_summary: pd.DataFrame
    daily_support: pd.DataFrame
    metadata: dict
    history: dict | None = None

    def save(self, output_dir):
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=False)
        for name in ("candidate_results", "turnout", "seats", "daily_summary", "daily_support"):
            getattr(self, name).to_csv(directory / f"{name}.csv", index=False)
        for cid, result in self.constituencies.items():
            result.save(directory / f"C{cid:02d}")
        (directory / "manifest.json").write_text(json.dumps(self.metadata, indent=2, allow_nan=False)+"\n")
        if self.history is not None:
            np.savez_compressed(directory / "region_history.npz", **self.history)


def _map_party(candidates, constituency_id, party_id):
    rows = candidates.loc[(candidates.constituency_id == constituency_id) & candidates.party_id.eq(party_id)]
    if len(rows) != 1:
        raise ValueError(f"party {party_id!r} needs exactly one candidate in constituency {constituency_id}")
    return rows.candidate_id.iloc[0]


def simulate_region(config: SimulationConfig, constituency_ids, seed=42, spillover=0.0,
                    *, data=None, data_dir=None, events_file=None, campaign_actions=None,
                    agents=None, party_campaigns=None, party_agents=None, record_history=False):
    """Run one region with a single synchronous social snapshot per day.

    Pass full tables as data or load CSVs via data_dir. Party campaigns/agents
    are lists of dictionaries replacing candidate_id with party_id (and smear
    target_candidate_id with target_party_id). Reach and budgets apply locally.
    spillover scales raw cross-constituency social tie weights before normalization.
    Local campaigns, smear claims and news remain local; social opinions spill over.
    """
    if isinstance(spillover, bool) or not isinstance(spillover, (int, float)) or not math.isfinite(spillover) or not 0 <= spillover <= 1:
        raise ValueError("spillover must be finite and in [0, 1]")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    ids = list(constituency_ids)
    if not ids or any(isinstance(cid, bool) or not isinstance(cid, int) or cid < 1 for cid in ids) or len(set(ids)) != len(ids):
        raise ValueError("constituency_ids must be nonempty, unique positive integers")
    ids.sort()
    if data is None:
        root = data_dir or Path(__file__).resolve().parents[1] / "data/synthetic_election_dataset"
        data = load_region(root, config.days, events_file)
    voters = data["voters"].loc[data["voters"].constituency_id.isin(ids)].sort_values("voter_id").reset_index(drop=True).copy()
    candidates = data["candidates"].loc[data["candidates"].constituency_id.isin(ids)].sort_values("candidate_id").reset_index(drop=True).copy()
    if set(voters.constituency_id) != set(ids) or set(candidates.constituency_id) != set(ids):
        raise ValueError("every selected constituency must have voters and candidates")
    if "region_id" not in voters or voters.region_id.isna().any() or voters.region_id.nunique() != 1:
        raise ValueError("selected constituencies must belong to one region")
    if candidates.candidate_id.duplicated().any() or candidates.candidate_id.isna().any():
        raise ValueError("regional candidate IDs must be present and unique")
    if "region_id" in candidates and not candidates.region_id.eq(voters.region_id.iloc[0]).all():
        raise ValueError("candidate and voter regions must match")
    if "party_id" not in candidates or candidates.party_id.isna().any():
        raise ValueError("regional candidates need party_id for seat aggregation")
    selected = set(voters.voter_id)
    original_edges = data["social_network"]
    inside = original_edges.source_voter_id.isin(selected) & original_edges.target_voter_id.isin(selected)
    edges = original_edges.loc[inside].sort_values(["target_voter_id", "source_voter_id"]).copy()
    # Validate original edges before multiplying or dropping cross ties.
    build_incoming(voters, edges, config.rho)
    geography = voters.set_index("voter_id").constituency_id
    cross = edges.source_voter_id.map(geography).ne(edges.target_voter_id.map(geography))
    cross_count = int(cross.sum())
    if spillover == 0:
        effective_edges = edges.loc[~cross].copy()
    else:
        effective_edges = edges.copy()
        effective_edges.loc[cross, "edge_weight"] *= spillover
    graph = build_incoming(voters, effective_edges, config.rho)
    actions = list(campaign_actions or [])
    all_agents = list(agents or [])
    for spec in party_campaigns or []:
        for cid in ids:
            row = dict(spec)
            party = row.pop("party_id")
            row["candidate_id"] = _map_party(candidates, cid, party)
            row["action_id"] = f"{row['action_id']}_C{cid:02d}"
            if "target_party_id" in row:
                row["target_candidate_id"] = _map_party(candidates, cid, row.pop("target_party_id"))
            actions.append(CampaignAction(**row))
    for spec in party_agents or []:
        for cid in ids:
            row = dict(spec)
            row["candidate_id"] = _map_party(candidates, cid, row.pop("party_id"))
            all_agents.append(CandidateAgent(**row))
    candidate_geography = candidates.set_index("candidate_id").constituency_id.to_dict()
    for action in actions:
        if action.candidate_id not in candidate_geography:
            raise ValueError("campaign candidate is outside selected constituencies")
        if action.target_candidate_id and candidate_geography.get(action.target_candidate_id) != candidate_geography[action.candidate_id]:
            raise ValueError("smear target must be on the same local ballot")
    if any(agent.candidate_id not in candidate_geography for agent in all_agents):
        raise ValueError("agent candidate is outside selected constituencies")
    events = data.get("events", pd.DataFrame())
    engines, snapshots, indices = {}, {}, {}
    for cid in ids:
        index = np.flatnonzero(voters.constituency_id.to_numpy() == cid)
        indices[cid] = index
        local_voters = voters.iloc[index].copy()
        local_ids = set(local_voters.voter_id)
        local_candidates = candidates.loc[candidates.constituency_id == cid].copy()
        local_events = events.copy()
        if not events.empty:
            if not {"region_id", "constituency_id"}.issubset(events):
                raise ValueError("regional events require explicit geography columns")
            cons = pd.to_numeric(events.constituency_id, errors="coerce")
            region = pd.to_numeric(events.region_id, errors="coerce")
            local_events = events.loc[cons.eq(cid) | (cons.isna() & (region.eq(voters.region_id.iloc[0]) | region.isna()))].copy()
            # An out-of-ballot fact-check addresses no claim on this ballot.
            if "event_type" in local_events:
                local_events = local_events.loc[~local_events.event_type.eq("fact_check") | local_events.candidate_id.isin(local_candidates.candidate_id)]
        local_data = {"voters": local_voters, "candidates": local_candidates,
            "social_network": edges.loc[edges.source_voter_id.isin(local_ids) & edges.target_voter_id.isin(local_ids)],
            "events": local_events}
        engine = _simulate_steps(replace(config, constituency_id=cid), local_data, seed,
            [a for a in actions if candidate_geography[a.candidate_id] == cid], record_history,
            [a for a in all_agents if candidate_geography[a.candidate_id] == cid])
        engines[cid] = engine
        snapshots[cid] = next(engine)
    x = np.empty(len(voters)); susceptibility = np.empty(len(voters)); confidence = np.empty(len(voters))
    results = {}
    for day in range(1, config.days + 1):
        for cid, snapshot in snapshots.items():
            index = indices[cid]
            x[index], susceptibility[index], confidence[index] = snapshot[:3]
        updated, _ = social_step(x, susceptibility, confidence, graph, config.lambda_social)
        for cid in ids:
            try:
                snapshots[cid] = engines[cid].send(updated[indices[cid]])
            except StopIteration as finished:
                if day != config.days:
                    raise RuntimeError("constituency engine ended early")
                results[cid] = finished.value
    def combine(name):
        return pd.concat([getattr(results[cid], name).assign(constituency_id=cid) for cid in ids], ignore_index=True)
    table = combine("candidate_results").merge(candidates[["candidate_id", "party_id"]], on="candidate_id", validate="one_to_one")
    turnout = pd.DataFrame([{**results[cid].turnout, "constituency_id": cid} for cid in ids])
    seats = []
    for party in sorted(candidates.party_id.unique()):
        won, tied = 0, 0
        for cid in ids:
            winners = results[cid].turnout["winner_ids"]
            parties = [candidates.set_index("candidate_id").loc[winner, "party_id"] for winner in winners]
            won += int(len(winners) == 1 and party in parties)
            tied += int(len(winners) > 1 and party in parties)
        seats.append({"party_id": party, "seats_won": won, "tied_constituencies": tied})
    metadata = {"synthetic": True, "config": asdict(config), "master_seed": seed,
        "constituency_ids": ids, "region_id": int(voters.region_id.iloc[0]), "spillover": spillover,
        "voter_count": len(voters), "selected_edge_count": len(edges),
        "cross_constituency_edge_count": cross_count, "effective_edge_count": len(effective_edges),
        "excluded_boundary_edge_count": int(((original_edges.source_voter_id.isin(selected) | original_edges.target_voter_id.isin(selected)) & ~inside).sum()),
        "edge_sha256": _fingerprint(edges),
        "spillover_model": "Cross social tie weights scaled before incoming normalization; campaign and smear claims remain local",
        "seed_policy": "Same master seed per local engine, preserving standalone controls",
        "unresolved_constituencies": [cid for cid in ids if len(results[cid].turnout["winner_ids"]) != 1]}
    history = None
    if record_history:
        opinions = np.empty((config.days + 1, len(voters)), dtype=np.float32)
        for cid in ids:
            opinions[:, indices[cid]] = results[cid].history["opinions"]
        history = {"voter_ids": voters.voter_id.to_numpy(dtype=str),
                   "constituency_ids": voters.constituency_id.to_numpy(), "opinions": opinions}
    for cid, result in results.items():
        result.metadata["regional_context"] = {"constituency_ids": ids, "spillover": spillover,
            "edge_sha256": metadata["edge_sha256"], "social_graph": "regional; local eligible_edges diagnostic counts internal edges"}
    return RegionResult(results, table, turnout, pd.DataFrame(seats), combine("daily_summary"),
                        combine("daily_support"), metadata, history)


def spillover_sweep(config, constituency_ids, seed=42, values=(0, .05, .1, .2), **kwargs):
    """Paired spillover sweep; returns one candidate-result row per coefficient."""
    ids = list(constituency_ids)
    return pd.concat([simulate_region(config, ids, seed, value, **kwargs).candidate_results.assign(
        spillover=value, seed=seed) for value in values], ignore_index=True)
