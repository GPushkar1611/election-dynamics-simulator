"""Small standard-library HTTP/SSE server that streams the existing simulator."""

from __future__ import annotations

import argparse
import base64
from dataclasses import replace
import json
from pathlib import Path
from queue import Empty, Queue
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse
import zlib

import numpy as np
import pandas as pd

from election_sim.agents import CandidateAgent
from election_sim.campaign.models import CampaignAction
from election_sim.config import SimulationConfig
from election_sim.dynamics import social_step
from election_sim.network import build_incoming
from election_sim.region import load_region
from election_sim.simulation import _simulate_steps

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data" / "synthetic_election_dataset"
DEFAULT_CONFIG = ROOT / "configs" / "baseline.json"
PAGE = Path(__file__).with_name("dashboard.html")
STREAM_LOCK = threading.Lock()
DAY_DELAY_SECONDS = 1.1


class _ClientDisconnected(Exception):
    """Stop a background simulation after its browser stream is gone."""


def _read_inputs():
    voters = pd.read_csv(DEFAULT_DATA / "voters.csv", usecols=["constituency_id", "voter_id"])
    candidates = pd.read_csv(DEFAULT_DATA / "candidates.csv")
    counts = voters.groupby("constituency_id").size().to_dict()
    rows = []
    for cid, group in candidates.groupby("constituency_id", sort=True):
        voter_ids = sorted(voters.loc[voters.constituency_id.eq(cid), "voter_id"].astype(str).tolist())
        rows.append({"constituency_id": int(cid), "voter_count": int(counts.get(cid, 0)),
                     "candidate_count": len(group),
                     "voter_ids": voter_ids,
                     "candidates": [{"candidate_id": str(r.candidate_id),
                                     "party_id": str(r.party_id),
                                     "party_name": str(r.party_name)}
                                    for r in group.itertuples()]})
    return rows


def _action_inputs(data: dict, cid: int, mode: str, days: int):
    local = data["candidates"].loc[data["candidates"].constituency_id.eq(cid)].sort_values(
        "candidate_id")
    candidate_ids = local.candidate_id.astype(str).tolist()
    local_voters = data["voters"].loc[data["voters"].constituency_id.eq(cid)]
    size = len(local_voters)
    if mode == "campaign":
        duration = min(6, days)
        start = min(7, days - duration + 1)
        action = CampaignAction(
            action_id=f"live_ordinary_C{cid:02d}", candidate_id=candidate_ids[0],
            start_day=start, duration=duration, intensity=0.65, reach_fraction=0.12)
        return [action], []
    if mode == "full":
        strategies = ("greedy", "go_negative", "rebut")
        agents = [CandidateAgent(candidate_id, strategy=strategies[index % len(strategies)],
                    budget=int(size * 0.08 * days), start_day=1, decide_every=3,
                    reach_fraction=0.06, intensity=0.58, smear_gap=0.08)
                  for index, candidate_id in enumerate(candidate_ids)]
        return [], agents
    return [], []


def _pack_day_state(day_state: dict) -> dict:
    """Pack every voter's state into a small compressed binary block.

    Binary layout: opinion float32[N], support float32[N x candidates], then
    social/campaign/event opinion deltas float32[N] each. Voter order is the
    canonical ID order supplied by the simulator.
    """
    cid = day_state["constituency_id"]
    support = day_state["support"]
    source = day_state["source_deltas"]
    arrays = [np.asarray(day_state["opinions"], dtype="<f4").tobytes(),
              np.asarray(support, dtype="<f4").tobytes()]
    arrays.extend(np.asarray(source[key], dtype="<f4").tobytes()
                  for key in ("social", "campaign", "events"))
    packed = base64.b64encode(zlib.compress(b"".join(arrays), level=3)).decode("ascii")
    return {"day": int(day_state["day"]), "constituency_id": int(cid),
            "summary": day_state["summary"], "candidates": day_state["candidates"],
            "voter_count": len(day_state["voter_ids"]),
            "candidate_ids": [str(row["candidate_id"]) for row in day_state["candidates"]],
            "state_blob": packed,
            "actions": day_state["actions"], "events": day_state["events"]}


def _run_state(seed: int, mode: str, focus_id: int, publish,
               day_delay_seconds: float | None = None,
               config_overrides: dict | None = None,
               cancel_event: threading.Event | None = None):
    """Coordinate all local ballots on one synchronous statewide social graph."""
    if day_delay_seconds is not None and day_delay_seconds < 0:
        raise ValueError("day_delay_seconds must be nonnegative")

    def check_cancelled():
        if cancel_event is not None and cancel_event.is_set():
            raise _ClientDisconnected()

    config = SimulationConfig(**json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8")))
    data = load_region(DEFAULT_DATA, config.days)
    check_cancelled()
    voters = data["voters"].sort_values("voter_id").reset_index(drop=True).copy()
    candidates = data["candidates"].sort_values("candidate_id").reset_index(drop=True).copy()
    edges = data["social_network"].loc[
        data["social_network"].source_voter_id.isin(set(voters.voter_id)) &
        data["social_network"].target_voter_id.isin(set(voters.voter_id))].copy()
    if mode == "social":
        config = replace(config, lambda_event=0.0, lambda_campaign=0.0)
    elif mode in {"campaign", "full"}:
        config = replace(config, lambda_campaign=0.25)
    if config_overrides:
        config = replace(config, **config_overrides)

    graph = build_incoming(voters, edges, config.rho)
    constituency_ids = sorted(int(value) for value in voters.constituency_id.unique())
    indices, engines, snapshots, results = {}, {}, {}, {}
    day_payloads: dict[int, dict[int, dict]] = {}
    package_lock = threading.Lock()
    selected_focus = focus_id if focus_id in constituency_ids else constituency_ids[0]
    sample_sizes = {}
    previous_leaders = {}
    previous_support = {}
    previous_party_support = {}

    def local_observer(cid):
        def observe(payload):
            with package_lock:
                day = payload["day"]
                day_payloads.setdefault(day, {})[cid] = payload
                if len(day_payloads[day]) != len(constituency_ids):
                    return
                complete = day_payloads.pop(day)
            tiles, candidate_rows, seat_frames, actions, events = [], [], [], [], []
            statewide_expected = {}
            switched_total = 0
            switch_pairs = {}
            opinion_sum = 0.0
            movement_sum = 0.0
            population = 0
            for local_id in constituency_ids:
                item = complete[local_id]
                summary = item["summary"]
                pop = sample_sizes[local_id]
                population += pop
                opinion_sum += summary["opinion_mean"] * pop
                movement_sum += summary["mean_absolute_daily_movement"] * pop
                local_candidates = item["candidates"]
                supports = item["support"]
                leading = supports.argmax(axis=1)
                lead_counts = np.bincount(leading, minlength=supports.shape[1])
                prior = previous_leaders.get(local_id)
                if prior is not None:
                    changed = prior != leading
                    switched_total += int(changed.sum())
                    if changed.any():
                        pairs = np.stack([prior[changed], leading[changed]], axis=1)
                        keys, counts = np.unique(pairs, axis=0, return_counts=True)
                        for (old, new), count in zip(keys, counts):
                            pair = f"{item['candidates'][int(old)]['candidate_id']}->{item['candidates'][int(new)]['candidate_id']}"
                            switch_pairs[pair] = switch_pairs.get(pair, 0) + int(count)
                previous_leaders[local_id] = leading.copy()
                leader_index = int(np.argmax(lead_counts))
                leader = local_candidates[leader_index]
                candidate_party = party_by_candidate.get(leader["candidate_id"], "IND")
                tiles.append({"constituency_id": local_id, "leader": leader["candidate_id"],
                              "party_id": candidate_party,
                              "support": float(lead_counts[leader_index] / len(leading)),
                              "leading_voters": int(lead_counts[leader_index]),
                              "opinion_mean": summary["opinion_mean"],
                              "movement": summary["mean_absolute_daily_movement"],
                              "smear_believers": summary["smear_believers"]})
                candidate_rows.extend({**row, "constituency_id": local_id,
                                       "party_id": party_by_candidate.get(row["candidate_id"], "IND"),
                                       "party_name": party_name_by_candidate.get(
                                           row["candidate_id"], "Independent")}
                                      for row in local_candidates)
                local_leading_party = party_by_candidate.get(leader["candidate_id"], "IND")
                for candidate_index, candidate in enumerate(local_candidates):
                    candidate_id = candidate["candidate_id"]
                    old_support = previous_support.get(local_id, {}).get(candidate_id)
                    candidate_rows[-len(local_candidates) + candidate_index]["support_change"] = (
                        None if old_support is None else candidate["mean_support"] - old_support)
                    party_id = party_by_candidate.get(candidate_id, "IND")
                    record = statewide_expected.setdefault(party_id, {
                        "candidate_id": party_id,
                        "party_name": party_name_by_candidate.get(candidate_id, party_id),
                        "party_id": party_id,
                        "expected_support": 0.0, "voters_leading": 0,
                        "constituencies_led": 0})
                    record["expected_support"] += candidate["mean_support"] * pop
                    record["voters_leading"] += int(lead_counts[candidate_index])
                    if party_id == local_leading_party and leader_index == candidate_index:
                        record["constituencies_led"] += 1
                previous_support[local_id] = {row["candidate_id"]: row["mean_support"]
                                              for row in local_candidates}
                seat_frames.append(_pack_day_state(item))
                if local_id == selected_focus:
                    actions.extend(item["actions"])
                    events.extend(item["events"])
            for record in statewide_expected.values():
                record["projected_share"] = record["expected_support"] / population
                old_share = previous_party_support.get(record["party_id"])
                record["support_change"] = (None if old_share is None else
                                             record["projected_share"] - old_share)
            previous_party_support.clear()
            previous_party_support.update({row["party_id"]: row["projected_share"]
                                           for row in statewide_expected.values()})
            statewide_candidates = sorted(statewide_expected.values(),
                key=lambda row: (-row["expected_support"], row["candidate_id"]))
            payload = {"day": day, "days": config.days, "mode": mode,
                       "focus_constituency": selected_focus,
                       "summary": {"opinion_mean": opinion_sum / population,
                                   "mean_absolute_daily_movement": movement_sum / population,
                                   "voter_count": population,
                                   "constituency_count": len(constituency_ids)},
                       "constituencies": tiles, "candidates": candidate_rows,
                       "state_candidates": statewide_candidates,
                       "seat_frames": seat_frames,
                       "switched_voters": switched_total,
                       "switch_pairs": switch_pairs,
                       "actions": actions, "events": events}
            publish("day", payload)
            if day < config.days:
                time.sleep(DAY_DELAY_SECONDS if day_delay_seconds is None else day_delay_seconds)
        return observe

    party_by_candidate = dict(zip(candidates.candidate_id.astype(str),
                                  candidates.party_id.astype(str)))
    party_name_by_candidate = dict(zip(candidates.candidate_id.astype(str),
                                       candidates.party_name.astype(str)))
    for cid in constituency_ids:
        check_cancelled()
        index = np.flatnonzero(voters.constituency_id.to_numpy() == cid)
        indices[cid] = index
        local_data = {
            "voters": voters.iloc[index].copy(),
            "candidates": candidates.loc[candidates.constituency_id.eq(cid)].copy(),
            "social_network": edges.loc[edges.source_voter_id.isin(
                set(voters.iloc[index].voter_id)) & edges.target_voter_id.isin(
                set(voters.iloc[index].voter_id))].copy(),
            "events": data["events"].copy(),
        }
        actions, agents = _action_inputs(data, cid, mode, config.days)
        local_config = replace(config, constituency_id=cid)
        observer = local_observer(cid)
        engine = _simulate_steps(local_config, local_data, seed, actions,
                                 agents=agents, on_day=observer)
        engines[cid] = engine
        sample_sizes[cid] = len(index)
        snapshots[cid] = next(engine)

    # Each engine has now emitted day zero; callbacks publish it once all seats
    # have reached the same point. Advance all voters synchronously each day.
    all_opinions = np.empty(len(voters), dtype=float)
    all_susceptibility = np.empty(len(voters), dtype=float)
    all_confidence = np.empty(len(voters), dtype=float)
    for day in range(1, config.days + 1):
        check_cancelled()
        for cid, snapshot in snapshots.items():
            check_cancelled()
            index = indices[cid]
            all_opinions[index], all_susceptibility[index], all_confidence[index] = snapshot[:3]
        updated, _ = social_step(all_opinions, all_susceptibility, all_confidence,
                                 graph, config.lambda_social)
        for cid in constituency_ids:
            try:
                snapshots[cid] = engines[cid].send(updated[indices[cid]])
            except StopIteration as finished:
                results[cid] = finished.value

    results_payload = []
    for cid in constituency_ids:
        result = results[cid]
        for row in result.candidate_results.to_dict("records"):
            results_payload.append({"constituency_id": cid, "candidate_id": row["candidate_id"],
                                    "party_id": party_by_candidate.get(row["candidate_id"], "IND"),
                                    "votes": int(row["votes"]),
                                    "vote_share": float(row["vote_share"]),
                                    "winner": row["candidate_id"] in result.turnout["winner_ids"],
                                    "final_mean_support": float(row["final_mean_support"])})
    publish("results", {"candidates": results_payload,
                         "turnouts": [dict(result.turnout, constituency_id=cid)
                                      for cid, result in results.items()],
                         "seats": len(constituency_ids)})


class DashboardHandler(BaseHTTPRequestHandler):
    server_version = "ElectionLive/1.0"

    def _cors(self):
        origin = self.headers.get("Origin")
        if not origin:
            return
        parsed = urlparse(origin) if origin != "null" else None
        is_loopback = parsed is not None and parsed.scheme in {"http", "https"} and parsed.hostname in {
            "localhost", "127.0.0.1", "::1"
        }
        if origin == "null" or is_loopback:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")

    def _json(self, value, status=200):
        body = json.dumps(value, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/":
            body = PAGE.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if parsed.path == "/api/constituencies":
            try:
                self._json(_read_inputs())
            except (OSError, KeyError, ValueError, pd.errors.ParserError) as exc:
                self._json({"error": str(exc)}, 500)
            return
        if parsed.path == "/api/stream":
            query = parse_qs(parsed.query)
            try:
                seed = int(query.get("seed", ["42"])[0])
                focus = int(query.get("focus", ["1"])[0])
                mode = query.get("mode", ["full"])[0]
                if seed < 0 or mode not in {"social", "campaign", "full"}:
                    raise ValueError("Choose a nonnegative seed and a supported scenario")
            except ValueError as exc:
                self._json({"error": str(exc)}, 400)
                return
            if not STREAM_LOCK.acquire(blocking=False):
                message = json.dumps({"message": "A statewide run is already in progress; please wait for it to stop."})
                body = f"event: error\ndata: {message}\n\n".encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-cache, no-transform")
                self.send_header("Connection", "close")
                self._cors()
                self.end_headers()
                try:
                    self.wfile.write(body)
                    self.wfile.flush()
                except OSError:
                    self.close_connection = True
                return
            # Keep every day for the browser's replay timeline. Runs are
            # serialized, so only one stream can accumulate frames at a time.
            channel = Queue()
            disconnected = threading.Event()

            def publish(kind, data):
                if disconnected.is_set():
                    raise _ClientDisconnected()
                channel.put_nowait((kind, data))

            def work():
                try:
                    publish("status", {"message": "Loading and validating the supplied synthetic dataset…"})
                    _run_state(seed, mode, focus, publish, cancel_event=disconnected)
                    publish("complete", {"message": "Simulation complete"})
                except _ClientDisconnected:
                    pass
                except Exception as exc:  # stream errors are surfaced in the dashboard
                    if not disconnected.is_set():
                        channel.put_nowait(("error", {"message": f"{type(exc).__name__}: {exc}"}))
                finally:
                    STREAM_LOCK.release()

            threading.Thread(target=work, name="election-live-run", daemon=True).start()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache, no-transform")
            self.send_header("Connection", "keep-alive")
            self._cors()
            self.end_headers()
            try:
                while True:
                    try:
                        kind, value = channel.get(timeout=1)
                        self.wfile.write(f"event: {kind}\ndata: ".encode("utf-8"))
                        self.wfile.write(json.dumps(value, allow_nan=False).encode("utf-8"))
                        self.wfile.write(b"\n\n")
                        self.wfile.flush()
                        if kind in {"complete", "error"}:
                            self.close_connection = True
                            break
                    except Empty:
                        self.wfile.write(b": alive\n\n")
                        self.wfile.flush()
            except OSError:
                disconnected.set()
                self.close_connection = True
            return
        self.send_error(404)

    def log_message(self, fmt, *args):
        print(f"[live-dashboard] {self.address_string()} - {fmt % args}")


def main():
    parser = argparse.ArgumentParser(description="Run the local live election dashboard")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: localhost only)")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), DashboardHandler)
    print("Synthetic election dashboard is ready")
    print(f"Open http://{args.host}:{args.port} in your browser")
    print("Press Ctrl+C to stop the dashboard server")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping dashboard server")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
