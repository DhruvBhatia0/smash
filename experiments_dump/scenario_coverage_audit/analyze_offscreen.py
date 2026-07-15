from collections import Counter
from pathlib import Path
import json
import os

from slippi import Game
from slippi.event import StateFlags


ROOT = Path(os.environ.get("AUDIT_ROOT", "/home/daytona/smash-scenario-audit"))
REPLAY_DIR = Path(os.environ.get("AUDIT_REPLAY_DIR", ROOT / "replays"))
FIRST_PLAYABLE = -39


def finish_episode(episodes, replay_file, port, start, end):
    if start is None:
        return
    episodes.append(
        {
            "replayFile": replay_file,
            "port": port,
            "startFrame": start,
            "endFrame": end,
            "frames": end - start + 1,
            "seconds": (end - start + 1) / 60,
        }
    )


result = {
    "format": "PY_SLIPPI_EXACT_OFFSCREEN_AUDIT_V1",
    "definition": "Post-frame StateFlags.OFF_SCREEN (magnifying-glass/offscreen flag), sampled at native 60 Hz and 20 Hz",
    "replaysFound": 0,
    "replaysAnalyzed": 0,
    "failures": [],
    "worldFrames60": 0,
    "playerFrames60": 0,
    "worldFrames20": 0,
    "playerFrames20": 0,
    "flaggedOffscreenPlayerFrames60": 0,
    "offscreenPlayerFrames60": 0,
    "offscreenPlayerFrames20": 0,
    "deadPlayerFrames60": 0,
    "configuration20": Counter(),
    "replaysWithOffscreen": set(),
    "episodes": [],
    "replaySummaries": [],
}

replay_files = sorted(REPLAY_DIR.glob("*.slp"))
result["replaysFound"] = len(replay_files)
for index, replay_path in enumerate(replay_files, 1):
    try:
        game = Game(str(replay_path))
        ports = [port for port in range(4) if any(frame.ports[port] and frame.ports[port].leader for frame in game.frames[:200])]
        if len(ports) != 2:
            raise RuntimeError(f"expected two leader ports, got {ports}")
        starts = {port: None for port in ports}
        last_frame = None
        replay_offscreen60 = 0
        replay_offscreen20 = 0
        replay_world60 = 0
        replay_world20 = 0
        replay_configs = Counter()
        for frame in game.frames:
            if frame.index < FIRST_PLAYABLE:
                continue
            values = []
            complete = True
            for port in ports:
                leader = frame.ports[port].leader if frame.ports[port] else None
                if leader is None:
                    complete = False
                    break
                values.append(leader.post)
            if not complete:
                continue
            last_frame = frame.index
            replay_world60 += 1
            result["worldFrames60"] += 1
            result["playerFrames60"] += 2
            raw_flags = [bool(post.flags & StateFlags.OFF_SCREEN) for post in values]
            dead = [bool(post.flags & StateFlags.DEAD) for post in values]
            # The raw flag can overlap death/KO states. The primary scenario label means an alive
            # player is outside the camera and represented by the magnifying glass.
            flags = [flag and not is_dead for flag, is_dead in zip(raw_flags, dead)]
            count = sum(flags)
            result["flaggedOffscreenPlayerFrames60"] += sum(raw_flags)
            replay_offscreen60 += count
            result["offscreenPlayerFrames60"] += count
            result["deadPlayerFrames60"] += sum(dead)
            for port, flag in zip(ports, flags):
                if flag and starts[port] is None:
                    starts[port] = frame.index
                if not flag and starts[port] is not None:
                    finish_episode(result["episodes"], replay_path.name, port, starts[port], frame.index - 1)
                    starts[port] = None
            if (frame.index - FIRST_PLAYABLE) % 3 == 0:
                replay_world20 += 1
                result["worldFrames20"] += 1
                result["playerFrames20"] += 2
                replay_offscreen20 += count
                result["offscreenPlayerFrames20"] += count
                config = "both-offscreen" if count == 2 else "player0-only-offscreen" if flags[0] else "player1-only-offscreen" if flags[1] else "none-offscreen"
                result["configuration20"][config] += 1
                replay_configs[config] += 1
        for port in ports:
            if starts[port] is not None and last_frame is not None:
                finish_episode(result["episodes"], replay_path.name, port, starts[port], last_frame)
        if replay_offscreen60:
            result["replaysWithOffscreen"].add(replay_path.name)
        result["replaySummaries"].append(
            {
                "replayFile": replay_path.name,
                "ports": ports,
                "worldFrames60": replay_world60,
                "worldFrames20": replay_world20,
                "offscreenPlayerFrames60": replay_offscreen60,
                "offscreenPlayerFrames20": replay_offscreen20,
                "configuration20": dict(replay_configs),
            }
        )
        result["replaysAnalyzed"] += 1
    except Exception as error:
        result["failures"].append({"replayFile": replay_path.name, "error": repr(error)})
    if index % 10 == 0:
        print(f"processed {index}/{len(replay_files)}", flush=True)

result["episodes"].sort(key=lambda episode: episode["frames"], reverse=True)
result["longestEpisodes"] = result["episodes"][:20]
result["episodeCount"] = len(result["episodes"])
result["replaysWithOffscreen"] = len(result["replaysWithOffscreen"])
result["configuration20"] = dict(result["configuration20"])
result["offscreenPlayerFrameRate60"] = result["offscreenPlayerFrames60"] / max(1, result["playerFrames60"])
result["offscreenWorldFrameRate20"] = (
    sum(value for key, value in result["configuration20"].items() if key != "none-offscreen")
    / max(1, result["worldFrames20"])
)
(ROOT / "offscreen-analysis.json").write_text(json.dumps(result, indent=2) + "\n")
print(
    json.dumps(
        {
            "output": str(ROOT / "offscreen-analysis.json"),
            "analyzed": result["replaysAnalyzed"],
            "failures": len(result["failures"]),
            "replaysWithOffscreen": result["replaysWithOffscreen"],
            "episodeCount": result["episodeCount"],
            "configuration20": result["configuration20"],
            "offscreenWorldFrameRate20": result["offscreenWorldFrameRate20"],
        },
        indent=2,
    )
)
