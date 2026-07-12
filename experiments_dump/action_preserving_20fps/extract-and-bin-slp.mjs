#!/usr/bin/env node

import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import process from "node:process";
import { SlippiGame } from "@slippi/slippi-js/node";

import { binTicks, canonicalJson } from "./binning.mjs";

const DEFAULTS = {
  sourceFps: 60,
  targetFps: 20,
};

function usage() {
  return `Usage:
  node extract-and-bin-slp.mjs \\
    --input ../replays/realtimeTest.slp \\
    --output-dir runs/realtime-playable \\
    --first-frame -39 --last-frame 2181 \\
    --video-first-slp-frame -122

Options:
  --input PATH                    Required .slp replay
  --output-dir PATH               Required output directory
  --first-frame INT               First post-frame state to retain
  --last-frame INT                Last post-frame state to retain
  --video-first-slp-frame INT     SLP frame represented by normalized video index 0
  --source-fps INT                Default: 60
  --target-fps INT                Default: 20
`;
}

function parseArgs(argv) {
  const options = { ...DEFAULTS };
  for (let index = 0; index < argv.length; index += 1) {
    const name = argv[index];
    if (name === "--help" || name === "-h") {
      process.stdout.write(usage());
      process.exit(0);
    }
    if (!name.startsWith("--") || index + 1 >= argv.length) {
      throw new Error(`Invalid argument: ${name}\n\n${usage()}`);
    }
    const value = argv[++index];
    const key = {
      "--input": "input",
      "--output-dir": "outputDir",
      "--first-frame": "firstFrame",
      "--last-frame": "lastFrame",
      "--video-first-slp-frame": "videoFirstSlpFrame",
      "--source-fps": "sourceFps",
      "--target-fps": "targetFps",
    }[name];
    if (!key) {
      throw new Error(`Unknown option: ${name}\n\n${usage()}`);
    }
    options[key] = ["input", "outputDir"].includes(key) ? value : Number(value);
  }

  for (const required of ["input", "outputDir", "firstFrame", "lastFrame"]) {
    if (options[required] === undefined) {
      throw new Error(`Missing --${required.replace(/[A-Z]/g, (c) => `-${c.toLowerCase()}`)}`);
    }
  }
  for (const integer of ["firstFrame", "lastFrame", "sourceFps", "targetFps"]) {
    if (!Number.isInteger(options[integer])) {
      throw new TypeError(`${integer} must be an integer`);
    }
  }
  if (options.videoFirstSlpFrame !== undefined && !Number.isInteger(options.videoFirstSlpFrame)) {
    throw new TypeError("videoFirstSlpFrame must be an integer");
  }
  if (options.firstFrame > options.lastFrame) {
    throw new RangeError("firstFrame must be <= lastFrame");
  }
  return options;
}

function finiteOrNull(value) {
  return typeof value === "number" && Number.isFinite(value) ? value : value ?? null;
}

function jsonSafe(value) {
  if (value === undefined || (typeof value === "number" && !Number.isFinite(value))) {
    return null;
  }
  if (value === null || ["boolean", "number", "string"].includes(typeof value)) {
    return value;
  }
  if (Array.isArray(value)) {
    return value.map(jsonSafe);
  }
  if (typeof value === "object") {
    return Object.fromEntries(
      Object.entries(value).map(([key, child]) => [key, jsonSafe(child)]),
    );
  }
  throw new TypeError(`Unsupported replay value: ${typeof value}`);
}

function uint32Mask(value, field, playerIndex, isFollower) {
  if (!Number.isSafeInteger(value)) {
    const role = isFollower ? "follower" : "leader";
    throw new TypeError(`${role} ${playerIndex} is missing integer ${field}`);
  }
  return Number(BigInt.asUintN(32, BigInt(value)));
}

function compactAction(pre, playerIndex, isFollower) {
  return {
    playerIndex,
    isFollower,
    joystickX: finiteOrNull(pre.joystickX),
    joystickY: finiteOrNull(pre.joystickY),
    rawJoystickX: finiteOrNull(pre.rawJoystickX),
    cStickX: finiteOrNull(pre.cStickX),
    cStickY: finiteOrNull(pre.cStickY),
    trigger: finiteOrNull(pre.trigger),
    physicalLTrigger: finiteOrNull(pre.physicalLTrigger),
    physicalRTrigger: finiteOrNull(pre.physicalRTrigger),
    buttons: uint32Mask(pre.buttons, "buttons", playerIndex, isFollower),
    physicalButtons: uint32Mask(pre.physicalButtons, "physicalButtons", playerIndex, isFollower),
  };
}

function frameActorEntries(frameEntry, activePlayerIndices) {
  const leaders = activePlayerIndices.map((playerIndex) => ({
    playerIndex,
    isFollower: false,
    frame: frameEntry.players?.[playerIndex],
  }));
  const followers = Object.entries(frameEntry.followers ?? {})
    .map(([playerIndex, frame]) => ({ playerIndex: Number(playerIndex), isFollower: true, frame }))
    .filter(({ frame }) => frame != null);
  return [...leaders, ...followers];
}

function requireCompleteActor({ sourceFrame, playerIndex, isFollower, frame }) {
  if (!frame?.pre || !frame?.post) {
    const role = isFollower ? "follower" : "leader";
    throw new Error(`Frame ${sourceFrame} is missing ${role} ${playerIndex} pre/post data`);
  }
}

function buildTick(frameEntry, sourceFrame, sourceIndex, activePlayerIndices, videoFirstSlpFrame) {
  if (!frameEntry) {
    throw new Error(`Replay is missing source frame ${sourceFrame}`);
  }
  const actors = frameActorEntries(frameEntry, activePlayerIndices);
  actors.forEach((actor) => requireCompleteActor({ sourceFrame, ...actor }));
  const videoIndex = videoFirstSlpFrame === undefined ? null : sourceFrame - videoFirstSlpFrame;
  return {
    sourceIndex,
    sourceFrame,
    videoIndex,
    state: jsonSafe({
      frame: sourceFrame,
      players: actors.map(({ playerIndex, isFollower, frame }) => ({
        playerIndex,
        isFollower,
        post: frame.post,
      })),
      items: frameEntry.items ?? [],
      stageEvents: frameEntry.stageEvents ?? [],
      frameStart: frameEntry.start ?? null,
    }),
    // Slippi's pre-frame input at f precedes its post-frame state at f. Therefore
    // the transition post(f-1) -> post(f) is conditioned on pre(f).
    actionFromPrevious: jsonSafe({
      sourceFrame,
      players: actors.map(({ playerIndex, isFollower, frame }) =>
        compactAction(frame.pre, playerIndex, isFollower),
      ),
    }),
  };
}

function popcount32(value) {
  let bits = Number(value) >>> 0;
  let count = 0;
  while (bits !== 0) {
    bits &= bits - 1;
    count += 1;
  }
  return count;
}

function playerMap(action) {
  return new Map(
    action.players.map((player) => [`${player.playerIndex}:${Number(player.isFollower)}`, player]),
  );
}

function risingEdgeCount(actions, field) {
  let count = 0;
  for (let index = 1; index < actions.length; index += 1) {
    const previous = playerMap(actions[index - 1]);
    for (const player of actions[index].players) {
      const key = `${player.playerIndex}:${Number(player.isFollower)}`;
      const before = Number(previous.get(key)?.[field] ?? 0) >>> 0;
      const after = Number(player[field] ?? 0) >>> 0;
      count += popcount32(after & ~before);
    }
  }
  return count;
}

function risingEdgesOnSelectedFrames(actions, field, selectedSourceIndices) {
  const selected = new Set(selectedSourceIndices);
  let count = 0;
  for (let index = 1; index < actions.length; index += 1) {
    if (!selected.has(index)) {
      continue;
    }
    const previous = playerMap(actions[index - 1]);
    for (const player of actions[index].players) {
      const key = `${player.playerIndex}:${Number(player.isFollower)}`;
      const before = Number(previous.get(key)?.[field] ?? 0) >>> 0;
      const after = Number(player[field] ?? 0) >>> 0;
      count += popcount32(after & ~before);
    }
  }
  return count;
}

function fullyHiddenPulseCount(actions, field, selectedSourceIndices) {
  const selected = new Set(selectedSourceIndices);
  let count = 0;
  for (let index = 1; index < actions.length; index += 1) {
    const previous = playerMap(actions[index - 1]);
    for (const player of actions[index].players) {
      const key = `${player.playerIndex}:${Number(player.isFollower)}`;
      const before = Number(previous.get(key)?.[field] ?? 0) >>> 0;
      const after = Number(player[field] ?? 0) >>> 0;
      const rising = (after & ~before) >>> 0;
      for (let bit = 0; bit < 32; bit += 1) {
        if (((rising >>> bit) & 1) === 0) {
          continue;
        }
        let visibleAtSample = false;
        for (let cursor = index; cursor < actions.length; cursor += 1) {
          const mask = Number(playerMap(actions[cursor]).get(key)?.[field] ?? 0) >>> 0;
          if (((mask >>> bit) & 1) === 0) {
            break;
          }
          if (selected.has(cursor)) {
            visibleAtSample = true;
          }
        }
        if (!visibleAtSample) {
          count += 1;
        }
      }
    }
  }
  return count;
}

function naiveSamplingMetrics(ticks, selectedSourceIndices) {
  const nativeActions = ticks.map((tick) => tick.actionFromPrevious);
  const sampledActions = selectedSourceIndices.map((index) => ticks[index].actionFromPrevious);
  const fields = ["buttons", "physicalButtons"];
  const nativeRisingEdges = Object.fromEntries(
    fields.map((field) => [field, risingEdgeCount(nativeActions, field)]),
  );
  const sampledRisingEdges = Object.fromEntries(
    fields.map((field) => [field, risingEdgeCount(sampledActions, field)]),
  );
  const risingEdgeCountReduction = Object.fromEntries(
    fields.map((field) => [field, nativeRisingEdges[field] - sampledRisingEdges[field]]),
  );
  const exactFrameRisingEdgesRetained = Object.fromEntries(
    fields.map((field) => [
      field,
      risingEdgesOnSelectedFrames(nativeActions, field, selectedSourceIndices),
    ]),
  );
  const exactFrameRisingEdgesLost = Object.fromEntries(
    fields.map((field) => [
      field,
      nativeRisingEdges[field] - exactFrameRisingEdgesRetained[field],
    ]),
  );
  const fullyHiddenPulses = Object.fromEntries(
    fields.map((field) => [
      field,
      fullyHiddenPulseCount(nativeActions, field, selectedSourceIndices),
    ]),
  );
  return {
    definition: "Physical-button metrics compare native 60 Hz edges with every-third-frame samples.",
    nativeRisingEdges,
    sampledRisingEdges,
    risingEdgeCountReduction,
    exactFrameRisingEdgesRetained,
    exactFrameRisingEdgesLost,
    fullyHiddenPulses,
  };
}

function sha256File(filePath) {
  return crypto.createHash("sha256").update(fs.readFileSync(filePath)).digest("hex");
}

function writeJsonl(filePath, rows) {
  fs.writeFileSync(filePath, `${rows.map((row) => canonicalJson(row)).join("\n")}\n`);
}

const options = parseArgs(process.argv.slice(2));
const inputPath = path.resolve(options.input);
const outputDir = path.resolve(options.outputDir);
const game = new SlippiGame(inputPath);
const settings = game.getSettings();
const frames = game.getFrames();
if (!settings) {
  throw new Error(`Could not read settings from ${inputPath}`);
}
if (settings.isPAL) {
  throw new Error("v1 is NTSC-only; PAL 50 -> 20 FPS needs alternating 2/3-step bins");
}

const activePlayerIndices = (settings.players ?? []).map((player) => player.playerIndex);
if (activePlayerIndices.length === 0) {
  throw new Error("Replay has no active players");
}

const ticks = [];
for (let sourceFrame = options.firstFrame; sourceFrame <= options.lastFrame; sourceFrame += 1) {
  ticks.push(
    buildTick(
      frames[sourceFrame],
      sourceFrame,
      ticks.length,
      activePlayerIndices,
      options.videoFirstSlpFrame,
    ),
  );
}

const result = binTicks(ticks, options);
const replay = {
  path: path.relative(process.cwd(), inputPath),
  sha256: sha256File(inputPath),
  firstFrame: options.firstFrame,
  lastFrame: options.lastFrame,
  videoFirstSlpFrame: options.videoFirstSlpFrame ?? null,
  activePlayerIndices,
  isPAL: settings.isPAL ?? null,
  slpVersion: settings.slpVersion,
};
const manifest = {
  ...result.manifest,
  replay,
  actionAlignment: "post(f-1) -> pre-input(f) -> post(f)",
  naiveEveryThirdFrame: naiveSamplingMetrics(ticks, result.manifest.selectedSourceIndices),
};

fs.mkdirSync(outputDir, { recursive: true });
writeJsonl(path.join(outputDir, "observations.jsonl"), result.observations);
writeJsonl(path.join(outputDir, "transitions.jsonl"), result.transitions);
fs.writeFileSync(path.join(outputDir, "manifest.json"), `${JSON.stringify(manifest, null, 2)}\n`);

process.stdout.write(
  `${JSON.stringify({
    outputDir,
    sourceObservationCount: manifest.sourceObservationCount,
    outputObservationCount: manifest.outputObservationCount,
    outputTransitionCount: manifest.outputTransitionCount,
    partialTransitionCount: manifest.partialTransitionCount,
    actionRoundTripExact: manifest.actionRoundTripExact,
    naivePhysicalButtonEdges: {
      native: manifest.naiveEveryThirdFrame.nativeRisingEdges.physicalButtons,
      exactFrameLost: manifest.naiveEveryThirdFrame.exactFrameRisingEdgesLost.physicalButtons,
      fullyHidden: manifest.naiveEveryThirdFrame.fullyHiddenPulses.physicalButtons,
    },
  }, null, 2)}\n`,
);
