import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { performance } from "node:perf_hooks";
import { fileURLToPath } from "node:url";
import { SlippiGame } from "@slippi/slippi-js/node";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const workspaceRoot = path.resolve(__dirname, "../..");

const RECORD_BYTES = 132;
const FORMAT = "SLPFRAMESTATEv2";

function usage() {
  console.error(`Usage:
  node extract-frame-state-batch.mjs --output-dir <dir> [--repeat <n>] [--dedupe] [--continue-on-error] <slp-or-dir>...

Writes one fixed-width player-frame binary per SLP plus a combined summary.
Directories are scanned recursively for .slp files.
`);
}

function parseArgs(argv) {
  const args = {
    inputs: [],
    outputDir: path.resolve(__dirname, "runs/frame-state-batch"),
    repeat: 1,
    dedupe: false,
    continueOnError: false,
  };

  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === "--output-dir") {
      args.outputDir = path.resolve(argv[++index]);
    } else if (arg === "--repeat") {
      args.repeat = Number(argv[++index]);
    } else if (arg === "--dedupe") {
      args.dedupe = true;
    } else if (arg === "--continue-on-error") {
      args.continueOnError = true;
    } else if (arg === "--help" || arg === "-h") {
      usage();
      process.exit(0);
    } else {
      args.inputs.push(path.resolve(arg));
    }
  }

  if (!Number.isInteger(args.repeat) || args.repeat < 1) {
    throw new Error("--repeat must be a positive integer");
  }

  if (args.inputs.length === 0) {
    args.inputs.push(path.resolve(workspaceRoot, "experiments_dump/replays/realtimeTest.slp"));
  }

  return args;
}

function slpFiles(inputPath) {
  const stat = fs.statSync(inputPath);
  if (stat.isFile()) {
    return inputPath.endsWith(".slp") ? [inputPath] : [];
  }

  const files = [];
  for (const entry of fs.readdirSync(inputPath, { withFileTypes: true })) {
    const child = path.join(inputPath, entry.name);
    if (entry.isDirectory()) {
      files.push(...slpFiles(child));
    } else if (entry.isFile() && child.endsWith(".slp")) {
      files.push(child);
    }
  }
  return files;
}

function number(value, fallback = 0) {
  return Number.isFinite(value) ? value : fallback;
}

function booleanByte(value) {
  return value ? 1 : 0;
}

function writeUInt32(buffer, offset, value) {
  buffer.writeUInt32LE(number(value) >>> 0, offset);
}

function writeInt32(buffer, offset, value) {
  buffer.writeInt32LE(number(value), offset);
}

function writeFloat(buffer, offset, value) {
  buffer.writeFloatLE(number(value), offset);
}

function collectRows(frames) {
  const frameNumbers = Object.keys(frames).map(Number).sort((a, b) => a - b);
  const rows = [];
  const stats = {
    frameEntries: frameNumbers.length,
    emptyPlayerSlots: 0,
    playerFrameEntries: 0,
    completePlayerRows: 0,
    skippedPlayerFrames: 0,
    followerFrameEntries: 0,
    completeFollowerRows: 0,
    skippedFollowerFrames: 0,
  };

  for (const frameNumber of frameNumbers) {
    const frameEntry = frames[frameNumber];
    for (const playerFrame of frameEntry.players ?? []) {
      if (!playerFrame) {
        stats.emptyPlayerSlots += 1;
        continue;
      }
      stats.playerFrameEntries += 1;
      if (playerFrame?.pre && playerFrame?.post) {
        stats.completePlayerRows += 1;
        rows.push({
          frameNumber,
          slot: playerFrame.pre.playerIndex ?? 0,
          isFollower: false,
          source: playerFrame,
        });
      } else {
        stats.skippedPlayerFrames += 1;
      }
      if (playerFrame?.follower) {
        stats.followerFrameEntries += 1;
        if (playerFrame.follower.pre && playerFrame.follower.post) {
          stats.completeFollowerRows += 1;
          rows.push({
            frameNumber,
            slot: playerFrame.follower.pre.playerIndex ?? playerFrame.pre?.playerIndex ?? 0,
            isFollower: true,
            source: playerFrame.follower,
          });
        } else {
          stats.skippedFollowerFrames += 1;
        }
      }
    }
  }

  return { frameNumbers, rows, stats };
}

function encodeRows(rows) {
  const output = Buffer.allocUnsafe(RECORD_BYTES * rows.length);

  rows.forEach((row, index) => {
    const offset = index * RECORD_BYTES;
    const pre = row.source.pre ?? {};
    const post = row.source.post ?? {};
    const speeds = post.selfInducedSpeeds ?? {};

    writeInt32(output, offset + 0, row.frameNumber);
    output.writeUInt8(number(row.slot), offset + 4);
    output.writeUInt8(booleanByte(row.isFollower), offset + 5);
    output.writeUInt8(booleanByte(post.isAirborne), offset + 6);
    output.writeUInt8(number(post.internalCharacterId), offset + 7);
    writeUInt32(output, offset + 8, pre.seed);
    output.writeUInt16LE(number(pre.actionStateId), offset + 12);
    output.writeUInt16LE(number(post.actionStateId), offset + 14);

    writeFloat(output, offset + 16, pre.positionX);
    writeFloat(output, offset + 20, pre.positionY);
    writeFloat(output, offset + 24, pre.facingDirection);
    writeFloat(output, offset + 28, pre.joystickX);
    writeFloat(output, offset + 32, pre.joystickY);
    writeFloat(output, offset + 36, pre.cStickX);
    writeFloat(output, offset + 40, pre.cStickY);
    writeFloat(output, offset + 44, pre.trigger);
    writeUInt32(output, offset + 48, pre.buttons);
    writeUInt32(output, offset + 52, pre.physicalButtons);
    writeFloat(output, offset + 56, pre.physicalLTrigger);
    writeFloat(output, offset + 60, pre.physicalRTrigger);
    writeFloat(output, offset + 64, pre.rawJoystickX);
    writeFloat(output, offset + 68, pre.percent);

    writeFloat(output, offset + 72, post.positionX);
    writeFloat(output, offset + 76, post.positionY);
    writeFloat(output, offset + 80, post.facingDirection);
    writeFloat(output, offset + 84, post.percent);
    writeFloat(output, offset + 88, post.shieldSize);
    output.writeUInt8(number(post.lastAttackLanded), offset + 92);
    output.writeUInt8(number(post.currentComboCount), offset + 93);
    output.writeUInt8(number(post.lastHitBy), offset + 94);
    output.writeUInt8(number(post.stocksRemaining), offset + 95);
    writeFloat(output, offset + 96, post.actionStateCounter);
    writeFloat(output, offset + 100, post.miscActionState);
    output.writeUInt16LE(number(post.lastGroundId), offset + 104);
    output.writeUInt8(number(post.jumpsRemaining), offset + 106);
    output.writeUInt8(number(post.lCancelStatus), offset + 107);
    output.writeUInt8(number(post.hurtboxCollisionState), offset + 108);
    output.writeUInt8(0, offset + 109);
    output.writeUInt16LE(number(post.instanceHitBy), offset + 110);
    output.writeUInt16LE(number(post.instanceId), offset + 112);
    writeFloat(output, offset + 116, post.hitlagRemaining);
    writeUInt32(output, offset + 120, post.animationIndex);
    writeFloat(output, offset + 124, speeds.x);
    writeFloat(output, offset + 128, speeds.y);
  });

  return output;
}

function digestFile(inputPath) {
  const hash = crypto.createHash("sha1");
  hash.update(fs.readFileSync(inputPath));
  return hash.digest("hex");
}

function percentile(values, ratio) {
  if (values.length === 0) {
    return null;
  }
  const sorted = [...values].sort((a, b) => a - b);
  if (ratio <= 0) {
    return sorted[0];
  }
  const index = Math.min(sorted.length - 1, Math.ceil(sorted.length * ratio) - 1);
  return sorted[index];
}

function processSlp(inputPath, outputDir, iteration, fileIndex) {
  const started = performance.now();
  const inputBytes = fs.statSync(inputPath).size;
  const sha1 = digestFile(inputPath);
  const parsedHashMs = performance.now();

  const game = new SlippiGame(inputPath);
  const settings = game.getSettings();
  const metadata = game.getMetadata();
  const frames = game.getFrames();
  const parsedMs = performance.now();

  const { frameNumbers, rows, stats } = collectRows(frames);
  const rowBuffer = encodeRows(rows);
  const encodedMs = performance.now();

  const safeName = `${String(fileIndex).padStart(5, "0")}-${path.basename(inputPath, ".slp")}-${sha1.slice(0, 12)}-r${iteration}`;
  const binaryPath = path.join(outputDir, `${safeName}.frame-state.bin`);
  const manifestPath = path.join(outputDir, `${safeName}.manifest.json`);

  fs.writeFileSync(binaryPath, rowBuffer);
  const wroteBinaryMs = performance.now();

  const manifest = {
    format: FORMAT,
    inputPath,
    inputBytes,
    inputSha1: sha1,
    binaryPath,
    binaryBytes: rowBuffer.length,
    recordBytes: RECORD_BYTES,
    frameRange: {
      first: frameNumbers[0] ?? null,
      last: frameNumbers.at(-1) ?? null,
      metadataLastFrame: metadata?.lastFrame ?? null,
    },
    frameEntries: frameNumbers.length,
    rowCount: rows.length,
    frameRowStats: stats,
    players:
      settings?.players?.map((player) => ({
        playerIndex: player.playerIndex,
        port: player.port,
        characterId: player.characterId,
        type: player.type,
      })) ?? [],
    timingsSeconds: {
      sha1Read: Number(((parsedHashMs - started) / 1000).toFixed(6)),
      parse: Number(((parsedMs - parsedHashMs) / 1000).toFixed(6)),
      collectAndEncode: Number(((encodedMs - parsedMs) / 1000).toFixed(6)),
      binaryWrite: Number(((wroteBinaryMs - encodedMs) / 1000).toFixed(6)),
    },
  };
  manifest.timingsSeconds.totalBeforeManifest = Number(
    ((wroteBinaryMs - started) / 1000).toFixed(6),
  );

  const doneMs = performance.now();
  manifest.timingsSeconds.manifestWrite = Number(((doneMs - wroteBinaryMs) / 1000).toFixed(6));
  manifest.timingsSeconds.total = Number(((doneMs - started) / 1000).toFixed(6));
  fs.writeFileSync(manifestPath, `${JSON.stringify(manifest, null, 2)}\n`);

  return manifest;
}

const args = parseArgs(process.argv.slice(2));
const discovered = args.inputs.flatMap((input) => slpFiles(input)).sort();
const inputs = args.dedupe ? [...new Set(discovered.map((file) => fs.realpathSync(file)))] : discovered;

if (inputs.length === 0) {
  throw new Error("No .slp files found");
}

fs.mkdirSync(args.outputDir, { recursive: true });

const batchStarted = performance.now();
const manifests = [];
const failures = [];
for (let iteration = 0; iteration < args.repeat; iteration += 1) {
  for (const [fileIndex, inputPath] of inputs.entries()) {
    try {
      manifests.push(processSlp(inputPath, args.outputDir, iteration, fileIndex));
    } catch (error) {
      const failure = {
        inputPath,
        iteration,
        errorName: error?.name ?? "Error",
        errorMessage: error?.message ?? String(error),
      };
      failures.push(failure);
      if (!args.continueOnError) {
        throw error;
      }
    }
  }
}
const batchDone = performance.now();

const totals = manifests.map((manifest) => manifest.timingsSeconds.total);
const parseTotals = manifests.map((manifest) => manifest.timingsSeconds.parse);
const outputBytes = manifests.reduce((sum, manifest) => sum + manifest.binaryBytes, 0);
const skippedPlayerFrames = manifests.reduce(
  (sum, manifest) => sum + (manifest.frameRowStats?.skippedPlayerFrames ?? 0),
  0,
);
const skippedFollowerFrames = manifests.reduce(
  (sum, manifest) => sum + (manifest.frameRowStats?.skippedFollowerFrames ?? 0),
  0,
);
const summary = {
  format: FORMAT,
  recordBytes: RECORD_BYTES,
  outputDir: args.outputDir,
  inputFiles: inputs.length,
  repeat: args.repeat,
  runs: manifests.length,
  failures: failures.length,
  batchWallSeconds: Number(((batchDone - batchStarted) / 1000).toFixed(6)),
  totalOutputBytes: outputBytes,
  runsOverOneSecond: totals.filter((seconds) => seconds > 1).length,
  rowCompleteness: {
    skippedPlayerFrames,
    skippedFollowerFrames,
    skippedFramesTotal: skippedPlayerFrames + skippedFollowerFrames,
  },
  perFileSeconds: {
    min: percentile(totals, 0),
    median: percentile(totals, 0.5),
    p95: percentile(totals, 0.95),
    max: percentile(totals, 1),
  },
  parseSeconds: {
    min: percentile(parseTotals, 0),
    median: percentile(parseTotals, 0.5),
    p95: percentile(parseTotals, 0.95),
    max: percentile(parseTotals, 1),
  },
  files: manifests.map((manifest) => ({
    inputPath: manifest.inputPath,
    inputBytes: manifest.inputBytes,
    inputSha1: manifest.inputSha1,
    recordBytes: manifest.recordBytes,
    frameRange: manifest.frameRange,
    frameEntries: manifest.frameEntries,
    rowCount: manifest.rowCount,
    binaryBytes: manifest.binaryBytes,
    frameRowStats: manifest.frameRowStats,
    timingsSeconds: manifest.timingsSeconds,
  })),
  failedFiles: failures,
};

const summaryPath = path.join(args.outputDir, "batch-summary.json");
fs.writeFileSync(summaryPath, `${JSON.stringify(summary, null, 2)}\n`);
console.log(JSON.stringify(summary, null, 2));
