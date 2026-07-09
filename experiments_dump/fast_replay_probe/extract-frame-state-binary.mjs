import fs from "node:fs";
import path from "node:path";
import { performance } from "node:perf_hooks";
import { fileURLToPath } from "node:url";
import { SlippiGame } from "@slippi/slippi-js/node";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const rootDir = path.resolve(__dirname, "..");

const inputPath = path.resolve(
  rootDir,
  process.argv[2] ?? "replays/realtimeTest.slp",
);
const outputBase = path.resolve(
  __dirname,
  process.argv[3] ?? "runs/frame-state/realtimeTest",
);

const RECORD_BYTES = 132;
const MAGIC = "SLPFRAMESTATEv2";

function number(value, fallback = 0) {
  return Number.isFinite(value) ? value : fallback;
}

function booleanByte(value) {
  return value ? 1 : 0;
}

function writeUInt32(buffer, offset, value) {
  buffer.writeUInt32LE((number(value) >>> 0), offset);
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
        rows.push({ frameNumber, slot: playerFrame.pre.playerIndex ?? 0, isFollower: false, source: playerFrame });
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

const started = performance.now();
const game = new SlippiGame(inputPath);
const settings = game.getSettings();
const metadata = game.getMetadata();
const frames = game.getFrames();
const parsedMs = performance.now();

const { frameNumbers, rows, stats } = collectRows(frames);
const rowBuffer = encodeRows(rows);
const encodedMs = performance.now();

fs.mkdirSync(path.dirname(outputBase), { recursive: true });
const binaryPath = `${outputBase}.frame-state.bin`;
const manifestPath = `${outputBase}.manifest.json`;

fs.writeFileSync(binaryPath, rowBuffer);
const wroteBinaryMs = performance.now();

const manifest = {
  format: MAGIC,
  inputPath,
  inputBytes: fs.statSync(inputPath).size,
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
  players: settings?.players?.map((player) => ({
    playerIndex: player.playerIndex,
    port: player.port,
    characterId: player.characterId,
    type: player.type,
  })) ?? [],
  timingsSeconds: {
    parse: Number(((parsedMs - started) / 1000).toFixed(6)),
    collectAndEncode: Number(((encodedMs - parsedMs) / 1000).toFixed(6)),
    binaryWrite: Number(((wroteBinaryMs - encodedMs) / 1000).toFixed(6)),
    totalBeforeManifest: Number(((wroteBinaryMs - started) / 1000).toFixed(6)),
  },
};
const doneMs = performance.now();
manifest.timingsSeconds.manifestWrite = Number(((doneMs - wroteBinaryMs) / 1000).toFixed(6));
manifest.timingsSeconds.total = Number(((doneMs - started) / 1000).toFixed(6));

fs.writeFileSync(manifestPath, `${JSON.stringify(manifest, null, 2)}\n`);
console.log(JSON.stringify(manifest, null, 2));
