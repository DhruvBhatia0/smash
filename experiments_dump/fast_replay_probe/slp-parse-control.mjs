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
const outputPath = path.resolve(
  __dirname,
  process.argv[3] ?? "runs/slp-parse-control.json",
);

const started = performance.now();
const game = new SlippiGame(inputPath);
const settings = game.getSettings();
const metadata = game.getMetadata();
const frames = game.getFrames();
let playerFrameRows = 0;
let checksum = 0;

for (const [frameText, frameEntry] of Object.entries(frames)) {
  const frameNumber = Number(frameText);
  for (const playerFrame of frameEntry.players ?? []) {
    if (!playerFrame?.pre || !playerFrame?.post) continue;
    playerFrameRows += 1;
    checksum +=
      frameNumber +
      (playerFrame.post.actionStateId ?? 0) +
      Math.trunc((playerFrame.post.positionX ?? 0) * 1000);
  }
}

const elapsedMs = performance.now() - started;
const summary = {
  inputPath,
  bytes: fs.statSync(inputPath).size,
  elapsedSeconds: Number((elapsedMs / 1000).toFixed(6)),
  slpVersion: settings?.slpVersion ?? null,
  firstFrame: Math.min(...Object.keys(frames).map(Number)),
  lastFrame: metadata?.lastFrame ?? null,
  frameEntries: Object.keys(frames).length,
  playerFrameRows,
  checksum,
};

fs.mkdirSync(path.dirname(outputPath), { recursive: true });
fs.writeFileSync(outputPath, `${JSON.stringify(summary, null, 2)}\n`);
console.log(JSON.stringify(summary, null, 2));
