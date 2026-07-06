import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const rootDir = path.resolve(__dirname, "..");

const sourceDir = path.resolve(
  rootDir,
  process.argv[2] ?? "frames/replay-playback.capture-test",
);

const manifestPath = path.join(sourceDir, "manifest.json");
if (!fs.existsSync(manifestPath)) {
  throw new Error(`Missing render manifest: ${manifestPath}`);
}

const renderManifest = JSON.parse(fs.readFileSync(manifestPath, "utf8"));
const commandId =
  renderManifest.replayConfig?.commandId ?? path.basename(sourceDir);
const outputDir = path.resolve(
  rootDir,
  process.argv[3] ?? path.join("game-frames", commandId),
);

const runLogPath = renderManifest.logs?.run
  ? path.resolve(renderManifest.logs.run)
  : path.join(sourceDir, "render-replay-debug.log");
if (!fs.existsSync(runLogPath)) {
  throw new Error(`Missing render log: ${runLogPath}`);
}

const currentFrames = fs
  .readFileSync(runLogPath, "utf8")
  .split(/\r?\n/)
  .map((line) => line.match(/\[CURRENT_FRAME\]\s+(-?\d+)/)?.[1])
  .filter((value) => value != null)
  .map(Number);

if (currentFrames.length === 0) {
  throw new Error(`No CURRENT_FRAME entries found in ${runLogPath}`);
}

const dumpFrames = fs
  .readdirSync(sourceDir)
  .map((name) => {
    const match = name.match(/^framedump_(\d+)\.png$/);
    return match
      ? {
          name,
          number: Number(match[1]),
          path: path.join(sourceDir, name),
        }
      : null;
  })
  .filter(Boolean)
  .sort((a, b) => a.number - b.number);

if (dumpFrames.length < currentFrames.length) {
  throw new Error(
    `Only found ${dumpFrames.length} PNGs, but render log has ${currentFrames.length} CURRENT_FRAME entries`,
  );
}

fs.mkdirSync(outputDir, { recursive: true });
for (const name of fs.readdirSync(outputDir)) {
  if (/^frame_-?\d+\.png$/.test(name) || name === "frames.jsonl") {
    fs.rmSync(path.join(outputDir, name));
  }
}

const frameFileName = (frame) =>
  `frame_${frame < 0 ? "-" : ""}${String(Math.abs(frame)).padStart(6, "0")}.png`;

const rows = currentFrames.map((replayFrame, index) => {
  const source = dumpFrames[index];
  const imageName = frameFileName(replayFrame);
  const imagePath = path.join(outputDir, imageName);

  try {
    fs.linkSync(source.path, imagePath);
  } catch {
    fs.copyFileSync(source.path, imagePath);
  }

  return {
    replayFrame,
    sourceDumpFrame: source.number,
    imagePath,
    relativeImagePath: path.relative(rootDir, imagePath),
  };
});

const alignedManifest = {
  commandId,
  generatedAt: new Date().toISOString(),
  sourceDir,
  outputDir,
  replayJson: renderManifest.replayJson,
  replayConfig: renderManifest.replayConfig,
  frameCount: rows.length,
  replayFrameRange: {
    first: rows[0]?.replayFrame ?? null,
    last: rows.at(-1)?.replayFrame ?? null,
  },
  sourceDumpFrameRange: {
    first: rows[0]?.sourceDumpFrame ?? null,
    last: rows.at(-1)?.sourceDumpFrame ?? null,
  },
  droppedTrailingDumpFrames: dumpFrames.length - rows.length,
  mapping: "CURRENT_FRAME log entry order mapped to sorted framedump_*.png order",
  outputs: {
    framesJsonl: path.join(outputDir, "frames.jsonl"),
    manifest: path.join(outputDir, "manifest.json"),
  },
};

fs.writeFileSync(
  path.join(outputDir, "frames.jsonl"),
  `${rows.map((row) => JSON.stringify(row)).join("\n")}\n`,
);
fs.writeFileSync(
  path.join(outputDir, "manifest.json"),
  `${JSON.stringify(alignedManifest, null, 2)}\n`,
);

console.log(
  JSON.stringify(
    {
      outputDir,
      frameCount: alignedManifest.frameCount,
      replayFrameRange: alignedManifest.replayFrameRange,
      sourceDumpFrameRange: alignedManifest.sourceDumpFrameRange,
      droppedTrailingDumpFrames: alignedManifest.droppedTrailingDumpFrames,
    },
    null,
    2,
  ),
);
