import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const rootDir = path.resolve(__dirname, "..");

const stateRowsPath = path.resolve(
  rootDir,
  process.argv[2] ?? "data/realtimeTest.frames.jsonl",
);
const frameMapPath = path.resolve(
  rootDir,
  process.argv[3] ?? "game-frames/realtimeTest-capture-test/frames.jsonl",
);
const outputPath = path.resolve(
  rootDir,
  process.argv[4] ?? "data/realtimeTest.capture-test.image-rows.jsonl",
);
const manifestPath = outputPath.replace(/\.jsonl$/, ".manifest.json");

if (!fs.existsSync(stateRowsPath)) {
  throw new Error(`Missing state rows: ${stateRowsPath}`);
}
if (!fs.existsSync(frameMapPath)) {
  throw new Error(`Missing frame image map: ${frameMapPath}`);
}

const imageByFrame = new Map(
  fs
    .readFileSync(frameMapPath, "utf8")
    .trim()
    .split(/\r?\n/)
    .filter(Boolean)
    .map((line) => {
      const row = JSON.parse(line);
      return [row.replayFrame, row];
    }),
);

const outputRows = [];
let inputRows = 0;
for (const line of fs.readFileSync(stateRowsPath, "utf8").split(/\r?\n/)) {
  if (!line) continue;
  inputRows += 1;
  const row = JSON.parse(line);
  const image = imageByFrame.get(row.frame);
  if (!image) continue;

  outputRows.push({
    ...row,
    screenshotPath: image.relativeImagePath,
    screenshotAbsolutePath: image.imagePath,
    sourceDumpFrame: image.sourceDumpFrame,
  });
}

fs.mkdirSync(path.dirname(outputPath), { recursive: true });
fs.writeFileSync(
  outputPath,
  `${outputRows.map((row) => JSON.stringify(row)).join("\n")}\n`,
);

const manifest = {
  generatedAt: new Date().toISOString(),
  stateRowsPath,
  frameMapPath,
  outputPath,
  inputRows,
  outputRows: outputRows.length,
  imageFrames: imageByFrame.size,
  replayFrameRange: {
    first: Math.min(...imageByFrame.keys()),
    last: Math.max(...imageByFrame.keys()),
  },
};
fs.writeFileSync(manifestPath, `${JSON.stringify(manifest, null, 2)}\n`);

console.log(
  JSON.stringify(
    {
      outputPath,
      outputRows: manifest.outputRows,
      imageFrames: manifest.imageFrames,
      replayFrameRange: manifest.replayFrameRange,
    },
    null,
    2,
  ),
);
