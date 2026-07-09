import fs from "node:fs";
import path from "node:path";
import { spawnSync } from "node:child_process";
import { performance } from "node:perf_hooks";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const workspaceRoot = path.resolve(__dirname, "../..");
const extractorPath = path.join(__dirname, "extract-frame-state-binary.mjs");

function usage() {
  console.error(`Usage:
  node bench-frame-state-process.mjs --output-dir <dir> [--repeat <n>] [--continue-on-error] <slp-or-dir>...

Spawns one fresh Node process per SLP extraction and writes process-bench-summary.json.
`);
}

function parseArgs(argv) {
  const args = {
    inputs: [],
    outputDir: path.resolve(__dirname, "runs/frame-state-process-bench"),
    repeat: 1,
    continueOnError: false,
  };

  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === "--output-dir") {
      args.outputDir = path.resolve(argv[++index]);
    } else if (arg === "--repeat") {
      args.repeat = Number(argv[++index]);
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

function safeOutputBase(inputPath, fileIndex, iteration, outputDir) {
  const basename = path.basename(inputPath, ".slp").replaceAll(/[^A-Za-z0-9_.-]/g, "_");
  return path.join(outputDir, `${String(fileIndex).padStart(5, "0")}-${basename}-r${iteration}`);
}

function parseManifest(stdout, inputPath) {
  try {
    return JSON.parse(stdout);
  } catch (error) {
    throw new Error(`Could not parse extractor JSON for ${inputPath}: ${error.message}`);
  }
}

function runOne(inputPath, outputDir, iteration, fileIndex) {
  const outputBase = safeOutputBase(inputPath, fileIndex, iteration, outputDir);
  const started = performance.now();
  const result = spawnSync(process.execPath, [extractorPath, inputPath, outputBase], {
    cwd: __dirname,
    encoding: "utf8",
    maxBuffer: 128 * 1024 * 1024,
  });
  const done = performance.now();
  const processWallSeconds = Number(((done - started) / 1000).toFixed(6));

  if (result.status !== 0) {
    const stderr = result.stderr.trim();
    const stdout = result.stdout.trim();
    throw new Error(
      `Extractor exited ${result.status} for ${inputPath}\n${stderr || stdout}`,
    );
  }

  const manifest = parseManifest(result.stdout, inputPath);
  return {
    inputPath,
    iteration,
    processWallSeconds,
    startupAndSpawnOverheadSeconds: Number(
      Math.max(0, processWallSeconds - (manifest.timingsSeconds?.total ?? 0)).toFixed(6),
    ),
    manifest,
  };
}

const args = parseArgs(process.argv.slice(2));
const inputs = args.inputs.flatMap((input) => slpFiles(input)).sort();

if (inputs.length === 0) {
  throw new Error("No .slp files found");
}

fs.mkdirSync(args.outputDir, { recursive: true });

const benchStarted = performance.now();
const runs = [];
const failures = [];

for (let iteration = 0; iteration < args.repeat; iteration += 1) {
  for (const [fileIndex, inputPath] of inputs.entries()) {
    try {
      runs.push(runOne(inputPath, args.outputDir, iteration, fileIndex));
    } catch (error) {
      failures.push({
        inputPath,
        iteration,
        errorName: error?.name ?? "Error",
        errorMessage: error?.message ?? String(error),
      });
      if (!args.continueOnError) {
        throw error;
      }
    }
  }
}

const benchDone = performance.now();
const wallTimes = runs.map((run) => run.processWallSeconds);
const childTotals = runs.map((run) => run.manifest.timingsSeconds.total);
const overheads = runs.map((run) => run.startupAndSpawnOverheadSeconds);
const skippedPlayerFrames = runs.reduce(
  (sum, run) => sum + (run.manifest.frameRowStats?.skippedPlayerFrames ?? 0),
  0,
);
const skippedFollowerFrames = runs.reduce(
  (sum, run) => sum + (run.manifest.frameRowStats?.skippedFollowerFrames ?? 0),
  0,
);

const summary = {
  format: runs[0]?.manifest.format ?? null,
  recordBytes: runs[0]?.manifest.recordBytes ?? null,
  nodeExecutable: process.execPath,
  extractorPath,
  outputDir: args.outputDir,
  inputFiles: inputs.length,
  repeat: args.repeat,
  runs: runs.length,
  failures: failures.length,
  benchWallSeconds: Number(((benchDone - benchStarted) / 1000).toFixed(6)),
  runsOverOneSecond: wallTimes.filter((seconds) => seconds > 1).length,
  rowCompleteness: {
    skippedPlayerFrames,
    skippedFollowerFrames,
    skippedFramesTotal: skippedPlayerFrames + skippedFollowerFrames,
  },
  processWallSeconds: {
    min: percentile(wallTimes, 0),
    median: percentile(wallTimes, 0.5),
    p95: percentile(wallTimes, 0.95),
    max: percentile(wallTimes, 1),
  },
  childTotalSeconds: {
    min: percentile(childTotals, 0),
    median: percentile(childTotals, 0.5),
    p95: percentile(childTotals, 0.95),
    max: percentile(childTotals, 1),
  },
  startupAndSpawnOverheadSeconds: {
    min: percentile(overheads, 0),
    median: percentile(overheads, 0.5),
    p95: percentile(overheads, 0.95),
    max: percentile(overheads, 1),
  },
  files: runs.map((run) => ({
    inputPath: run.inputPath,
    iteration: run.iteration,
    format: run.manifest.format,
    recordBytes: run.manifest.recordBytes,
    processWallSeconds: run.processWallSeconds,
    startupAndSpawnOverheadSeconds: run.startupAndSpawnOverheadSeconds,
    inputBytes: run.manifest.inputBytes,
    frameEntries: run.manifest.frameEntries,
    rowCount: run.manifest.rowCount,
    binaryBytes: run.manifest.binaryBytes,
    frameRange: run.manifest.frameRange,
    frameRowStats: run.manifest.frameRowStats,
    timingsSeconds: run.manifest.timingsSeconds,
  })),
  failedFiles: failures,
};

const summaryPath = path.join(args.outputDir, "process-bench-summary.json");
fs.writeFileSync(summaryPath, `${JSON.stringify(summary, null, 2)}\n`);
console.log(JSON.stringify(summary, null, 2));
