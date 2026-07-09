import fs from "node:fs";
import path from "node:path";

function usage() {
  console.error(`Usage:
  node verify-frame-state-summary.mjs [--max-seconds <n>] <summary.json>...

Verifies benchmark summaries for zero failures, zero occupied-frame skips, valid binary sizes,
and no per-file extraction over the configured threshold.
`);
}

function parseArgs(argv) {
  const args = {
    summaries: [],
    maxSeconds: 1,
  };

  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === "--max-seconds") {
      args.maxSeconds = Number(argv[++index]);
    } else if (arg === "--help" || arg === "-h") {
      usage();
      process.exit(0);
    } else {
      args.summaries.push(path.resolve(arg));
    }
  }

  if (!Number.isFinite(args.maxSeconds) || args.maxSeconds <= 0) {
    throw new Error("--max-seconds must be a positive number");
  }
  if (args.summaries.length === 0) {
    throw new Error("At least one summary path is required");
  }

  return args;
}

function recordBytesForFormat(format) {
  if (format === "SLPFRAMESTATEv2") {
    return 132;
  }
  if (format === "SLPFRAMESTATEv1") {
    return 128;
  }
  return null;
}

function extractionSeconds(file) {
  if (Number.isFinite(file.processWallSeconds)) {
    return file.processWallSeconds;
  }
  if (Number.isFinite(file.timingsSeconds?.total)) {
    return file.timingsSeconds.total;
  }
  return null;
}

function verifySummary(summaryPath, maxSeconds) {
  const summary = JSON.parse(fs.readFileSync(summaryPath, "utf8"));
  const errors = [];
  const recordBytes = summary.recordBytes ?? recordBytesForFormat(summary.format);
  const files = summary.files ?? [];

  if (summary.failures !== 0) {
    errors.push(`expected zero failures, got ${summary.failures}`);
  }
  if ((summary.failedFiles?.length ?? 0) !== 0) {
    errors.push(`expected no failedFiles entries, got ${summary.failedFiles.length}`);
  }
  if (summary.runs !== files.length) {
    errors.push(`summary.runs ${summary.runs} does not match files length ${files.length}`);
  }
  if (summary.runsOverOneSecond !== undefined && summary.runsOverOneSecond !== 0) {
    errors.push(`expected zero runsOverOneSecond, got ${summary.runsOverOneSecond}`);
  }
  if ((summary.rowCompleteness?.skippedFramesTotal ?? 0) !== 0) {
    errors.push(
      `expected zero skipped occupied frames, got ${summary.rowCompleteness.skippedFramesTotal}`,
    );
  }
  if (!recordBytes) {
    errors.push(`unknown record size for format ${summary.format}`);
  }

  let maxObservedSeconds = 0;
  for (const [index, file] of files.entries()) {
    const label = file.inputPath ?? `files[${index}]`;
    const fileRecordBytes = file.recordBytes ?? recordBytes;
    const seconds = extractionSeconds(file);
    if (!Number.isFinite(seconds)) {
      errors.push(`${label}: no extraction timing`);
    } else {
      maxObservedSeconds = Math.max(maxObservedSeconds, seconds);
      if (seconds > maxSeconds) {
        errors.push(`${label}: ${seconds}s exceeds ${maxSeconds}s`);
      }
    }
    if (file.rowCount <= 0) {
      errors.push(`${label}: rowCount must be positive`);
    }
    if (file.frameEntries <= 0) {
      errors.push(`${label}: frameEntries must be positive`);
    }
    if (fileRecordBytes && file.binaryBytes !== file.rowCount * fileRecordBytes) {
      errors.push(
        `${label}: binaryBytes ${file.binaryBytes} != rowCount ${file.rowCount} * ${fileRecordBytes}`,
      );
    }
    const stats = file.frameRowStats;
    if (stats) {
      const skipped = (stats.skippedPlayerFrames ?? 0) + (stats.skippedFollowerFrames ?? 0);
      const completeRows = (stats.completePlayerRows ?? 0) + (stats.completeFollowerRows ?? 0);
      if (skipped !== 0) {
        errors.push(`${label}: skipped ${skipped} occupied frame records`);
      }
      if (completeRows !== file.rowCount) {
        errors.push(`${label}: complete row stats ${completeRows} != rowCount ${file.rowCount}`);
      }
    }
    if (file.frameRange?.metadataLastFrame != null && file.frameRange.last !== file.frameRange.metadataLastFrame) {
      errors.push(
        `${label}: last frame ${file.frameRange.last} != metadataLastFrame ${file.frameRange.metadataLastFrame}`,
      );
    }
  }

  return {
    summaryPath,
    format: summary.format,
    runs: summary.runs,
    failures: summary.failures,
    maxSeconds,
    maxObservedSeconds: Number(maxObservedSeconds.toFixed(6)),
    errors,
  };
}

const args = parseArgs(process.argv.slice(2));
const reports = args.summaries.map((summaryPath) => verifySummary(summaryPath, args.maxSeconds));
const failedReports = reports.filter((report) => report.errors.length > 0);

console.log(JSON.stringify({ reports }, null, 2));

if (failedReports.length > 0) {
  process.exitCode = 1;
}
