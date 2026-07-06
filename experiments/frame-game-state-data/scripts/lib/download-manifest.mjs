import fs from "node:fs/promises";
import path from "node:path";

const safeId = (value) =>
  String(value)
    .trim()
    .replace(/[^A-Za-z0-9._-]+/g, "-")
    .replace(/^-+|-+$/g, "");

export class ReplayDownloadJob {
  constructor({ id, url, fileName, outputDir, expectedSha256 = null, source = null }) {
    this.id = safeId(id);
    this.url = url;
    this.fileNameValue = fileName;
    this.outputDir = outputDir;
    this.expectedSha256 = expectedSha256;
    this.source = source;
  }

  static fromManifestEntry(entry, defaults = {}) {
    if (!entry || typeof entry !== "object") {
      throw new Error("Replay manifest entry must be an object");
    }

    const url = entry.url;
    if (!url) {
      throw new Error("Replay manifest entry is missing url");
    }

    let parsedUrl;
    try {
      parsedUrl = new URL(url);
    } catch {
      throw new Error(`Invalid replay URL: ${url}`);
    }

    const inferredFileName = path.basename(parsedUrl.pathname) || "replay.slp";
    const fileName = entry.fileName ?? inferredFileName;
    const id = entry.id ?? path.basename(fileName, path.extname(fileName));

    return new ReplayDownloadJob({
      id,
      url,
      fileName,
      outputDir: entry.outputDir ?? defaults.outputDir ?? "replays/downloaded",
      expectedSha256: entry.sha256 ?? entry.expectedSha256 ?? null,
      source: entry.source ?? defaults.source ?? null,
    });
  }

  fileName() {
    return this.fileNameValue.endsWith(".slp")
      ? this.fileNameValue
      : `${this.fileNameValue}.slp`;
  }

  outputPath(rootDir) {
    return path.resolve(rootDir, this.outputDir, this.fileName());
  }

  toJSON(rootDir) {
    return {
      id: this.id,
      url: this.url,
      fileName: this.fileName(),
      outputPath: this.outputPath(rootDir),
      expectedSha256: this.expectedSha256,
      source: this.source,
    };
  }
}

export class DownloadManifest {
  constructor({ path: manifestPath, defaults = {}, jobs = [] }) {
    this.path = manifestPath;
    this.defaults = defaults;
    this.replayJobs = jobs;
  }

  static async load(manifestPath) {
    const raw = JSON.parse(await fs.readFile(manifestPath, "utf8"));
    const defaults = raw.defaults ?? {};
    const entries = raw.replays ?? raw.downloads ?? [];

    if (!Array.isArray(entries) || entries.length === 0) {
      throw new Error(`Manifest has no replays: ${manifestPath}`);
    }

    const jobs = entries.map((entry) =>
      ReplayDownloadJob.fromManifestEntry(entry, defaults),
    );

    const seen = new Set();
    for (const job of jobs) {
      if (seen.has(job.id)) {
        throw new Error(`Duplicate replay id in manifest: ${job.id}`);
      }
      seen.add(job.id);
    }

    return new DownloadManifest({ path: manifestPath, defaults, jobs });
  }

  jobs() {
    return [...this.replayJobs];
  }
}
