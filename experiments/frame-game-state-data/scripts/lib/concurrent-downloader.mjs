import fs from "node:fs";
import fsp from "node:fs/promises";
import path from "node:path";
import { Readable } from "node:stream";
import { pipeline } from "node:stream/promises";
import { FileHasher } from "./file-hasher.mjs";

export class ConcurrentDownloader {
  constructor({ rootDir, concurrency = 4, force = false, fetchImpl = fetch }) {
    this.rootDir = rootDir;
    this.concurrency = Math.max(1, Number(concurrency) || 1);
    this.force = force;
    this.fetchImpl = fetchImpl;
  }

  async downloadAll(jobs) {
    const queue = [...jobs];
    const results = [];
    const workerCount = Math.min(this.concurrency, queue.length);

    const workers = Array.from({ length: workerCount }, async () => {
      while (queue.length > 0) {
        const job = queue.shift();
        try {
          results.push(await this.downloadOne(job));
        } catch (error) {
          results.push({
            id: job.id,
            url: job.url,
            outputPath: job.outputPath(this.rootDir),
            status: "failed",
            error: error instanceof Error ? error.message : String(error),
          });
        }
      }
    });

    await Promise.all(workers);
    return results.sort((a, b) => a.id.localeCompare(b.id));
  }

  async downloadOne(job) {
    const outputPath = job.outputPath(this.rootDir);
    await fsp.mkdir(path.dirname(outputPath), { recursive: true });

    if (!this.force && fs.existsSync(outputPath)) {
      const sha256 = await FileHasher.sha256(outputPath);
      if (!job.expectedSha256 || job.expectedSha256 === sha256) {
        const stat = await fsp.stat(outputPath);
        return {
          id: job.id,
          url: job.url,
          outputPath,
          status: "skipped",
          bytes: stat.size,
          sha256,
        };
      }
    }

    const tempPath = `${outputPath}.part`;
    await fsp.rm(tempPath, { force: true });

    const response = await this.fetchImpl(job.url, {
      headers: { "user-agent": "smash-frame-data-experiment" },
    });

    if (!response.ok || !response.body) {
      throw new Error(`HTTP ${response.status} ${response.statusText} for ${job.url}`);
    }

    await pipeline(
      Readable.fromWeb(response.body),
      fs.createWriteStream(tempPath, { flags: "wx" }),
    );

    const sha256 = await FileHasher.sha256(tempPath);
    if (job.expectedSha256 && job.expectedSha256 !== sha256) {
      await fsp.rm(tempPath, { force: true });
      throw new Error(
        `SHA-256 mismatch for ${job.id}: expected ${job.expectedSha256}, got ${sha256}`,
      );
    }

    await fsp.rename(tempPath, outputPath);
    const stat = await fsp.stat(outputPath);
    return {
      id: job.id,
      url: job.url,
      outputPath,
      status: "downloaded",
      bytes: stat.size,
      sha256,
    };
  }
}
