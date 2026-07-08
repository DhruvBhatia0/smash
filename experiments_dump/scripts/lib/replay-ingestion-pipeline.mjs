import fs from "node:fs/promises";
import path from "node:path";
import { ConcurrentDownloader } from "./concurrent-downloader.mjs";
import { DownloadManifest } from "./download-manifest.mjs";
import { SlpMetadataExtractor } from "./slp-metadata-extractor.mjs";

export class ReplayIngestionPipeline {
  constructor({
    rootDir,
    manifestPath,
    metadataDir = "metadata/replays",
    concurrency = 4,
    force = false,
  }) {
    this.rootDir = rootDir;
    this.manifestPath = manifestPath;
    this.metadataDir = path.resolve(rootDir, metadataDir);
    this.concurrency = concurrency;
    this.force = force;
    this.extractor = new SlpMetadataExtractor();
  }

  async run() {
    const manifest = await DownloadManifest.load(this.manifestPath);
    const jobs = manifest.jobs();
    const downloader = new ConcurrentDownloader({
      rootDir: this.rootDir,
      concurrency: this.concurrency,
      force: this.force,
    });

    const downloadResults = await downloader.downloadAll(jobs);
    const metadataResults = await this.extractAll(jobs, downloadResults);
    await this.writeIndex({ manifest, downloadResults, metadataResults });

    return {
      manifestPath: this.manifestPath,
      metadataDir: this.metadataDir,
      downloads: this.summarize(downloadResults),
      metadata: this.summarize(metadataResults),
      indexPath: path.join(this.metadataDir, "index.jsonl"),
      manifestOutputPath: path.join(this.metadataDir, "manifest.json"),
    };
  }

  async extractAll(jobs, downloadResults) {
    await fs.mkdir(this.metadataDir, { recursive: true });

    const successfulDownloads = new Map(
      downloadResults
        .filter((result) => result.status === "downloaded" || result.status === "skipped")
        .map((result) => [result.id, result]),
    );

    const results = [];
    for (const job of jobs) {
      const downloadResult = successfulDownloads.get(job.id);
      if (!downloadResult) {
        results.push({
          id: job.id,
          status: "skipped",
          reason: "download-failed",
        });
        continue;
      }

      try {
        const metadata = await this.extractor.extract(downloadResult.outputPath);
        const outputPath = path.join(this.metadataDir, `${job.id}.metadata.json`);
        const enrichedMetadata = {
          ...metadata,
          download: {
            id: job.id,
            url: job.url,
            source: job.source,
            status: downloadResult.status,
          },
        };

        await fs.writeFile(outputPath, `${JSON.stringify(enrichedMetadata, null, 2)}\n`);
        results.push({
          id: job.id,
          status: "extracted",
          outputPath,
          replayPath: downloadResult.outputPath,
          stage: enrichedMetadata.game.stage,
          winner: enrichedMetadata.winner,
          playerCount: enrichedMetadata.players.length,
        });
      } catch (error) {
        results.push({
          id: job.id,
          status: "failed",
          replayPath: downloadResult.outputPath,
          error: error instanceof Error ? error.message : String(error),
        });
      }
    }

    return results;
  }

  async writeIndex({ manifest, downloadResults, metadataResults }) {
    await fs.mkdir(this.metadataDir, { recursive: true });
    const metadataById = new Map(metadataResults.map((result) => [result.id, result]));
    const rows = downloadResults.map((download) => ({
      id: download.id,
      download,
      metadata: metadataById.get(download.id) ?? null,
    }));

    await fs.writeFile(
      path.join(this.metadataDir, "index.jsonl"),
      `${rows.map((row) => JSON.stringify(row)).join("\n")}\n`,
    );

    await fs.writeFile(
      path.join(this.metadataDir, "manifest.json"),
      `${JSON.stringify(
        {
          generatedAt: new Date().toISOString(),
          sourceManifest: manifest.path,
          concurrency: this.concurrency,
          force: this.force,
          counts: {
            jobs: manifest.jobs().length,
            downloads: this.summarize(downloadResults),
            metadata: this.summarize(metadataResults),
          },
        },
        null,
        2,
      )}\n`,
    );
  }

  summarize(results) {
    return results.reduce((acc, result) => {
      acc[result.status] = (acc[result.status] ?? 0) + 1;
      return acc;
    }, {});
  }
}
