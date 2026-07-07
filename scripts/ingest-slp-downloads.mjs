import path from "node:path";
import { fileURLToPath } from "node:url";
import { ReplayIngestionPipeline } from "./lib/replay-ingestion-pipeline.mjs";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const rootDir = path.resolve(__dirname, "..");

class IngestSlpCli {
  constructor(argv) {
    this.argv = argv;
  }

  parse() {
    const args = {
      manifest: "download-manifests/slippi-js-samples.json",
      metadataDir: "metadata/replays",
      concurrency: 4,
      force: false,
    };

    for (let index = 0; index < this.argv.length; index += 1) {
      const arg = this.argv[index];
      const next = () => this.argv[++index];

      if (arg === "--") continue;
      else if (arg === "--manifest") args.manifest = next();
      else if (arg === "--metadata-dir") args.metadataDir = next();
      else if (arg === "--concurrency") args.concurrency = Number(next());
      else if (arg === "--force") args.force = true;
      else if (arg === "--help" || arg === "-h") args.help = true;
      else throw new Error(`Unknown argument: ${arg}`);
    }

    return args;
  }

  usage() {
    return [
      "Usage:",
      "  npm run ingest:slp -- [--manifest <path>] [--metadata-dir <path>] [--concurrency <n>] [--force]",
      "",
      "Defaults:",
      "  --manifest download-manifests/slippi-js-samples.json",
      "  --metadata-dir metadata/replays",
      "  --concurrency 4",
    ].join("\n");
  }

  async run() {
    const args = this.parse();
    if (args.help) {
      console.log(this.usage());
      return;
    }

    const pipeline = new ReplayIngestionPipeline({
      rootDir,
      manifestPath: path.resolve(rootDir, args.manifest),
      metadataDir: args.metadataDir,
      concurrency: args.concurrency,
      force: args.force,
    });

    const result = await pipeline.run();
    console.log(JSON.stringify(result, null, 2));
  }
}

new IngestSlpCli(process.argv.slice(2)).run().catch((error) => {
  console.error(error instanceof Error ? error.stack : error);
  process.exit(1);
});
