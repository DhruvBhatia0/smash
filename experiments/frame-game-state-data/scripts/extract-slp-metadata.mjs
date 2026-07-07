#!/usr/bin/env node
import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { SlpMetadataExtractor } from "./lib/slp-metadata-extractor.mjs";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const rootDir = path.resolve(__dirname, "..");

class ExtractSlpMetadataCli {
  constructor(argv) {
    this.argv = argv;
  }

  usage() {
    return [
      "Usage:",
      "  node scripts/extract-slp-metadata.mjs <input.slp> [output.json]",
    ].join("\n");
  }

  parse() {
    const [input, output] = this.argv;
    if (!input || input === "--help" || input === "-h") {
      return { help: true };
    }
    return {
      inputPath: path.resolve(rootDir, input),
      outputPath: output ? path.resolve(rootDir, output) : null,
    };
  }

  async run() {
    const args = this.parse();
    if (args.help) {
      console.log(this.usage());
      return;
    }

    const extractor = new SlpMetadataExtractor();
    const metadata = await extractor.extract(args.inputPath);
    const output = `${JSON.stringify(metadata, null, 2)}\n`;

    if (args.outputPath) {
      await fs.mkdir(path.dirname(args.outputPath), { recursive: true });
      await fs.writeFile(args.outputPath, output);
    } else {
      process.stdout.write(output);
    }
  }
}

new ExtractSlpMetadataCli(process.argv.slice(2)).run().catch((error) => {
  console.error(error instanceof Error ? error.stack : error);
  process.exit(1);
});
