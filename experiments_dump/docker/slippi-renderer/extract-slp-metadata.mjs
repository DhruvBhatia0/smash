#!/usr/bin/env node
import crypto from "node:crypto";
import fs from "node:fs";
import fsp from "node:fs/promises";
import path from "node:path";
import {
  GameEndMethod,
  SlippiGame,
  characters,
  frameToGameTimer,
  stages,
} from "@slippi/slippi-js/node";

const [inputPath, outputPath, sourceReference = inputPath, provider = "unknown"] =
  process.argv.slice(2);

if (!inputPath || !outputPath) {
  console.error(
    "usage: extract-slp-metadata.mjs <input.slp> <output.json> [source-reference] [provider]",
  );
  process.exit(2);
}

const finite = (value) =>
  typeof value === "number" && Number.isFinite(value) ? value : null;
const valueOrNull = (value) => value ?? null;
const ratio = (value) => ({
  count: finite(value?.count),
  total: finite(value?.total),
  ratio: finite(value?.ratio),
});
const safeName = (lookup, ...args) => {
  try {
    return lookup(...args);
  } catch {
    return null;
  }
};

function sha256(filePath) {
  return new Promise((resolve, reject) => {
    const hash = crypto.createHash("sha256");
    const stream = fs.createReadStream(filePath);
    stream.on("error", reject);
    stream.on("data", (chunk) => hash.update(chunk));
    stream.on("end", () => resolve(hash.digest("hex")));
  });
}

function sourceName(reference) {
  const member = reference.includes("::") ? reference.split("::").at(-1) : reference;
  return path.basename(member);
}

function filenameRanks(filename) {
  const ranks = "grandmaster|master|diamond|platinum|gold|silver|bronze";
  const match = filename.match(new RegExp(`__(?:[^_]+-)?(${ranks})-(${ranks})-`, "i"));
  if (!match) return [];
  return [match[1].toLowerCase(), match[2].toLowerCase()];
}

function endMethodName(method) {
  return method == null ? null : GameEndMethod[method] ?? "UNKNOWN";
}

function formatDuration(seconds) {
  if (seconds == null) return null;
  const minutes = Math.floor(seconds / 60);
  return `${minutes}:${(seconds - minutes * 60).toFixed(3).padStart(6, "0")}`;
}

function overallStats(overall) {
  if (!overall) return null;
  return {
    inputCounts: overall.inputCounts ?? null,
    inputsPerMinute: ratio(overall.inputsPerMinute),
    digitalInputsPerMinute: ratio(overall.digitalInputsPerMinute),
    conversionCount: finite(overall.conversionCount),
    successfulConversions: ratio(overall.successfulConversions),
    killCount: finite(overall.killCount),
    totalDamage: finite(overall.totalDamage),
    openingsPerKill: ratio(overall.openingsPerKill),
    damagePerOpening: ratio(overall.damagePerOpening),
    neutralWinRatio: ratio(overall.neutralWinRatio),
    counterHitRatio: ratio(overall.counterHitRatio),
    beneficialTradeRatio: ratio(overall.beneficialTradeRatio),
  };
}

function winnerFrom(gameEnd, finalStates) {
  const placements = (gameEnd?.placements ?? [])
    .filter((entry) => entry.position >= 0 && finalStates.has(entry.playerIndex))
    .map((entry) => ({
      playerIndex: entry.playerIndex,
      place: entry.position + 1,
    }))
    .sort((a, b) => a.place - b.place);
  const method = gameEnd?.gameEndMethod;
  const resolved = method !== GameEndMethod.NO_CONTEST && method !== GameEndMethod.UNRESOLVED;
  return {
    playerIndex: resolved ? placements[0]?.playerIndex ?? null : null,
    source: resolved && placements.length ? "game-end-placement" : "unresolved",
    placements,
  };
}

async function main() {
  const game = new SlippiGame(inputPath);
  const settings = game.getSettings();
  if (!settings) throw new Error(`could not read game settings from ${inputPath}`);

  const metadata = game.getMetadata() ?? {};
  const stats = game.getStats() ?? {};
  const gameEnd = game.getGameEnd() ?? null;
  const frames = game.getFrames();
  const frameNumbers = Object.keys(frames).map(Number).filter(Number.isFinite).sort((a, b) => a - b);
  const firstFrame = frameNumbers[0] ?? null;
  const lastFrame = finite(stats.lastFrame) ?? finite(metadata.lastFrame) ?? frameNumbers.at(-1) ?? null;
  const firstPlayableFrame = -39;
  const playableFrameCount = finite(stats.playableFrameCount) ??
    (lastFrame == null ? null : Math.max(0, lastFrame - firstPlayableFrame));
  const durationSeconds = playableFrameCount == null ? null : playableFrameCount / 60;
  const lastFrameEntry = lastFrame == null ? null : frames[lastFrame];
  const ranks = filenameRanks(sourceName(sourceReference));
  const overallByPlayer = new Map((stats.overall ?? []).map((row) => [row.playerIndex, row]));
  const actionsByPlayer = new Map((stats.actionCounts ?? []).map((row) => [row.playerIndex, row]));
  const finalStates = new Map();

  const players = (settings.players ?? []).map((player, order) => {
    const post = lastFrameEntry?.players?.[player.playerIndex]?.post ?? null;
    const finalState = {
      frame: lastFrame,
      stocksRemaining: finite(post?.stocksRemaining),
      percent: finite(post?.percent),
      actionStateId: finite(post?.actionStateId),
      positionX: finite(post?.positionX),
      positionY: finite(post?.positionY),
    };
    finalStates.set(player.playerIndex, finalState);
    const characterId = finite(player.characterId);
    const colorId = finite(player.characterColor);
    const metadataPlayer = metadata.players?.[player.playerIndex] ?? null;
    return {
      playerIndex: player.playerIndex,
      port: finite(player.port),
      type: finite(player.type),
      isHuman: player.type === 0,
      character: {
        id: characterId,
        name: characterId == null ? null : safeName(characters.getCharacterName, characterId),
        shortName: characterId == null ? null : safeName(characters.getCharacterShortName, characterId),
        colorId,
        colorName:
          characterId == null || colorId == null
            ? null
            : safeName(characters.getCharacterColorName, characterId, colorId),
      },
      rank: {
        label: ranks[order] ?? null,
        source: ranks[order] ? "source-filename" : null,
        mmr: null,
      },
      identity: {
        nametag: player.nametag || null,
        displayName: player.displayName || metadataPlayer?.names?.netplay || null,
        connectCode: player.connectCode || metadataPlayer?.names?.code || null,
        userId: player.userId || null,
      },
      start: {
        stocks: finite(player.startStocks),
        handicap: finite(player.handicap),
        teamId: finite(player.teamId),
        teamShade: finite(player.teamShade),
        cpuLevel: finite(player.cpuLevel),
        controllerFix: valueOrNull(player.controllerFix),
        rumbleEnabled: valueOrNull(player.rumbleEnabled),
      },
      finalState,
      stats: overallStats(overallByPlayer.get(player.playerIndex)),
      actionCounts: actionsByPlayer.get(player.playerIndex) ?? null,
    };
  });

  const fileStat = await fsp.stat(inputPath);
  const startAt = metadata.startAt ?? null;
  const output = {
    schemaVersion: 1,
    generatedAt: new Date().toISOString(),
    source: {
      provider,
      reference: sourceReference,
      originalName: sourceName(sourceReference),
    },
    file: {
      name: "input.slp",
      bytes: fileStat.size,
      sha256: await sha256(inputPath),
    },
    match: {
      timestamp: {
        startAt,
        available: startAt != null,
        note: startAt == null ? "This replay does not contain a wall-clock match timestamp." : null,
      },
      playedOn: metadata.playedOn ?? null,
      consoleNick: metadata.consoleNick ?? null,
      slippiVersion: settings.slpVersion ?? null,
      stage: {
        id: finite(settings.stageId),
        name: settings.stageId == null ? null : safeName(stages.getStageName, settings.stageId),
      },
      rules: {
        isTeams: valueOrNull(settings.isTeams),
        isPAL: valueOrNull(settings.isPAL),
        isFrozenPokemonStadium: valueOrNull(settings.isFrozenPS),
        timerType: finite(settings.timerType),
        startingTimerSeconds: finite(settings.startingTimerSeconds),
        friendlyFireEnabled: valueOrNull(settings.friendlyFireEnabled),
        itemSpawnBehavior: finite(settings.itemSpawnBehavior),
        enabledItems: finite(settings.enabledItems),
        gameMode: finite(settings.gameMode),
      },
      frames: {
        first: firstFrame,
        firstPlayable: firstPlayableFrame,
        last: lastFrame,
        storedFrameCount: frameNumbers.length,
        playableFrameCount,
        framesPerSecond: 60,
        slpFrameToSeconds: "(slpFrame - firstPlayable) / 60",
      },
      duration: {
        seconds: durationSeconds,
        formatted: formatDuration(durationSeconds),
        lastGameTimer:
          lastFrame == null ? null : safeName(frameToGameTimer, lastFrame, settings),
      },
      gameComplete: valueOrNull(stats.gameComplete),
      matchInfo: settings.matchInfo ?? null,
      randomSeed: finite(settings.randomSeed),
      end: {
        methodId: finite(gameEnd?.gameEndMethod),
        methodName: endMethodName(gameEnd?.gameEndMethod),
        lrasInitiatorIndex: finite(gameEnd?.lrasInitiatorIndex),
      },
      winner: winnerFrom(gameEnd, finalStates),
    },
    players,
    aggregate: {
      stockCount: stats.stocks?.length ?? null,
      conversionCount: stats.conversions?.length ?? null,
      comboCount: stats.combos?.length ?? null,
    },
    video: null,
    availability: {
      trueRankOrMmr: false,
      rankNote: "SLP files do not contain true rank or MMR; rank labels may be inferred from the source filename.",
      wallClockTime: startAt != null,
    },
  };

  await fsp.mkdir(path.dirname(outputPath), { recursive: true });
  await fsp.writeFile(outputPath, `${JSON.stringify(output, null, 2)}\n`);
}

main().catch((error) => {
  console.error(error instanceof Error ? error.stack : error);
  process.exit(1);
});
