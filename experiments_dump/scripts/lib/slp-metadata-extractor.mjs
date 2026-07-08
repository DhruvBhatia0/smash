import fs from "node:fs/promises";
import path from "node:path";
import { SlippiGame, characters, frameToGameTimer, stages } from "@slippi/slippi-js/node";
import { FileHasher } from "./file-hasher.mjs";

const finiteOrNull = (value) =>
  typeof value === "number" && Number.isFinite(value) ? value : null;

const ratioOrNull = (value) => finiteOrNull(value?.ratio);
const countOrNull = (value) => finiteOrNull(value?.count);

export class SlpMetadataExtractor {
  async extract(filePath) {
    const game = new SlippiGame(filePath);
    const settings = game.getSettings();
    const metadata = game.getMetadata();
    const stats = game.getStats();
    const frames = game.getFrames();

    if (!settings) {
      throw new Error(`Could not read settings from ${filePath}`);
    }

    const frameNumbers = Object.keys(frames)
      .map(Number)
      .filter(Number.isFinite)
      .sort((a, b) => a - b);

    const fileStat = await fs.stat(filePath);
    const sha256 = await FileHasher.sha256(filePath);
    const gameInfo = this.extractGameInfo(settings, metadata, stats, frameNumbers);
    const players = this.extractPlayers(settings, metadata, stats, frames);
    const winner = this.extractWinner(settings, players);

    return {
      schemaVersion: 1,
      generatedAt: new Date().toISOString(),
      file: {
        path: filePath,
        name: path.basename(filePath),
        bytes: fileStat.size,
        sha256,
      },
      game: gameInfo,
      winner,
      players,
      extractionNotes: {
        truePlayerRank: "Not present in ordinary .slp files. skillSignals are replay-derived proxies, not MMR/rank.",
      },
    };
  }

  extractGameInfo(settings, metadata, stats, frameNumbers) {
    const lastFrame =
      finiteOrNull(stats?.lastFrame) ??
      finiteOrNull(metadata?.lastFrame) ??
      frameNumbers.at(-1) ??
      null;

    return {
      slpVersion: settings.slpVersion ?? null,
      playedOn: metadata?.playedOn ?? null,
      startAt: metadata?.startAt ?? null,
      consoleNick: metadata?.consoleNick ?? null,
      stage: {
        id: finiteOrNull(settings.stageId),
        name: settings.stageId == null ? null : stages.getStageName(settings.stageId),
      },
      isTeams: settings.isTeams ?? null,
      isPAL: settings.isPAL ?? null,
      timerType: finiteOrNull(settings.timerType),
      startingTimerSeconds: finiteOrNull(settings.startingTimerSeconds),
      gameComplete: stats?.gameComplete ?? null,
      firstFrame: frameNumbers[0] ?? null,
      firstPlayableFrame: -39,
      lastFrame,
      playableFrameCount: finiteOrNull(stats?.playableFrameCount),
      lastGameTimer:
        lastFrame == null || !settings
          ? null
          : frameToGameTimer(lastFrame, settings),
      matchInfo: settings.matchInfo ?? null,
    };
  }

  extractPlayers(settings, metadata, stats, frames) {
    const overallByPlayer = new Map(
      (stats?.overall ?? []).map((entry) => [entry.playerIndex, entry]),
    );
    const actionsByPlayer = new Map(
      (stats?.actionCounts ?? []).map((entry) => [entry.playerIndex, entry]),
    );
    const stocksByPlayer = this.finalStocksByPlayer(stats?.stocks ?? []);
    const metadataPlayers = metadata?.players ?? {};

    return (settings.players ?? []).map((player) => {
      const playerIndex = player.playerIndex;
      const frameSample = this.findFirstPlayerFrame(frames, playerIndex);
      const dominantCharacter = this.dominantCharacter(metadataPlayers[playerIndex]);
      const internalCharacterId =
        dominantCharacter?.internalCharacterId ??
        finiteOrNull(frameSample?.post?.internalCharacterId);
      const finalStock = stocksByPlayer.get(playerIndex) ?? null;
      const overall = overallByPlayer.get(playerIndex) ?? null;
      const actions = actionsByPlayer.get(playerIndex) ?? null;

      return {
        playerIndex,
        port: finiteOrNull(player.port),
        teamId: finiteOrNull(player.teamId),
        type: finiteOrNull(player.type),
        controllerFix: player.controllerFix ?? null,
        nametag: player.nametag ?? null,
        displayName: player.displayName || null,
        connectCode: player.connectCode || null,
        userId: player.userId || null,
        character: {
          settingsCharacterId: finiteOrNull(player.characterId),
          internalCharacterId,
          name:
            internalCharacterId == null
              ? null
              : characters.getCharacterName(internalCharacterId),
          framesPlayed: dominantCharacter?.framesPlayed ?? null,
        },
        finalState: {
          stocksRemaining: finalStock?.count ?? null,
          percent: finalStock?.currentPercent ?? null,
          lastStockStartedAtFrame: finalStock?.startFrame ?? null,
        },
        skillSignals: this.extractSkillSignals(overall, actions),
      };
    });
  }

  extractSkillSignals(overall, actions) {
    return {
      source: "slippi-js stats; not a player rank",
      rank: null,
      inputCounts: overall?.inputCounts ?? null,
      inputsPerMinute: ratioOrNull(overall?.inputsPerMinute),
      digitalInputsPerMinute: ratioOrNull(overall?.digitalInputsPerMinute),
      conversionCount: finiteOrNull(overall?.conversionCount),
      killCount: finiteOrNull(overall?.killCount),
      totalDamage: finiteOrNull(overall?.totalDamage),
      openingsPerKill: ratioOrNull(overall?.openingsPerKill),
      damagePerOpening: ratioOrNull(overall?.damagePerOpening),
      neutralWinRatio: ratioOrNull(overall?.neutralWinRatio),
      successfulConversionRatio: ratioOrNull(overall?.successfulConversions),
      actionCounts: actions
        ? {
            wavedashCount: finiteOrNull(actions.wavedashCount),
            wavelandCount: finiteOrNull(actions.wavelandCount),
            dashDanceCount: finiteOrNull(actions.dashDanceCount),
            ledgegrabCount: finiteOrNull(actions.ledgegrabCount),
            rollCount: finiteOrNull(actions.rollCount),
            lCancelSuccess: finiteOrNull(actions.lCancelCount?.success),
            lCancelFail: finiteOrNull(actions.lCancelCount?.fail),
            grabSuccess: finiteOrNull(actions.grabCount?.success),
            grabFail: finiteOrNull(actions.grabCount?.fail),
          }
        : null,
      rawTotals: overall
        ? {
            neutralWinCount: countOrNull(overall.neutralWinRatio),
            neutralWinTotal: finiteOrNull(overall.neutralWinRatio?.total),
            successfulConversions: countOrNull(overall.successfulConversions),
            successfulConversionTotal: finiteOrNull(overall.successfulConversions?.total),
          }
        : null,
    };
  }

  extractWinner(settings, players) {
    if (players.length === 0) {
      return { playerIndex: null, teamId: null, method: "unavailable" };
    }

    if (settings.isTeams) {
      return this.extractTeamWinner(players);
    }

    const ranked = [...players].sort((a, b) => {
      const stockDelta =
        (b.finalState.stocksRemaining ?? -1) - (a.finalState.stocksRemaining ?? -1);
      if (stockDelta !== 0) return stockDelta;
      return (a.finalState.percent ?? Infinity) - (b.finalState.percent ?? Infinity);
    });

    const best = ranked[0];
    const second = ranked[1] ?? null;
    const isTie =
      second &&
      best.finalState.stocksRemaining === second.finalState.stocksRemaining &&
      best.finalState.percent === second.finalState.percent;

    if (isTie) {
      return { playerIndex: null, teamId: null, method: "tie-or-unresolved" };
    }

    return {
      playerIndex: best.playerIndex,
      teamId: settings.isTeams ? best.teamId : null,
      method: "derived-from-final-stocks-and-percent",
      placements: ranked.map((player, index) => ({
        place: index + 1,
        playerIndex: player.playerIndex,
        teamId: player.teamId,
        stocksRemaining: player.finalState.stocksRemaining,
        percent: player.finalState.percent,
      })),
    };
  }

  extractTeamWinner(players) {
    const teamMap = new Map();

    for (const player of players) {
      if (player.teamId == null || player.finalState.stocksRemaining == null) {
        return { playerIndex: null, teamId: null, method: "team-unresolved" };
      }

      const team = teamMap.get(player.teamId) ?? {
        teamId: player.teamId,
        playerIndexes: [],
        stocksRemaining: 0,
        percent: 0,
      };

      team.playerIndexes.push(player.playerIndex);
      team.stocksRemaining += player.finalState.stocksRemaining;
      team.percent += player.finalState.percent ?? 0;
      teamMap.set(player.teamId, team);
    }

    const rankedTeams = [...teamMap.values()].sort((a, b) => {
      const stockDelta = b.stocksRemaining - a.stocksRemaining;
      if (stockDelta !== 0) return stockDelta;
      return a.percent - b.percent;
    });

    const best = rankedTeams[0] ?? null;
    const second = rankedTeams[1] ?? null;
    const isTie =
      second &&
      best?.stocksRemaining === second.stocksRemaining &&
      best?.percent === second.percent;

    if (!best || isTie) {
      return { playerIndex: null, teamId: null, method: "team-tie-or-unresolved" };
    }

    return {
      playerIndex: null,
      teamId: best.teamId,
      method: "team-derived-from-final-stocks-and-percent",
      placements: rankedTeams.map((team, index) => ({
        place: index + 1,
        teamId: team.teamId,
        playerIndexes: team.playerIndexes,
        stocksRemaining: team.stocksRemaining,
        percent: team.percent,
      })),
    };
  }

  finalStocksByPlayer(stocks) {
    const latest = new Map();
    for (const stock of stocks) {
      const existing = latest.get(stock.playerIndex);
      if (!existing || (stock.startFrame ?? -Infinity) > (existing.startFrame ?? -Infinity)) {
        latest.set(stock.playerIndex, stock);
      }
    }
    return latest;
  }

  dominantCharacter(metadataPlayer) {
    const entries = Object.entries(metadataPlayer?.characters ?? {})
      .map(([internalCharacterId, framesPlayed]) => ({
        internalCharacterId: Number(internalCharacterId),
        framesPlayed,
      }))
      .filter((entry) => Number.isFinite(entry.internalCharacterId));

    return entries.sort((a, b) => b.framesPlayed - a.framesPlayed)[0] ?? null;
  }

  findFirstPlayerFrame(frames, playerIndex) {
    const frameNumbers = Object.keys(frames)
      .map(Number)
      .filter(Number.isFinite)
      .sort((a, b) => a - b);

    for (const frameNumber of frameNumbers) {
      const frame = frames[frameNumber]?.players?.[playerIndex];
      if (frame?.post) return frame;
    }

    return null;
  }
}
