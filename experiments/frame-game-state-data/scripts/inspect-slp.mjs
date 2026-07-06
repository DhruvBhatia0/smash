import path from "node:path";
import { fileURLToPath } from "node:url";
import { SlippiGame, characters, stages } from "@slippi/slippi-js/node";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const rootDir = path.resolve(__dirname, "..");
const inputPath = path.resolve(rootDir, process.argv[2] ?? "replays/realtimeTest.slp");

const game = new SlippiGame(inputPath);
const settings = game.getSettings();
const metadata = game.getMetadata();
const frames = game.getFrames();
const frameNumbers = Object.keys(frames)
  .map(Number)
  .filter(Number.isFinite)
  .sort((a, b) => a - b);

const firstPlayableFrame = frames[0];
const activePlayers = (firstPlayableFrame?.players ?? [])
  .map((playerFrame, playerIndex) => ({ playerIndex, playerFrame }))
  .filter(({ playerFrame }) => Boolean(playerFrame?.pre && playerFrame?.post))
  .map(({ playerIndex, playerFrame }) => ({
    playerIndex,
    port: settings?.players?.find((player) => player.playerIndex === playerIndex)?.port,
    characterId: playerFrame.post.internalCharacterId,
    characterName: characters.getCharacterName(playerFrame.post.internalCharacterId),
    sampleInput: {
      joystickX: playerFrame.pre.joystickX,
      joystickY: playerFrame.pre.joystickY,
      cStickX: playerFrame.pre.cStickX,
      cStickY: playerFrame.pre.cStickY,
      trigger: playerFrame.pre.trigger,
      buttons: playerFrame.pre.buttons,
      physicalButtons: playerFrame.pre.physicalButtons,
    },
    sampleState: {
      actionStateId: playerFrame.post.actionStateId,
      positionX: playerFrame.post.positionX,
      positionY: playerFrame.post.positionY,
      percent: playerFrame.post.percent,
      stocksRemaining: playerFrame.post.stocksRemaining,
    },
  }));

console.log(
  JSON.stringify(
    {
      inputPath: path.relative(rootDir, inputPath),
      slpVersion: settings?.slpVersion,
      stageId: settings?.stageId,
      stageName: settings?.stageId == null ? null : stages.getStageName(settings.stageId),
      playedOn: metadata?.playedOn ?? null,
      metadataLastFrame: metadata?.lastFrame ?? null,
      firstFrame: frameNumbers[0] ?? null,
      firstPlayableFrame: -39,
      timerStartFrame: 0,
      lastFrame: frameNumbers.at(-1) ?? null,
      frameEntryCount: frameNumbers.length,
      activePlayers,
    },
    null,
    2,
  ),
);

