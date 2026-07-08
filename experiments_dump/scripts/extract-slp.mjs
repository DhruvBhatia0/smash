import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import {
  SlippiGame,
  characters,
  frameToGameTimer,
  stages,
} from "@slippi/slippi-js/node";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const rootDir = path.resolve(__dirname, "..");

const inputPath = path.resolve(
  rootDir,
  process.argv[2] ?? "replays/realtimeTest.slp",
);
const outputDir = path.resolve(rootDir, process.argv[3] ?? "data");
fs.mkdirSync(outputDir, { recursive: true });

const replayId = path.basename(inputPath, path.extname(inputPath));
const sourceUrl =
  "https://raw.githubusercontent.com/project-slippi/slippi-js/master/slp/realtimeTest.slp";

const game = new SlippiGame(inputPath);
const settings = game.getSettings();
const metadata = game.getMetadata();
const stats = game.getStats();
const frames = game.getFrames();

if (!settings) {
  throw new Error(`Could not read game settings from ${inputPath}`);
}

const frameNumbers = Object.keys(frames)
  .map((frame) => Number(frame))
  .filter((frame) => Number.isFinite(frame))
  .sort((a, b) => a - b);

const playerByIndex = new Map(
  (settings.players ?? []).map((player) => [player.playerIndex, player]),
);

const physicalButtonBits = {
  dpadLeft: 0x0001,
  dpadRight: 0x0002,
  dpadDown: 0x0004,
  dpadUp: 0x0008,
  z: 0x0010,
  rDigital: 0x0020,
  lDigital: 0x0040,
  a: 0x0100,
  b: 0x0200,
  x: 0x0400,
  y: 0x0800,
  start: 0x1000,
};

const processedDirectionBits = {
  joystickUp: 0x00010000,
  joystickDown: 0x00020000,
  joystickLeft: 0x00040000,
  joystickRight: 0x00080000,
  cStickUp: 0x00100000,
  cStickDown: 0x00200000,
  cStickLeft: 0x00400000,
  cStickRight: 0x00800000,
  anyTrigger: 0x80000000,
};

const decodeBits = (value, bitMap) =>
  Object.entries(bitMap)
    .filter(([, bit]) => ((value ?? 0) & bit) !== 0)
    .map(([name]) => name);

const finiteOrNull = (value) =>
  value === undefined || Number.isNaN(value) ? null : value;

const round = (value) =>
  typeof value === "number" && Number.isFinite(value)
    ? Number(value.toFixed(6))
    : finiteOrNull(value);

const compactPre = (pre) => ({
  actionStateId: finiteOrNull(pre?.actionStateId),
  position: {
    x: round(pre?.positionX),
    y: round(pre?.positionY),
  },
  facingDirection: round(pre?.facingDirection),
  percent: round(pre?.percent),
});

const compactPost = (post) => {
  const characterId = finiteOrNull(post?.internalCharacterId);
  return {
    internalCharacterId: characterId,
    characterName:
      characterId == null ? null : characters.getCharacterName(characterId),
    actionStateId: finiteOrNull(post?.actionStateId),
    position: {
      x: round(post?.positionX),
      y: round(post?.positionY),
    },
    facingDirection: round(post?.facingDirection),
    percent: round(post?.percent),
    shieldSize: round(post?.shieldSize),
    lastAttackLanded: finiteOrNull(post?.lastAttackLanded),
    currentComboCount: finiteOrNull(post?.currentComboCount),
    lastHitBy: finiteOrNull(post?.lastHitBy),
    stocksRemaining: finiteOrNull(post?.stocksRemaining),
    actionStateCounter: round(post?.actionStateCounter),
    miscActionState: round(post?.miscActionState),
    isAirborne: post?.isAirborne ?? null,
    lastGroundId: finiteOrNull(post?.lastGroundId),
    jumpsRemaining: finiteOrNull(post?.jumpsRemaining),
    lCancelStatus: finiteOrNull(post?.lCancelStatus),
    hurtboxCollisionState: finiteOrNull(post?.hurtboxCollisionState),
    selfInducedSpeeds: {
      airX: round(post?.selfInducedSpeeds?.airX),
      y: round(post?.selfInducedSpeeds?.y),
      attackX: round(post?.selfInducedSpeeds?.attackX),
      attackY: round(post?.selfInducedSpeeds?.attackY),
      groundX: round(post?.selfInducedSpeeds?.groundX),
    },
    hitlagRemaining: round(post?.hitlagRemaining),
    animationIndex: finiteOrNull(post?.animationIndex),
    instanceHitBy: finiteOrNull(post?.instanceHitBy),
    instanceId: finiteOrNull(post?.instanceId),
  };
};

const compactInput = (pre) => {
  const buttons = pre?.buttons ?? 0;
  const physicalButtons = pre?.physicalButtons ?? 0;
  return {
    joystick: {
      x: round(pre?.joystickX),
      y: round(pre?.joystickY),
      rawX: finiteOrNull(pre?.rawJoystickX),
    },
    cStick: {
      x: round(pre?.cStickX),
      y: round(pre?.cStickY),
    },
    trigger: round(pre?.trigger),
    physicalTriggers: {
      l: round(pre?.physicalLTrigger),
      r: round(pre?.physicalRTrigger),
    },
    buttons,
    physicalButtons,
    pressedButtons: decodeBits(physicalButtons, physicalButtonBits),
    processedDirections: decodeBits(buttons, processedDirectionBits),
  };
};

const playerFrameEntries = (frameEntry) =>
  (frameEntry.players ?? [])
    .map((playerFrame, playerIndex) => ({ playerIndex, playerFrame }))
    .filter(({ playerFrame }) => Boolean(playerFrame?.pre && playerFrame?.post));

const stageId = settings.stageId;
const stage = {
  id: finiteOrNull(stageId),
  name: stageId == null ? null : stages.getStageName(stageId),
};

const rows = [];
const csvRows = [];

for (const frameNumber of frameNumbers) {
  const frameEntry = frames[frameNumber];
  const players = playerFrameEntries(frameEntry);

  for (const { playerIndex, playerFrame } of players) {
    const playerSettings = playerByIndex.get(playerIndex) ?? null;
    const opponentEntries = players
      .filter((entry) => entry.playerIndex !== playerIndex)
      .map((entry) => ({
        playerIndex: entry.playerIndex,
        port: playerByIndex.get(entry.playerIndex)?.port ?? null,
        preState: compactPre(entry.playerFrame.pre),
        postState: compactPost(entry.playerFrame.post),
        controllerInput: compactInput(entry.playerFrame.pre),
      }));

    const postState = compactPost(playerFrame.post);
    const controllerInput = compactInput(playerFrame.pre);
    const row = {
      replayId,
      frame: frameNumber,
      isPlayable: frameNumber >= -39,
      gameTimer: frameToGameTimer(frameNumber, settings),
      stage,
      playerIndex,
      port: playerSettings?.port ?? null,
      startingCharacterId: finiteOrNull(playerSettings?.characterId),
      playerType: finiteOrNull(playerSettings?.type),
      controllerFix: playerSettings?.controllerFix ?? null,
      preState: compactPre(playerFrame.pre),
      postState,
      controllerInput,
      opponents: opponentEntries,
      items: frameEntry.items ?? [],
      stageEvents: frameEntry.stageEvents ?? [],
      screenshotPath: null,
    };
    rows.push(row);

    const opponent = opponentEntries[0];
    csvRows.push({
      replay_id: replayId,
      frame: frameNumber,
      is_playable: row.isPlayable,
      game_timer: row.gameTimer,
      stage_id: stage.id,
      stage_name: stage.name,
      player_index: playerIndex,
      port: row.port,
      character_id: postState.internalCharacterId,
      character_name: postState.characterName,
      action_state_id: postState.actionStateId,
      x: postState.position.x,
      y: postState.position.y,
      facing: postState.facingDirection,
      percent: postState.percent,
      shield_size: postState.shieldSize,
      stocks: postState.stocksRemaining,
      is_airborne: postState.isAirborne,
      jumps_remaining: postState.jumpsRemaining,
      hitlag_remaining: postState.hitlagRemaining,
      input_joystick_x: controllerInput.joystick.x,
      input_joystick_y: controllerInput.joystick.y,
      input_cstick_x: controllerInput.cStick.x,
      input_cstick_y: controllerInput.cStick.y,
      input_trigger: controllerInput.trigger,
      input_l_trigger: controllerInput.physicalTriggers.l,
      input_r_trigger: controllerInput.physicalTriggers.r,
      buttons: controllerInput.buttons,
      physical_buttons: controllerInput.physicalButtons,
      pressed_buttons: controllerInput.pressedButtons.join("|"),
      processed_directions: controllerInput.processedDirections.join("|"),
      opponent_player_index: opponent?.playerIndex ?? null,
      opponent_character_id: opponent?.postState?.internalCharacterId ?? null,
      opponent_character_name: opponent?.postState?.characterName ?? null,
      opponent_action_state_id: opponent?.postState?.actionStateId ?? null,
      opponent_x: opponent?.postState?.position?.x ?? null,
      opponent_y: opponent?.postState?.position?.y ?? null,
      opponent_facing: opponent?.postState?.facingDirection ?? null,
      opponent_percent: opponent?.postState?.percent ?? null,
      opponent_stocks: opponent?.postState?.stocksRemaining ?? null,
      screenshot_path: null,
    });
  }
}

const csvColumns = Object.keys(csvRows[0] ?? {});
const csvEscape = (value) => {
  if (value === null || value === undefined) return "";
  const text = String(value);
  return /[",\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
};

const replayBytes = fs.readFileSync(inputPath);
const sha256 = crypto.createHash("sha256").update(replayBytes).digest("hex");

const manifest = {
  replayId,
  generatedAt: new Date().toISOString(),
  source: {
    url: sourceUrl,
    localPath: path.relative(rootDir, inputPath),
    sha256,
    bytes: replayBytes.length,
  },
  outputs: {
    jsonl: `data/${replayId}.frames.jsonl`,
    csv: `data/${replayId}.frames.csv`,
    manifest: `data/${replayId}.manifest.json`,
  },
  settings: {
    slpVersion: settings.slpVersion,
    stage,
    players: settings.players.map((player) => ({
      playerIndex: player.playerIndex,
      port: player.port,
      characterId: player.characterId,
      type: player.type,
      startStocks: player.startStocks,
      controllerFix: player.controllerFix,
      displayName: player.displayName,
      connectCode: player.connectCode,
    })),
    isTeams: settings.isTeams,
    timerType: settings.timerType,
    startingTimerSeconds: settings.startingTimerSeconds,
    isPAL: settings.isPAL,
    playedOn: metadata?.playedOn ?? null,
  },
  metadata,
  stats: {
    gameComplete: stats?.gameComplete ?? null,
    lastFrame: stats?.lastFrame ?? metadata?.lastFrame ?? null,
    playableFrameCount: stats?.playableFrameCount ?? null,
  },
  counts: {
    frameEntries: frameNumbers.length,
    playerFrameRows: rows.length,
    firstFrame: frameNumbers[0] ?? null,
    firstPlayableFrame: -39,
    timerStartFrame: 0,
    lastFrame: frameNumbers.at(-1) ?? null,
  },
  screenshotStatus: {
    available: false,
    reason:
      "The .slp contains structured state and inputs, not pixels. Rendering screenshots requires replaying the file in Slippi/Dolphin with a Melee ISO; no Slippi/Dolphin app was found locally during this run.",
  },
};

fs.writeFileSync(
  path.join(outputDir, `${replayId}.frames.jsonl`),
  `${rows.map((row) => JSON.stringify(row)).join("\n")}\n`,
);

fs.writeFileSync(
  path.join(outputDir, `${replayId}.frames.csv`),
  [
    csvColumns.join(","),
    ...csvRows.map((row) => csvColumns.map((column) => csvEscape(row[column])).join(",")),
  ].join("\n") + "\n",
);

fs.writeFileSync(
  path.join(outputDir, `${replayId}.manifest.json`),
  `${JSON.stringify(manifest, null, 2)}\n`,
);

console.log(
  JSON.stringify(
    {
      replayId,
      source: manifest.source,
      stage,
      frameEntries: manifest.counts.frameEntries,
      playerFrameRows: manifest.counts.playerFrameRows,
      outputs: manifest.outputs,
      screenshotAvailable: manifest.screenshotStatus.available,
    },
    null,
    2,
  ),
);

