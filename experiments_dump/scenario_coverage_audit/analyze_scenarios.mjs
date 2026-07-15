import fs from "node:fs";
import path from "node:path";
import process from "node:process";
import {
  ItemSpawnType,
  MoveId,
  SlippiGame,
  State,
  characters,
  isCommandGrabbed,
  isDamaged,
  isDead,
  isDown,
  isGrabbed,
  isInControl,
  isTeching,
  moves,
  stages,
} from "@slippi/slippi-js/node";

const ROOT = process.env.AUDIT_ROOT ?? "/home/daytona/smash-scenario-audit";
const REPLAY_DIR = process.env.AUDIT_REPLAY_DIR ?? path.join(ROOT, "replays");
const OUTPUT = path.join(ROOT, "scenario-analysis.json");
const FIRST_PLAYABLE = -39;
const BATTLEFIELD_EDGE_X = 68.4;
const GAMEPLAY_BUTTON_MASK = 0x0ff0;
const KNOWN_OBJECT_TYPES = {
  0x22: "Poke Ball",
  0x36: "Fox Laser",
  0x38: "Fox Illusion",
  0x4a: "Fox Blaster",
};

function inc(object, key, amount = 1) {
  object[key] = (object[key] ?? 0) + amount;
}

function addDeep(target, source) {
  for (const [key, value] of Object.entries(source ?? {})) {
    if (key === "playerIndex") continue;
    if (typeof value === "number" && Number.isFinite(value)) {
      target[key] = (target[key] ?? 0) + value;
    } else if (value && typeof value === "object" && !Array.isArray(value)) {
      target[key] ??= {};
      addDeep(target[key], value);
    }
  }
}

function number(value, fallback = 0) {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

function versionAtLeast(version, target) {
  const left = String(version ?? "0").split(".").map(Number);
  const right = target.split(".").map(Number);
  for (let index = 0; index < 3; index += 1) {
    if ((left[index] ?? 0) !== (right[index] ?? 0)) {
      return (left[index] ?? 0) > (right[index] ?? 0);
    }
  }
  return true;
}

function characterName(id) {
  try {
    return characters.getCharacterName(id);
  } catch {
    return `UnknownCharacter(${id})`;
  }
}

function stageName(id) {
  try {
    return stages.getStageName(id);
  } catch {
    return `UnknownStage(${id})`;
  }
}

function moveName(id) {
  try {
    return moves.getMoveShortName(id);
  } catch {
    return MoveId[id] ?? `move_${id}`;
  }
}

function percentBin(percent) {
  if (percent <= 0) return "0";
  if (percent < 40) return "1-39";
  if (percent < 80) return "40-79";
  if (percent < 120) return "80-119";
  if (percent < 160) return "120-159";
  if (percent < 200) return "160-199";
  return "200+";
}

function xBin(x) {
  if (x < -100) return "far-left";
  if (x < -35) return "left";
  if (x <= 35) return "center";
  if (x <= 100) return "right";
  return "far-right";
}

function yBin(y) {
  if (y < -25) return "deep-below";
  if (y < 0) return "below";
  if (y < 25) return "low";
  if (y < 60) return "mid";
  return "high";
}

function actionCategory(state) {
  if (isDead(state)) return "dead-or-respawn";
  if (isDamaged(state)) return "damage-or-hitstun";
  if (isGrabbed(state) || isCommandGrabbed(state)) return "captured";
  if (isDown(state)) return "knockdown-or-missed-tech";
  if (isTeching(state)) return "tech";
  if (state >= 252 && state <= 263) return "ledge";
  if (state >= State.GUARD_START && state <= State.GUARD_END) return "shield";
  if (state >= State.GROUND_ATTACK_START && state <= State.GROUND_ATTACK_END) return "ground-attack";
  if (state >= State.AERIAL_ATTACK_START && state <= 69) return "aerial-attack";
  if (state >= State.AERIAL_LANDING_START && state <= State.AERIAL_LANDING_END) return "aerial-landing";
  if (state === State.ROLL_FORWARD || state === State.ROLL_BACKWARD || state === State.SPOT_DODGE || state === State.AIR_DODGE) return "dodge-or-roll";
  if (state >= 212 && state <= 222) return "grab-pummel-or-throw";
  if (state === State.ACTION_WAIT) return "standing-idle";
  if (state >= State.GROUNDED_CONTROL_START && state < State.CONTROLLED_JUMP_START) return "ground-movement";
  if (state >= State.CONTROLLED_JUMP_START && state <= State.CONTROLLED_JUMP_END) return "jump-or-air-movement";
  if (state >= State.SQUAT_START && state <= State.SQUAT_END) return "crouch";
  if (state === State.LANDING_FALL_SPECIAL) return "special-landing";
  if (state >= 266) return "character-special-or-unique";
  if (isInControl(state)) return "other-in-control";
  return "other-shared-state";
}

function semanticZone(post) {
  const state = post.actionStateId ?? -1;
  const x = number(post.positionX);
  const y = number(post.positionY);
  if (isDead(state)) return "dead-or-respawn";
  if (state >= 252 && state <= 263) return x < 0 ? "left-ledge" : "right-ledge";
  if (y < 0) {
    if (Math.abs(x) < 55) return "under-stage";
    return x < 0 ? "below-left-offstage" : "below-right-offstage";
  }
  if (Math.abs(x) > BATTLEFIELD_EDGE_X) return x < 0 ? "side-left-offstage" : "side-right-offstage";
  if (!post.isAirborne) {
    if (y > 35) return "top-platform";
    if (y > 8) return x < 0 ? "left-platform" : "right-platform";
    if (x < -35) return "main-floor-left";
    if (x > 35) return "main-floor-right";
    return "main-floor-center";
  }
  if (y >= 60) return "above-stage-high";
  if (y >= 25) return "above-stage-mid";
  return "above-stage-low";
}

function offstage(post) {
  if (isDead(post.actionStateId ?? -1)) return false;
  return number(post.positionY) < 0 || Math.abs(number(post.positionX)) > BATTLEFIELD_EDGE_X;
}

function offscreenProxy(post) {
  if (isDead(post.actionStateId ?? -1)) return false;
  const x = Math.abs(number(post.positionX));
  const y = number(post.positionY);
  return x > 110 || y > 80 || y < -25;
}

function distanceBin(a, b) {
  const distance = Math.hypot(number(a.positionX) - number(b.positionX), number(a.positionY) - number(b.positionY));
  if (distance < 15) return "close";
  if (distance < 40) return "medium";
  if (distance < 80) return "far";
  return "extreme";
}

function facingRelationship(a, b) {
  const aToward = (number(b.positionX) - number(a.positionX)) * number(a.facingDirection, 1) >= 0;
  const bToward = (number(a.positionX) - number(b.positionX)) * number(b.facingDirection, 1) >= 0;
  if (aToward && bToward) return "both-facing-toward";
  if (!aToward && !bToward) return "both-facing-away";
  return "one-facing-toward";
}

function strictStill(frame, previous) {
  const post = frame?.post;
  const pre = frame?.pre;
  if (!post || !pre || !previous?.post) return false;
  return (
    post.actionStateId === State.ACTION_WAIT &&
    !post.isAirborne &&
    (Number(pre.physicalButtons ?? 0) & GAMEPLAY_BUTTON_MASK) === 0 &&
    Math.abs(number(pre.joystickX)) < 0.15 &&
    Math.abs(number(pre.joystickY)) < 0.15 &&
    Math.abs(number(pre.cStickX)) < 0.15 &&
    Math.abs(number(pre.cStickY)) < 0.15 &&
    Math.abs(number(post.positionX) - number(previous.post.positionX)) < 0.03 &&
    Math.abs(number(post.positionY) - number(previous.post.positionY)) < 0.03
  );
}

function bucketEpisode(frames) {
  if (frames < 30) return "under-0.5s";
  if (frames < 60) return "0.5-1s";
  if (frames < 120) return "1-2s";
  if (frames < 300) return "2-5s";
  return "5s+";
}

const coverage = new Map();
function markCoverage(key, replayFile, player = null, amount = 1) {
  let entry = coverage.get(key);
  if (!entry) {
    entry = { frames20: 0, matches: new Set(), players: new Set() };
    coverage.set(key, entry);
  }
  entry.frames20 += amount;
  entry.matches.add(replayFile);
  if (player != null) entry.players.add(`${replayFile}:${player}`);
}

const examples = {};
function saveExample(key, value, limit = 5) {
  examples[key] ??= [];
  if (examples[key].length < limit) examples[key].push(value);
}

const aggregate = {
  format: "MELEE_SCENARIO_COVERAGE_AUDIT_V1",
  generatedAt: new Date().toISOString(),
  definitions: {
    sampledClock: "20 Hz post-frame samples; raw events and episodes are detected at native 60 Hz",
    offstage: "Battlefield proxy: alive and y<0 or |x|>68.4",
    offscreenProxy: "coordinate proxy only: alive and |x|>110 or y>80 or y<-25; exact magnifying-glass flags are in offscreen-analysis.json",
    standingStill: "WAIT state, grounded, neutral sticks/buttons, and <0.03 world-unit position delta at 60 Hz",
    itemTelemetry: "runtime item updates require SLP 3.0+; object stream includes projectiles/articles, not only pickup items",
  },
  replayFilesFound: 0,
  replayFilesAnalyzed: 0,
  failures: [],
  frames: { world60: 0, player60: 0, world20: 0, player20: 0 },
  settings: {
    slpVersions: {}, stages: {}, matchups: {}, playerTypes: {}, itemSpawnBehavior: {}, itemOffMatches: 0,
    itemOnMatches: 0, teamsMatches: 0, humanOnlyMatches: 0, gameCompleteMatches: 0,
  },
  telemetry: {
    itemUpdateCapableMatches: 0, itemOwnerCapableMatches: 0, instanceAttributionCapableMatches: 0,
    matchesWithRuntimeItemArrays: 0,
  },
  distributions: {
    characters: {}, actionCategories: {}, actionStates: {}, percentBins: {}, stocks: {},
    semanticZones: {}, xyBins: {}, jointZones: {}, jointXyBins: {}, jointActionCategories: {},
    jointZoneActions: {}, offstageConfiguration: {},
    offscreenProxyConfiguration: {}, distanceBins: {}, facingRelationship: {}, stockRelationship: {},
    percentRelationship: {},
  },
  stationary: {
    strictStillFrames60: 0, mutualStillFrames60: 0, episodes: 0, mutualEpisodes: 0,
    episodeDurations: {}, mutualEpisodeDurations: {}, maxFrames: 0, maxMutualFrames: 0,
  },
  vulnerability: {
    damageFrames60: 0, hitlagFrames60: 0, capturedFrames60: 0, downFrames60: 0,
    techFrames60: 0, comboVictimFrames20: 0, comboAttackerFrames20: 0,
  },
  combos: {
    total: 0, kills: 0, zeroToDeaths: 0, totalDamage: 0, totalHits: 0,
    tightDurationFrames: 0, statDurationFrames: 0, hitBuckets: {}, damageBuckets: {},
    durationBuckets: {}, sequences: {}, landedMoves: {}, landedSmashAttacks: {},
    landedSpecials: {}, openingTypes: {}, longBeatdowns: 0,
  },
  techniques: {},
  items: {
    matchesWithObjects: 0, objectUpdateRows: 0, uniqueObjects: 0, basePickupObjects: 0,
    projectileOrArticleObjects: 0, pokeBallObjects: 0, pokemonObjects: 0,
    pickupOwnerTransitions: 0, attributedHitProxies: 0, typeIds: {},
  },
  replaySummaries: [],
};

function finishStillEpisode(replayFile, player, start, end, mutual = false) {
  const frames = end - start + 1;
  if (frames < 15) return;
  const target = mutual ? "mutual" : "player";
  if (mutual) {
    aggregate.stationary.mutualEpisodes += 1;
    aggregate.stationary.maxMutualFrames = Math.max(aggregate.stationary.maxMutualFrames, frames);
    inc(aggregate.stationary.mutualEpisodeDurations, bucketEpisode(frames));
  } else {
    aggregate.stationary.episodes += 1;
    aggregate.stationary.maxFrames = Math.max(aggregate.stationary.maxFrames, frames);
    inc(aggregate.stationary.episodeDurations, bucketEpisode(frames));
  }
  if (frames >= 60) saveExample(`${target}-standing-still-1s+`, { replayFile, player, startFrame: start, endFrame: end, frames, seconds: frames / 60 });
}

const replayFiles = fs.readdirSync(REPLAY_DIR).filter((file) => file.endsWith(".slp")).sort();
aggregate.replayFilesFound = replayFiles.length;

for (let replayIndex = 0; replayIndex < replayFiles.length; replayIndex += 1) {
  const replayFile = replayFiles[replayIndex];
  try {
    const game = new SlippiGame(path.join(REPLAY_DIR, replayFile));
    const settings = game.getSettings();
    const metadata = game.getMetadata();
    const frames = game.getFrames();
    const stats = game.getStats();
    const players = (settings?.players ?? []).map((player) => player.playerIndex).sort((a, b) => a - b);
    if (players.length !== 2 || settings?.isTeams) throw new Error("expected two-player singles");
    const playerSettings = new Map(settings.players.map((player) => [player.playerIndex, player]));
    const characterByPlayer = new Map(settings.players.map((player) => [player.playerIndex, characterName(player.characterId)]));
    const version = settings.slpVersion ?? "unknown";
    const stage = stageName(settings.stageId);
    const matchup = [...characterByPlayer.values()].sort().join(" vs ");
    inc(aggregate.settings.slpVersions, version);
    inc(aggregate.settings.stages, stage);
    inc(aggregate.settings.matchups, matchup);
    inc(aggregate.settings.itemSpawnBehavior, String(settings.itemSpawnBehavior));
    aggregate.settings.itemOffMatches += Number(settings.itemSpawnBehavior === ItemSpawnType.OFF || settings.itemSpawnBehavior === 255);
    aggregate.settings.itemOnMatches += Number(settings.itemSpawnBehavior !== ItemSpawnType.OFF && settings.itemSpawnBehavior !== 255);
    aggregate.settings.teamsMatches += Number(settings.isTeams);
    aggregate.settings.humanOnlyMatches += Number(settings.players.every((player) => player.type === 0));
    aggregate.settings.gameCompleteMatches += Number(stats?.gameComplete);
    aggregate.telemetry.itemUpdateCapableMatches += Number(versionAtLeast(version, "3.0.0"));
    aggregate.telemetry.itemOwnerCapableMatches += Number(versionAtLeast(version, "3.6.0"));
    aggregate.telemetry.instanceAttributionCapableMatches += Number(versionAtLeast(version, "3.16.0"));
    for (const player of settings.players) inc(aggregate.settings.playerTypes, String(player.type));
    for (const name of characterByPlayer.values()) inc(aggregate.distributions.characters, name);

    const available = Object.keys(frames).map(Number).filter((frame) => frame >= FIRST_PLAYABLE).sort((a, b) => a - b);
    const valid = available.filter((frame) => players.every((player) => frames[frame]?.players?.[player]?.post));
    if (!valid.length) throw new Error("no valid playable frames");
    const first = valid[0];
    const last = valid.at(-1);
    const comboRoles = new Map(players.map((player) => [player, new Map()]));
    for (const combo of stats?.combos ?? []) {
      if (!(combo.moves?.length)) continue;
      const attacker = combo.moves[0].playerIndex ?? combo.lastHitBy;
      const victim = combo.playerIndex;
      const lastMoveFrame = combo.moves.at(-1).frame;
      for (let frame = combo.startFrame; frame <= lastMoveFrame + 10; frame += 1) {
        comboRoles.get(attacker)?.set(frame, "attacker");
        comboRoles.get(victim)?.set(frame, "victim");
      }
    }

    const stillStart = new Map(players.map((player) => [player, null]));
    let mutualStart = null;
    const itemObjects = new Map();
    let sawRuntimeItemArray = false;
    let replayObjectUpdates = 0;
    const replayFlags = {};
    let previousFrame = null;
    let replayWorld60 = 0;
    let replayWorld20 = 0;

    for (const frameNumber of valid) {
      const frame = frames[frameNumber];
      const current = players.map((player) => frame.players[player]);
      replayWorld60 += 1;
      aggregate.frames.world60 += 1;
      aggregate.frames.player60 += players.length;
      const still = [];
      const activeInstanceIds = new Set((frame.items ?? []).map((item) => item.instanceId).filter((value) => value != null));
      for (let index = 0; index < players.length; index += 1) {
        const player = players[index];
        const value = current[index];
        const post = value.post;
        const state = post.actionStateId ?? -1;
        const category = actionCategory(state);
        inc(aggregate.distributions.actionCategories, `${characterByPlayer.get(player)}:${category}`);
        inc(aggregate.distributions.actionStates, `${characterByPlayer.get(player)}:${state}`);
        aggregate.vulnerability.damageFrames60 += Number(isDamaged(state));
        aggregate.vulnerability.hitlagFrames60 += Number(number(post.hitlagRemaining) > 0);
        aggregate.vulnerability.capturedFrames60 += Number(isGrabbed(state) || isCommandGrabbed(state));
        aggregate.vulnerability.downFrames60 += Number(isDown(state));
        aggregate.vulnerability.techFrames60 += Number(isTeching(state));
        if (previousFrame) {
          const previousPost = previousFrame.players[player]?.post;
          if (previousPost && number(post.percent) > number(previousPost.percent) && post.instanceHitBy != null && activeInstanceIds.has(post.instanceHitBy)) {
            aggregate.items.attributedHitProxies += 1;
          }
        }
        const isStill = strictStill(value, previousFrame?.players?.[player]);
        still.push(isStill);
        aggregate.stationary.strictStillFrames60 += Number(isStill);
        if (isStill && stillStart.get(player) == null) stillStart.set(player, frameNumber);
        if (!isStill && stillStart.get(player) != null) {
          finishStillEpisode(replayFile, player, stillStart.get(player), frameNumber - 1);
          stillStart.set(player, null);
        }
      }
      const mutual = still.every(Boolean);
      aggregate.stationary.mutualStillFrames60 += Number(mutual);
      if (mutual && mutualStart == null) mutualStart = frameNumber;
      if (!mutual && mutualStart != null) {
        finishStillEpisode(replayFile, "both", mutualStart, frameNumber - 1, true);
        mutualStart = null;
      }

      if (Object.prototype.hasOwnProperty.call(frame, "items")) sawRuntimeItemArray = true;
      for (let itemIndex = 0; itemIndex < (frame.items ?? []).length; itemIndex += 1) {
        const item = frame.items[itemIndex];
        aggregate.items.objectUpdateRows += 1;
        replayObjectUpdates += 1;
        const key = `${item.spawnId ?? `no-spawn:${item.typeId}:${itemIndex}`}`;
        let object = itemObjects.get(key);
        if (!object) {
          object = { typeId: item.typeId, firstFrame: frameNumber, lastFrame: frameNumber, owners: new Set(), previousOwner: item.owner };
          itemObjects.set(key, object);
        }
        object.lastFrame = frameNumber;
        if (item.owner != null) object.owners.add(item.owner);
        if ((object.previousOwner == null || object.previousOwner < 0) && item.owner != null && item.owner >= 0 && item.owner <= 3) {
          aggregate.items.pickupOwnerTransitions += 1;
        }
        object.previousOwner = item.owner;
      }

      if ((frameNumber - FIRST_PLAYABLE) % 3 === 0) {
        replayWorld20 += 1;
        aggregate.frames.world20 += 1;
        aggregate.frames.player20 += players.length;
        const posts = current.map((value) => value.post);
        for (let index = 0; index < players.length; index += 1) {
          const player = players[index];
          const post = posts[index];
          const char = characterByPlayer.get(player);
          const category = actionCategory(post.actionStateId ?? -1);
          const zone = semanticZone(post);
          const xy = `${xBin(number(post.positionX))}/${yBin(number(post.positionY))}`;
          const percent = percentBin(number(post.percent));
          const stocks = String(post.stocksRemaining ?? "unknown");
          inc(aggregate.distributions.percentBins, `${char}:${percent}`);
          inc(aggregate.distributions.stocks, `${char}:${stocks}`);
          inc(aggregate.distributions.semanticZones, `${char}:${zone}`);
          inc(aggregate.distributions.xyBins, `${char}:${xy}`);
          markCoverage(`character=${char}`, replayFile, player);
          markCoverage(`action=${char}:${category}`, replayFile, player);
          markCoverage(`state=${char}:${post.actionStateId}`, replayFile, player);
          markCoverage(`zone=${char}:${zone}`, replayFile, player);
          markCoverage(`zone-action=${char}:${zone}:${category}`, replayFile, player);
          markCoverage(`xy=${char}:${xy}`, replayFile, player);
          markCoverage(`percent=${char}:${percent}`, replayFile, player);
          markCoverage(`stocks=${char}:${stocks}`, replayFile, player);
          const role = comboRoles.get(player)?.get(frameNumber);
          if (role === "victim") aggregate.vulnerability.comboVictimFrames20 += 1;
          if (role === "attacker") aggregate.vulnerability.comboAttackerFrames20 += 1;
          if (role) {
            markCoverage(`combo-role=${char}:${role}`, replayFile, player);
            markCoverage(`combo-role-zone=${char}:${role}:${zone}`, replayFile, player);
            markCoverage(`combo-role-action=${char}:${role}:${category}`, replayFile, player);
          }
        }
        const zonePair = posts.map(semanticZone).join(" | ");
        const xyPair = posts.map((post) => `${xBin(number(post.positionX))}/${yBin(number(post.positionY))}`).join(" | ");
        const actionPair = posts.map((post) => actionCategory(post.actionStateId ?? -1)).join(" | ");
        const zoneActionPair = posts.map((post) => `${semanticZone(post)}/${actionCategory(post.actionStateId ?? -1)}`).join(" | ");
        inc(aggregate.distributions.jointZones, zonePair);
        inc(aggregate.distributions.jointXyBins, xyPair);
        inc(aggregate.distributions.jointActionCategories, actionPair);
        inc(aggregate.distributions.jointZoneActions, zoneActionPair);
        const offstageValues = posts.map(offstage);
        const offstageConfig = offstageValues.every(Boolean) ? "both-offstage" : offstageValues[0] ? "player0-only-offstage" : offstageValues[1] ? "player1-only-offstage" : "both-onstage";
        inc(aggregate.distributions.offstageConfiguration, offstageConfig);
        inc(replayFlags, offstageConfig);
        markCoverage(`joint-zone=${zonePair}`, replayFile);
        markCoverage(`joint-action=${actionPair}`, replayFile);
        markCoverage(`joint-zone-action=${zoneActionPair}`, replayFile);
        markCoverage(`offstage=${offstageConfig}`, replayFile);
        const proxyValues = posts.map(offscreenProxy);
        const proxyConfig = proxyValues.every(Boolean) ? "both-proxy-offscreen" : proxyValues[0] ? "player0-proxy-offscreen" : proxyValues[1] ? "player1-proxy-offscreen" : "none-proxy-offscreen";
        inc(aggregate.distributions.offscreenProxyConfiguration, proxyConfig);
        const distance = distanceBin(posts[0], posts[1]);
        inc(aggregate.distributions.distanceBins, distance);
        inc(aggregate.distributions.facingRelationship, facingRelationship(posts[0], posts[1]));
        markCoverage(`distance=${distance}`, replayFile);
        const stockDifference = number(posts[0].stocksRemaining) - number(posts[1].stocksRemaining);
        const stockRelationship = stockDifference === 0 ? "stocks-tied" : Math.abs(stockDifference) >= 2 ? "stock-lead-2+" : "stock-lead-1";
        inc(aggregate.distributions.stockRelationship, stockRelationship);
        const percentDifference = Math.abs(number(posts[0].percent) - number(posts[1].percent));
        const percentRelationship = number(posts[0].percent) >= 120 && number(posts[1].percent) >= 120 ? "both-120+" : percentDifference >= 80 ? "percent-gap-80+" : percentDifference >= 40 ? "percent-gap-40-79" : "percent-close";
        inc(aggregate.distributions.percentRelationship, percentRelationship);
        if (offstageConfig === "both-offstage" || proxyConfig !== "none-proxy-offscreen" || distance === "extreme" || stockRelationship === "stock-lead-2+" || percentRelationship === "both-120+") {
          const example = {
            replayFile, frame: frameNumber,
            players: players.map((player, index) => ({ player, character: characterByPlayer.get(player), x: number(posts[index].positionX), y: number(posts[index].positionY), state: posts[index].actionStateId, percent: number(posts[index].percent), stocks: posts[index].stocksRemaining, zone: semanticZone(posts[index]) })),
          };
          if (offstageConfig === "both-offstage") saveExample("both-offstage", example);
          if (proxyConfig !== "none-proxy-offscreen") saveExample(proxyConfig, example);
          if (distance === "extreme") saveExample("extreme-distance", example);
          if (stockRelationship === "stock-lead-2+") saveExample("stock-lead-2+", example);
          if (percentRelationship === "both-120+") saveExample("both-120+", example);
        }
      }
      previousFrame = frame;
    }
    for (const player of players) {
      if (stillStart.get(player) != null) finishStillEpisode(replayFile, player, stillStart.get(player), last);
    }
    if (mutualStart != null) finishStillEpisode(replayFile, "both", mutualStart, last, true);
    aggregate.telemetry.matchesWithRuntimeItemArrays += Number(sawRuntimeItemArray);
    aggregate.items.matchesWithObjects += Number(itemObjects.size > 0);
    aggregate.items.uniqueObjects += itemObjects.size;
    for (const object of itemObjects.values()) {
      const typeId = object.typeId ?? -1;
      const hex = `0x${typeId.toString(16).padStart(2, "0")}`;
      inc(aggregate.items.typeIds, `${hex}:${KNOWN_OBJECT_TYPES[typeId] ?? "unknown"}`);
      if (typeId >= 0 && typeId <= 0x22) aggregate.items.basePickupObjects += 1;
      else aggregate.items.projectileOrArticleObjects += 1;
      aggregate.items.pokeBallObjects += Number(typeId === 0x22);
      aggregate.items.pokemonObjects += Number(typeId >= 0xa0 && typeId <= 0xce);
    }

    const replayComboSummary = { combos: 0, kills: 0, zeroToDeaths: 0, longBeatdowns: 0 };
    for (const combo of stats?.combos ?? []) {
      if (!(combo.moves?.length)) continue;
      const attacker = combo.moves[0].playerIndex ?? combo.lastHitBy;
      const victim = combo.playerIndex;
      const attackerChar = characterByPlayer.get(attacker) ?? "unknown";
      const victimChar = characterByPlayer.get(victim) ?? "unknown";
      const names = combo.moves.map((move) => moveName(move.moveId));
      const sequence = `${attackerChar}:${names.join(">")}`;
      const damage = combo.moves.reduce((sum, move) => sum + number(move.damage), 0);
      const hits = combo.moves.reduce((sum, move) => sum + number(move.hitCount, 1), 0);
      const lastMoveFrame = combo.moves.at(-1).frame;
      const tightDuration = lastMoveFrame - combo.startFrame + 1;
      const statDuration = Number.isFinite(combo.endFrame) ? combo.endFrame - combo.startFrame + 1 : tightDuration;
      aggregate.combos.total += 1;
      aggregate.combos.kills += Number(combo.didKill);
      aggregate.combos.zeroToDeaths += Number(combo.didKill && number(combo.startPercent) <= 1);
      aggregate.combos.totalDamage += damage;
      aggregate.combos.totalHits += hits;
      aggregate.combos.tightDurationFrames += tightDuration;
      aggregate.combos.statDurationFrames += statDuration;
      inc(aggregate.combos.hitBuckets, hits >= 8 ? "8+" : hits >= 5 ? "5-7" : hits >= 3 ? "3-4" : String(hits));
      inc(aggregate.combos.damageBuckets, damage >= 80 ? "80+" : damage >= 50 ? "50-79" : damage >= 25 ? "25-49" : "under-25");
      inc(aggregate.combos.durationBuckets, tightDuration >= 180 ? "3s+" : tightDuration >= 120 ? "2-3s" : tightDuration >= 60 ? "1-2s" : "under-1s");
      inc(aggregate.combos.sequences, sequence);
      markCoverage(`combo-sequence=${sequence}`, replayFile);
      for (const name of new Set(names)) markCoverage(`combo-landed-move=${attackerChar}:${name}`, replayFile);
      aggregate.combos.longBeatdowns += Number(damage >= 50 || hits >= 5 || tightDuration >= 120);
      if (damage >= 50 || hits >= 5 || tightDuration >= 120) markCoverage(`combo=long-beatdown:${attackerChar}->${victimChar}`, replayFile);
      if (combo.didKill) markCoverage(`combo=killing:${attackerChar}->${victimChar}`, replayFile);
      if (combo.didKill && number(combo.startPercent) <= 1) markCoverage(`combo=zero-to-death:${attackerChar}->${victimChar}`, replayFile);
      replayComboSummary.combos += 1;
      replayComboSummary.kills += Number(combo.didKill);
      replayComboSummary.zeroToDeaths += Number(combo.didKill && number(combo.startPercent) <= 1);
      replayComboSummary.longBeatdowns += Number(damage >= 50 || hits >= 5 || tightDuration >= 120);
      for (const move of combo.moves) {
        const name = moveName(move.moveId);
        const key = `${attackerChar}:${name}`;
        inc(aggregate.combos.landedMoves, key, number(move.hitCount, 1));
        if ([MoveId.F_SMASH, MoveId.U_SMASH, MoveId.D_SMASH].includes(move.moveId)) inc(aggregate.combos.landedSmashAttacks, key, number(move.hitCount, 1));
        if ([MoveId.NEUTRAL_SPECIAL, MoveId.F_SPECIAL, MoveId.U_SPECIAL, MoveId.D_SPECIAL].includes(move.moveId)) inc(aggregate.combos.landedSpecials, key, number(move.hitCount, 1));
      }
      if (damage >= 50 || hits >= 5 || tightDuration >= 120 || combo.didKill) {
        saveExample("large-or-killing-combo", { replayFile, attacker, victim, attackerChar, victimChar, startFrame: combo.startFrame, lastMoveFrame, tightDuration, damage, hits, didKill: combo.didKill, startPercent: combo.startPercent, endPercent: combo.endPercent, sequence });
      }
      if (combo.didKill && number(combo.startPercent) <= 1) {
        saveExample("zero-to-death", { replayFile, attacker, victim, attackerChar, victimChar, startFrame: combo.startFrame, lastMoveFrame, damage, hits, sequence });
      }
    }
    for (const conversion of stats?.conversions ?? []) inc(aggregate.combos.openingTypes, conversion.openingType ?? "unknown");
    for (const actionCounts of stats?.actionCounts ?? []) {
      const char = characterByPlayer.get(actionCounts.playerIndex) ?? "unknown";
      aggregate.techniques[char] ??= {};
      addDeep(aggregate.techniques[char], actionCounts);
    }

    aggregate.replaySummaries.push({
      replayFile, slpVersion: version, stage, matchup, firstFrame: first, lastFrame: last,
      seconds: replayWorld60 / 60, worldFrames60: replayWorld60, worldFrames20: replayWorld20,
      itemSpawnBehavior: settings.itemSpawnBehavior, runtimeItemArrayAvailable: sawRuntimeItemArray,
      uniqueRuntimeObjects: itemObjects.size, objectUpdateRows: replayObjectUpdates,
      comboSummary: replayComboSummary, sampledScenarioFlags: replayFlags,
      metadataLastFrame: metadata?.lastFrame ?? null,
      playerSettings: players.map((player) => ({ player, character: characterByPlayer.get(player), startStocks: playerSettings.get(player)?.startStocks, type: playerSettings.get(player)?.type })),
    });
    aggregate.replayFilesAnalyzed += 1;
  } catch (error) {
    aggregate.failures.push({ replayFile, error: error.stack ?? String(error) });
  }
  if ((replayIndex + 1) % 10 === 0) process.stderr.write(`processed ${replayIndex + 1}/${replayFiles.length}\n`);
}

aggregate.coverage = [...coverage.entries()].map(([key, value]) => ({ key, frames20: value.frames20, matches: value.matches.size, players: value.players.size })).sort((a, b) => b.frames20 - a.frames20 || a.key.localeCompare(b.key));
aggregate.examples = examples;
aggregate.summary = {
  seconds: aggregate.frames.world60 / 60,
  hours: aggregate.frames.world60 / 216000,
  itemRulesOffRate: aggregate.settings.itemOffMatches / Math.max(1, aggregate.replayFilesAnalyzed),
  strictStillPlayerFrameRate: aggregate.stationary.strictStillFrames60 / Math.max(1, aggregate.frames.player60),
  mutualStillWorldFrameRate: aggregate.stationary.mutualStillFrames60 / Math.max(1, aggregate.frames.world60),
  damagePlayerFrameRate: aggregate.vulnerability.damageFrames60 / Math.max(1, aggregate.frames.player60),
  comboVictimPlayerFrameRate20: aggregate.vulnerability.comboVictimFrames20 / Math.max(1, aggregate.frames.player20),
  comboAttackerPlayerFrameRate20: aggregate.vulnerability.comboAttackerFrames20 / Math.max(1, aggregate.frames.player20),
  averageComboDamage: aggregate.combos.totalDamage / Math.max(1, aggregate.combos.total),
  averageComboHits: aggregate.combos.totalHits / Math.max(1, aggregate.combos.total),
};

fs.writeFileSync(OUTPUT, `${JSON.stringify(aggregate, null, 2)}\n`);
console.log(JSON.stringify({ output: OUTPUT, analyzed: aggregate.replayFilesAnalyzed, failures: aggregate.failures.length, summary: aggregate.summary, settings: aggregate.settings, items: aggregate.items, combos: { total: aggregate.combos.total, kills: aggregate.combos.kills, zeroToDeaths: aggregate.combos.zeroToDeaths, longBeatdowns: aggregate.combos.longBeatdowns } }, null, 2));
