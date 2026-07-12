import crypto from "node:crypto";

export const FORMAT = "ACTION_PRESERVING_20FPS_V1";
export const BUTTON_FIELDS = ["buttons", "physicalButtons"];

function normalized(value) {
  if (value === null || typeof value === "boolean" || typeof value === "string") {
    return value;
  }
  if (typeof value === "number") {
    if (!Number.isFinite(value)) {
      throw new TypeError(`Cannot hash non-finite number: ${value}`);
    }
    return value;
  }
  if (Array.isArray(value)) {
    return value.map(normalized);
  }
  if (typeof value === "object" && value !== undefined) {
    return Object.fromEntries(
      Object.keys(value)
        .sort()
        .map((key) => [key, normalized(value[key])]),
    );
  }
  throw new TypeError(`Unsupported canonical JSON value: ${typeof value}`);
}

export function canonicalJson(value) {
  return JSON.stringify(normalized(value));
}

export function sha256Json(value) {
  return crypto.createHash("sha256").update(canonicalJson(value)).digest("hex");
}

function asUint32(value) {
  const number = Number(value ?? 0);
  if (!Number.isSafeInteger(number)) {
    throw new TypeError(`Button mask must be a safe integer, got ${value}`);
  }
  return BigInt.asUintN(32, BigInt(number));
}

function edgeMasks(previous, current) {
  const before = asUint32(previous);
  const after = asUint32(current);
  return {
    pressed: Number(BigInt.asUintN(32, after & ~before)),
    released: Number(BigInt.asUintN(32, before & ~after)),
  };
}

function playerKey(player) {
  return `${player.playerIndex ?? player.slot ?? 0}:${player.isFollower ? 1 : 0}`;
}

export function deriveButtonEdges(previousAction, currentAction) {
  const previousPlayers = new Map(
    (previousAction?.players ?? []).map((player) => [playerKey(player), player]),
  );
  return (currentAction?.players ?? []).map((player) => {
    const previous = previousPlayers.get(playerKey(player)) ?? {};
    return {
      playerIndex: player.playerIndex ?? player.slot ?? 0,
      isFollower: Boolean(player.isFollower),
      fields: Object.fromEntries(
        BUTTON_FIELDS.map((field) => [field, edgeMasks(previous[field], player[field])]),
      ),
    };
  });
}

function validateTicks(ticks) {
  if (!Array.isArray(ticks) || ticks.length === 0) {
    throw new TypeError("ticks must contain at least one observation");
  }
  ticks.forEach((tick, index) => {
    if (tick.sourceIndex !== index) {
      throw new Error(`tick ${index} has sourceIndex=${tick.sourceIndex}; expected ${index}`);
    }
    if (!Number.isInteger(tick.sourceFrame)) {
      throw new TypeError(`tick ${index} has invalid sourceFrame`);
    }
    if (index > 0 && tick.sourceFrame !== ticks[index - 1].sourceFrame + 1) {
      throw new Error(
        `source frames must be contiguous: ${ticks[index - 1].sourceFrame} -> ${tick.sourceFrame}`,
      );
    }
    if (!("state" in tick)) {
      throw new Error(`tick ${index} is missing state`);
    }
    if (!("actionFromPrevious" in tick)) {
      throw new Error(`tick ${index} is missing actionFromPrevious`);
    }
  });
}

function integerRatio(sourceFps, targetFps) {
  if (!(sourceFps > 0) || !(targetFps > 0)) {
    throw new RangeError("sourceFps and targetFps must be positive");
  }
  const ratio = sourceFps / targetFps;
  if (!Number.isInteger(ratio)) {
    throw new RangeError(
      `v1 requires an integer FPS ratio; got ${sourceFps}/${targetFps}=${ratio}`,
    );
  }
  return ratio;
}

export function flattenValidMicroActions(transitions) {
  return transitions.flatMap((transition) => transition.microActions);
}

export function binTicks(ticks, { sourceFps = 60, targetFps = 20 } = {}) {
  validateTicks(ticks);
  const ratio = integerRatio(sourceFps, targetFps);
  if (sourceFps !== 60 || targetFps !== 20) {
    throw new RangeError(`v1 requires sourceFps=60 and targetFps=20; got ${sourceFps}/${targetFps}`);
  }
  const lastSourceIndex = ticks.length - 1;
  const anchorIndices = [];
  for (let index = 0; index <= lastSourceIndex; index += ratio) {
    anchorIndices.push(index);
  }
  if (anchorIndices.at(-1) !== lastSourceIndex) {
    anchorIndices.push(lastSourceIndex);
  }

  const observations = anchorIndices.map((sourceIndex, observationIndex) => {
    const tick = ticks[sourceIndex];
    return {
      observationIndex,
      sourceIndex,
      sourceFrame: tick.sourceFrame,
      videoIndex: tick.videoIndex ?? null,
      timestampSeconds: sourceIndex / sourceFps,
      stateSha256: sha256Json(tick.state),
      state: tick.state,
    };
  });

  const transitions = [];
  for (let transitionIndex = 0; transitionIndex + 1 < anchorIndices.length; transitionIndex += 1) {
    const startSourceIndex = anchorIndices[transitionIndex];
    const endSourceIndex = anchorIndices[transitionIndex + 1];
    const sourceSteps = endSourceIndex - startSourceIndex;
    const sourceTicks = ticks.slice(startSourceIndex + 1, endSourceIndex + 1);
    const microActions = sourceTicks.map((tick) => tick.actionFromPrevious);
    const actionSourceFrames = sourceTicks.map((tick) => tick.sourceFrame);
    const actionVideoIndices = sourceTicks.map((tick) => tick.videoIndex ?? null);

    let previousAction = ticks[startSourceIndex].actionFromPrevious;
    const buttonEdges = microActions.map((action) => {
      const edges = deriveButtonEdges(previousAction, action);
      previousAction = action;
      return edges;
    });

    const paddedMicroActions = [...microActions];
    while (paddedMicroActions.length < ratio) {
      paddedMicroActions.push(microActions.at(-1));
    }

    transitions.push({
      transitionIndex,
      startObservationIndex: transitionIndex,
      endObservationIndex: transitionIndex + 1,
      startSourceIndex,
      endSourceIndex,
      startSourceFrame: ticks[startSourceIndex].sourceFrame,
      endSourceFrame: ticks[endSourceIndex].sourceFrame,
      sourceSteps,
      durationSeconds: sourceSteps / sourceFps,
      isPartial: sourceSteps !== ratio,
      actionBefore: ticks[startSourceIndex].actionFromPrevious,
      actionSourceFrames,
      actionVideoIndices,
      microActions,
      paddedMicroActions,
      validActionMask: Array.from({ length: ratio }, (_, index) => index < sourceSteps),
      buttonEdges,
    });
  }

  const sourceActions = ticks.slice(1).map((tick) => tick.actionFromPrevious);
  const reconstructedActions = flattenValidMicroActions(transitions);
  const sourceActionSha256 = sha256Json(sourceActions);
  const reconstructedActionSha256 = sha256Json(reconstructedActions);
  if (canonicalJson(sourceActions) !== canonicalJson(reconstructedActions)) {
    throw new Error("Binned micro-actions do not reconstruct the source action stream");
  }

  const firstStateSha256 = sha256Json(ticks[0].state);
  const lastStateSha256 = sha256Json(ticks.at(-1).state);
  if (
    observations[0].stateSha256 !== firstStateSha256 ||
    observations.at(-1).stateSha256 !== lastStateSha256
  ) {
    throw new Error("Binned observations do not preserve source endpoints");
  }

  const partialTransitions = transitions.filter((transition) => transition.isPartial);
  const manifest = {
    format: FORMAT,
    sourceFps,
    targetFps,
    sourceStepsPerFullTransition: ratio,
    sourceObservationCount: ticks.length,
    sourceTransitionCount: Math.max(0, ticks.length - 1),
    outputObservationCount: observations.length,
    outputTransitionCount: transitions.length,
    fullTransitionCount: transitions.length - partialTransitions.length,
    partialTransitionCount: partialTransitions.length,
    finalTransitionSourceSteps: transitions.at(-1)?.sourceSteps ?? 0,
    strictCfrEndpointCompatible: (ticks.length - 1) % ratio === 0,
    selectedSourceIndices: anchorIndices,
    selectedSourceFrames: observations.map((observation) => observation.sourceFrame),
    selectedVideoIndices: observations.map((observation) => observation.videoIndex),
    firstStateSha256,
    lastStateSha256,
    sourceActionSha256,
    reconstructedActionSha256,
    actionRoundTripExact: sourceActionSha256 === reconstructedActionSha256,
    padding: {
      strategy: "repeat-last-with-explicit-mask",
      paddedWidth: ratio,
      warning: "Only entries whose validActionMask value is true are semantic actions.",
    },
  };

  return { manifest, observations, transitions };
}
