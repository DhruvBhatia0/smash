import assert from "node:assert/strict";
import test from "node:test";

import { binTicks, canonicalJson, flattenValidMicroActions } from "./binning.mjs";

function tick(sourceIndex, buttons = 0) {
  return {
    sourceIndex,
    sourceFrame: 100 + sourceIndex,
    videoIndex: 20 + sourceIndex,
    state: { exact: `state-${sourceIndex}` },
    actionFromPrevious: {
      sourceFrame: 100 + sourceIndex,
      players: [{ playerIndex: 0, isFollower: false, buttons, physicalButtons: buttons }],
    },
  };
}

test("divisible trajectory preserves endpoints and ordered actions exactly", () => {
  const ticks = [tick(0), tick(1, 1), tick(2, 2), tick(3, 4), tick(4, 8), tick(5, 16), tick(6, 32)];
  const result = binTicks(ticks);

  assert.deepEqual(result.manifest.selectedSourceIndices, [0, 3, 6]);
  assert.equal(result.manifest.strictCfrEndpointCompatible, true);
  assert.equal(result.manifest.partialTransitionCount, 0);
  assert.deepEqual(result.transitions[0].actionSourceFrames, [101, 102, 103]);
  assert.equal(
    canonicalJson(flattenValidMicroActions(result.transitions)),
    canonicalJson(ticks.slice(1).map((entry) => entry.actionFromPrevious)),
  );
  assert.equal(result.observations[0].state.exact, "state-0");
  assert.equal(result.observations.at(-1).state.exact, "state-6");
});

for (const subslot of [0, 1, 2]) {
  test(`one-frame pulse survives in action subslot ${subslot}`, () => {
    const ticks = [tick(0), tick(1), tick(2), tick(3)];
    ticks[subslot + 1] = tick(subslot + 1, 0x100);
    const transition = binTicks(ticks).transitions[0];

    assert.equal(transition.microActions[subslot].players[0].physicalButtons, 0x100);
    assert.deepEqual(transition.validActionMask, [true, true, true]);
  });
}

test("ordered chunks distinguish action streams that button OR would merge", () => {
  const leftThenRight = binTicks([tick(0), tick(1, 1), tick(2, 2), tick(3)]).transitions[0];
  const rightThenLeft = binTicks([tick(0), tick(1, 2), tick(2, 1), tick(3)]).transitions[0];
  const orMask = (transition) =>
    transition.microActions.reduce(
      (combined, action) => combined | action.players[0].physicalButtons,
      0,
    );

  assert.equal(orMask(leftThenRight), orMask(rightThenLeft));
  assert.notEqual(canonicalJson(leftThenRight.microActions), canonicalJson(rightThenLeft.microActions));
});

test("partial final transition preserves true endpoint with an explicit mask", () => {
  const result = binTicks([tick(0), tick(1, 1), tick(2, 2), tick(3, 4), tick(4, 8), tick(5, 16)]);
  const tail = result.transitions.at(-1);

  assert.deepEqual(result.manifest.selectedSourceIndices, [0, 3, 5]);
  assert.equal(result.manifest.strictCfrEndpointCompatible, false);
  assert.equal(tail.sourceSteps, 2);
  assert.equal(tail.isPartial, true);
  assert.deepEqual(tail.validActionMask, [true, true, false]);
  assert.equal(tail.paddedMicroActions.length, 3);
  assert.equal(result.observations.at(-1).state.exact, "state-5");
});

test("rejects gaps and non-integer FPS ratios", () => {
  assert.throws(() => binTicks([tick(0), { ...tick(1), sourceFrame: 103 }]), /contiguous/);
  assert.throws(() => binTicks([tick(0), tick(1)], { sourceFps: 50, targetFps: 20 }), /integer FPS ratio/);
  assert.throws(() => binTicks([tick(0), tick(1)], { sourceFps: 60, targetFps: 30 }), /v1 requires/);
});
