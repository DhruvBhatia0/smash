# SLP Frame-State Binary Schema

The semantic fast path writes one fixed-width row for every occupied player/follower frame record
that has both Slippi pre-frame and post-frame data. Empty controller ports are counted separately
and are not output rows.

Current format: `SLPFRAMESTATEv2`

Record size: `132` bytes, little-endian.

| Offset | Type | Field |
| ---: | --- | --- |
| 0 | int32 | frameNumber |
| 4 | uint8 | player slot/index |
| 5 | uint8 | isFollower |
| 6 | uint8 | post.isAirborne |
| 7 | uint8 | post.internalCharacterId |
| 8 | uint32 | pre.seed |
| 12 | uint16 | pre.actionStateId |
| 14 | uint16 | post.actionStateId |
| 16 | float32 | pre.positionX |
| 20 | float32 | pre.positionY |
| 24 | float32 | pre.facingDirection |
| 28 | float32 | pre.joystickX |
| 32 | float32 | pre.joystickY |
| 36 | float32 | pre.cStickX |
| 40 | float32 | pre.cStickY |
| 44 | float32 | pre.trigger |
| 48 | uint32 | pre.buttons |
| 52 | uint32 | pre.physicalButtons |
| 56 | float32 | pre.physicalLTrigger |
| 60 | float32 | pre.physicalRTrigger |
| 64 | float32 | pre.rawJoystickX |
| 68 | float32 | pre.percent |
| 72 | float32 | post.positionX |
| 76 | float32 | post.positionY |
| 80 | float32 | post.facingDirection |
| 84 | float32 | post.percent |
| 88 | float32 | post.shieldSize |
| 92 | uint8 | post.lastAttackLanded |
| 93 | uint8 | post.currentComboCount |
| 94 | uint8 | post.lastHitBy |
| 95 | uint8 | post.stocksRemaining |
| 96 | float32 | post.actionStateCounter |
| 100 | float32 | post.miscActionState |
| 104 | uint16 | post.lastGroundId |
| 106 | uint8 | post.jumpsRemaining |
| 107 | uint8 | post.lCancelStatus |
| 108 | uint8 | post.hurtboxCollisionState |
| 109 | uint8 | reserved |
| 110 | uint16 | post.instanceHitBy |
| 112 | uint16 | post.instanceId |
| 116 | float32 | post.hitlagRemaining |
| 120 | uint32 | post.animationIndex |
| 124 | float32 | post.selfInducedSpeeds.x |
| 128 | float32 | post.selfInducedSpeeds.y |

`v2` differs from the first experiment format by widening `post.instanceHitBy` to `uint16`; public
Slippi edge-case replays include stage hazard/platform hit instance IDs above `255`.
