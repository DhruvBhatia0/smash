# Melee pretraining scenario coverage audit

## Executive result

This 100-replay sample gives deep coverage of **Battlefield Fox vs. Captain Falcon**, but it is not
close to broad Melee coverage:

- 100/100 matches are Battlefield.
- 100/100 are Fox vs. Captain Falcon.
- Random item spawning is off in 100/100.
- 196/200 player slots are human; four are CPU opponents.
- The sample contains 4.124 hours of game time, 890,851 native 60 Hz world frames, and 296,989
  20 Hz world-model frames.
- All 100 files parsed successfully.

Within that narrow domain, the replays are behaviorally rich: neutral, platform movement, combos,
hitstun, recoveries, both players offstage, exact magnifying-glass/offscreen states, stock leads,
high-percent states, standing still, and hundreds of move sequences all occur. The immediate
pretraining problem is not “more random Fox-Falcon”; it is filling missing matchups, stages, items,
modes, and rare mechanics while deliberately retaining the long tail already visible here.

## “Ultimates,” balls, and items

Vanilla Super Smash Bros. Melee has **no Final Smashes, ultimate meter, or Smash Balls**. Those
mechanics were added after Melee. Melee does have Poké Balls, but they are ordinary items that
summon Pokémon; they do not grant an ultimate.

If “ultimate” meant smash attacks, the two players attempted 807 smash attacks in 4.124 hours
(3.26 per minute) and 342 smash-attack hits appeared in Slippi combo records:

| Character | Attempted F/U/D-smashes | Landed hits in combos |
|---|---:|---:|
| Fox | 618 | 295 |
| Captain Falcon | 189 | 47 |

There were 934 landed specials in combo records. Fox shine (`down-b`) accounts for 602. Falcon had
87 landed Raptor Boosts, 71 landed up-specials, and 13 landed down-specials. No Falcon Punch
(`neutral-b`) landed in a recorded combo. Captain Falcon fair landed 336 times, although Slippi's
move ID does not distinguish the strong knee hitbox from weaker fair hitboxes.

Actual item findings:

- Item spawning was disabled by the rules in all 100 matches.
- No base pickup item, Poké Ball, Pokémon, pickup, or item-use event appeared.
- Runtime object telemetry is only available in the 19 SLP 3.9 matches; the other 81 are SLP 2.0.1.
- Eighteen telemetry-capable matches contain 282 objects, all character-created Fox objects:
  158 Fox lasers, 96 Fox blasters, and 28 Fox Illusion objects.

This distinction matters: Slippi's `frame.items` stream contains projectiles and character
articles as well as random items. Counting its 4,246 update rows as “items used” would be wrong;
objects must be deduplicated by `spawnId` and classified by `typeId`.

## Standing still and getting beaten up

Strict standing still means WAIT state, grounded, neutral gameplay buttons/sticks, and less than
0.03 world-unit motion at native 60 Hz.

- 38,998 strict-still player frames: 650 seconds, or 2.19% of player-time.
- 497 still episodes lasting at least 0.25 seconds.
- 126 lasted 0.5–1 second, 32 lasted 1–2 seconds, and four lasted 2–5 seconds.
- The longest strict-still episode was 147 frames (2.45 seconds).
- Both players were strictly still simultaneously for only 205 frames total (3.42 seconds), with
  no simultaneous episode reaching 0.5 seconds.

Getting-beaten-up states are common:

- 228,477 damage/hitstun player frames: 3,808 seconds, or 12.82% of player-time.
- At the 20 Hz model clock, each side is an active-combo victim for 66,814 player frames (11.25%
  of player-time) and an attacker for the same count.
- One player is in standing-idle while the other is in damage/hitstun for 3,735 world frames at
  20 Hz, or 186.75 seconds.
- The sample contains 43,055 knockdown/missed-tech frames, 33,292 tech frames, 14,998 hitlag
  frames, and 7,314 captured/grabbed frames.

These should be separate clip flags. “Standing still in neutral,” “waiting while the opponent is
in hitstun,” and “unable to move because captured/damaged” look superficially inactive but have
very different dynamics.

## Combos and punish situations

Slippi reports 4,252 combo episodes (42.5 per match):

| Bucket | Count |
|---|---:|
| One hit | 2,412 |
| Two hits | 865 |
| Three–four hits | 678 |
| Five–seven hits | 223 |
| Eight-plus hits | 74 |
| Killing combos | 338 |
| Operational zero-to-deaths | 5 |
| “Long beatdowns” | 527 |

For this audit, a long beatdown means at least 50 damage, five hits, or two seconds from combo
start through the final landed move. These are 12.39% of combos. The average combo has 1.96 hits,
15.23 damage, and 0.63 seconds from start through its final move. Slippi's full combo `endFrame`
averages 2.09 seconds because its algorithm includes a 45-frame continuation/reset grace period.

Common multi-move sequences include:

| Sequence | Episodes | Independent matches |
|---|---:|---:|
| Falcon `jab > jab > jab` | 50 | 31 |
| Fox `dair > down-b` | 35 | 26 |
| Falcon `jab > jab` | 32 | 29 |
| Falcon `up-b > up-b` | 30 | 26 |
| Fox `nair > down-b` | 29 | 25 |
| Fox `utilt > bair` | 26 | 21 |
| Falcon `dthrow > fair` | 23 | 19 |
| Falcon `nair > fair` | 21 | 18 |

There are 842 distinct exact move sequences. Of those, 591 occur in only one match and 745 occur
in fewer than five matches. Exact sequence identity therefore has a very long tail even inside one
matchup on one stage. A pretraining clip selector should deliberately protect that tail instead of
sampling uniformly by frame.

## Space, offstage, and offscreen

The 20 Hz Battlefield position classifier covers all 25 coarse `x/y` bins for both characters and
17 semantic zones per character: main-floor left/center/right, each platform, low/mid/high air,
ledges, side/below offstage, under-stage, and dead/respawn.

| Joint spatial state | Share of world frames | Time |
|---|---:|---:|
| Both onstage | 73.29% | 181.4 min |
| Exactly one offstage | 19.93% | 49.3 min |
| Both offstage | 6.78% | 16.8 min |

Offscreen is measured separately and exactly with py-slippi's post-frame
`StateFlags.OFF_SCREEN`, excluding overlapping dead/KO states. This is the game's magnifying-glass
flag, not a coordinate guess.

- At least one alive player is offscreen in 6.30% of 20 Hz world frames (15.60 minutes).
- Both are offscreen in 345 frames: 0.116%, or 17.25 seconds.
- Every sampled replay contains at least one alive offscreen episode.
- There are 1,945 episodes; the median is 24 native frames (0.40 seconds), 204 last at least one
  second, 19 last at least two seconds, and the maximum is 3.18 seconds.
- Extreme player separation occurs in 42,831 20 Hz frames (14.42%).

Both directions and both-player permutations are retained. For example, the exact 20 Hz counts are
8,739 player-0-only offscreen, 9,632 player-1-only offscreen, and 345 both offscreen.

## What the combinatorics look like

The first-pass sparse flagger observed:

| Key family | Distinct observed keys |
|---|---:|
| Character/action-state ID | 327 |
| Semantic position | 34 |
| Character `x/y` grid | 50 |
| Joint action category | 324 |
| Joint semantic zone | 272 |
| Character zone × action | 346 |
| Joint zone × joint action | 10,070 |
| Exact combo sequence | 842 |

For joint zone/action states, 2,868 keys occur in only one match and 5,825 occur in fewer than five
matches. In other words, 57.8% of observed joint situations are already sparse at 100 games. This
is why generating a full Cartesian product is not useful: many combinations are impossible, while
the observed tail itself needs explicit support.

The most common joint situations are exactly what we would expect from this matchup: an aerial
attack hitting an airborne opponent, grounded attacks launching an opponent above center stage,
and one player moving on the ground while the other approaches aerially. The taxonomy still leaves
about 12% of raw player frames in broad `character-special-or-unique` or `other-shared-state`
categories. Before claiming “no unknown states,” those character-specific action IDs need exact
names and rare IDs must remain visible rather than being collapsed.

## Recommended pretraining scenario record

Emit one sparse tag record per 20 Hz transition, while detecting short events at the native 60 Hz
clock:

```text
match:
  stage, matchup, player type, rules/items, SLP telemetry version

per player:
  character + exact action-state ID + semantic action category
  stock bin + percent bin + shield/jumps/invulnerability
  semantic stage zone + coarse x/y bin
  onstage/offstage + exact alive-offscreen flag
  combo role: attacker / victim / neither

joint:
  ordered zone pair + ordered action pair
  distance + facing relationship
  stock lead + percent gap
  both/one/neither offstage and offscreen

episodes:
  strict stillness, hitlag, damage, capture, knockdown, tech
  combo move sequence, hits, damage, kill, opening type
  base item / projectile / stage object and outcome
```

Store only observed keys and count three independent quantities: frames, episodes, and matches.
Treat zero as absent, one–four matches as sparse, five–19 as weakly supported, and 20+ as supported.
For clip selection, save roughly ±2–3 seconds around episodes and greedily choose clips that add
new atomic or selected pairwise keys. Cap common neutral/idle keys so they cannot crowd out rare
recoveries, interactions, and move sequences.

“No unknown states” should mean:

1. No unknown character, stage, action-state, item/object, or protocol-event IDs.
2. Every reachable atomic bucket has a minimum number of independent matches.
3. Selected causal pairwise states—not the impossible full Cartesian product—have minimum support.
4. Every absent family is either collected/generated or explicitly declared outside the model's
   intended domain.

## Major pretraining gaps

This corpus is missing or cannot establish coverage for:

- 24 other playable characters and every other matchup.
- Every stage except Battlefield, including moving platforms and stage hazards.
- Random items, Poké Balls/Pokémon, item pickups/throws/swings/shooting, and item-on rules.
- Teams, free-for-all, doubles-specific camera states, and most CPU behavior.
- Sheik/Zelda transformations, Ice Climbers dual-body states, Kirby copied abilities, Peach
  turnips, Samus charge shot/missiles, Link bombs/boomerangs, and other character articles.
- Shield breaks, sleeps/freezes/buries, reflects/absorbs/counters, tethers, wall/ceiling techs,
  timeouts/sudden death, simultaneous KOs, and other rare mechanics at reliable support levels.

These are absent from this sample, not impossible in Melee. The next dataset-expansion pass should
source or deliberately generate these families before simply adding more random Battlefield
Fox-Falcon games.

## Reproduction and artifacts

The audit used a CPU-only `daytona-large` sandbox and a deterministic, content-deduplicated sample
of 100 SLPs from the pinned `DhruvBhatia0/smash-battlefield-fox` revision. The 1,492 recorded SLP
paths contain only 889 unique replays, so content deduplication is mandatory.

Tracked scripts:

- `download_100.py`: pinned HF tree enumeration, LFS-object deduplication, deterministic sampling,
  and concurrent download.
- `analyze_scenarios.mjs`: Slippi settings, positions, actions, combos, techniques, objects, and
  sparse combination flags.
- `analyze_offscreen.py`: exact raw Slippi magnifying-glass/offscreen state analysis.

Large local results are ignored under `runs/daytona-100/`:

- `scenario-analysis.json`
- `offscreen-analysis.json`
- `sample-manifest.json`

Result SHA-256 values:

- scenario analysis: `5a0a040b38af95300565e00113d594a4e0ed3a0701dc8d89dd32d07be0b29c7c`
- offscreen analysis: `124c2e867c1950f9b075321b6ac6581e90c2f103b9e82758bacaea68ffd84e21`
- sample manifest: `f3c89b6ce6b6a86e30b094e5b00acf4246b47bb4d97015cca61fef28173139a2`

Primary technical references: the
[Slippi replay specification](https://github.com/project-slippi/slippi-wiki/blob/master/SPEC.md)
and the Melee decompilation's
[item-kind enum](https://github.com/doldecomp/melee/blob/master/src/melee/it/forward.h).
