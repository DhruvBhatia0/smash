# Fox–Captain Falcon Battlefield video dataset

Status: production complete and audited on 2026-07-13.

This is the handoff for the replay/video corpus: where it lives, which copy is
authoritative, how samples and frames line up, what the metadata means, and how
to read or regenerate it. The production pipeline did not mirror the corpus to
this repository or the laptop. Google Drive is durable storage; Daytona was
CPU-only transient compute and its production sandboxes were deleted after the
run.

## Authoritative data

Use this output and no other recording folder:

- Google Drive: [final dataset folder](https://drive.google.com/drive/folders/16GUZtblgZcMuecHsQPZyclhpZk3ozWCL)
- Batches: [final `batches/` folder](https://drive.google.com/drive/folders/1CKKsAdaWw41QHAdS1I9feTO4bxxzBCfw)
- rclone: `smash-drive:hal-fox-captain-falcon-battlefield/recordings-252x208-20fps-slippi-pts-v3`
- Payload path below it: `batches/`

The source root also has `.smoke/`, `daytona-dry-runs/`,
`recordings-252x208-20fps/`, and
`recordings-252x208-20fps-slippi-pts-v3-orphan-quarantine/`. The authoritative
target itself has a `legacy-quarantine/` sibling beside `batches/`. These are
smoke tests, an older timeline, or quarantined uploads. Only the final target's
`batches/` directory is dataset data.

Final inventory:

| Item | Count or size |
|---|---:|
| Indexed replay references | 10,327 |
| MP4 videos | 10,319 |
| Explicit `no_playable_frames` skips | 8 |
| Video frames | 31,359,489 |
| Video duration at 20 FPS | 435 h 32 m 54.45 s |
| Committed batch pairs | 111 archives + 111 manifests |
| Final Drive bytes | 147,359,770,145 B (147.360 GB / 137.239 GiB) |

All 10,327 references are accounted for. There are no pending or unknown
samples.

## Source data and provenance

The filtered source corpus is in the [Drive source root](https://drive.google.com/drive/folders/1jDbGzsyZntuJDU319sBBfHgLvdd_67Ci),
also addressable as:

```text
smash-drive:hal-fox-captain-falcon-battlefield
```

It contains six `shard-NN.tar.zst` files, six
`shard-NN.tar.zst.slp-index.jsonl` sidecars, and six `report-NN.json` reports.
The six compressed source shards occupy 6,861,825,777 B (6.862 GB / 6.391
GiB). Their selected replay payloads total 33,567,151,677 B before shard
compression.

| Shard | Compressed bytes | Selected replays | Original input |
|---|---:|---:|---|
| 01 | 911,396,679 | 1,376 | [`ranked-anonymized-1-116248.7z`](https://drive.google.com/file/d/1pFjgh1dapX34s0T-Q1TC7JUO-qjYbQZf/view) |
| 02 | 1,132,033,861 | 1,704 | [`ranked-anonymized-2-151807.7z`](https://drive.google.com/file/d/1jEIzvhpV3778J2s2-Np9vCVqSLf9lZnk/view) |
| 03 | 1,201,007,715 | 1,829 | [`ranked-anonymized-3-128787.zip`](https://drive.google.com/file/d/1glzlkAPxHC58oXZljJXQV8dsTBKmlhkE/view) |
| 04 | 1,077,261,459 | 1,619 | [`ranked-anonymized-4-148358.zip`](https://drive.google.com/file/d/1qdIZUW4Er_Vu6rD3-VUvyak3lKa1KxVk/view) |
| 05 | 1,021,324,601 | 1,540 | [`ranked-anonymized-5-133261.zip`](https://drive.google.com/file/d/1Hqmj6C8g1BzuRAIqOrQcMDL0MX4GtffE/view) |
| 06 | 1,518,801,462 | 2,259 | [`ranked-anonymized-6-171694.zip`](https://drive.google.com/file/d/1g8yZ-Q4ldyhDEmXLSPBoWxywJRMRVGc3/view) |
| **Total** | **6,861,825,777** | **10,327** | **616.603 GB of original archives** |

The original archives contained 850,106 candidate replay members. Of those,
850,002 parsed, 104 failed parsing, and 10,327 matched the filter. Shard 05's
filename says 133,261, but its report records 133,212 actual members; use the
report rather than the filename as ground truth.

A replay was selected only when all of these were true:

- stage ID `31` (Battlefield);
- exactly two non-empty players; and
- the unordered character set was exactly Fox plus Captain Falcon.

Port, player order, costume, winner, and filename-derived rank can vary. Inside
a source shard, a selected replay is named
`files/<first-16-hex-of-SHA1(member-path-string)>__<original-basename>.slp`.
That prefix prevents naming collisions; it is not a content hash. The sidecar
index contains `archive` and `member`; together they form the canonical
reference:

```text
shard-01.tar.zst::files/<name>.slp
```

Each shard archive also carries extraction provenance: `manifest.jsonl` rows
with `source`, `output`, `bytes`, `stage`, and `characters`; `report.json` with
scan/selection totals; and `paths.txt` with selected original member paths.
The top-level `.slp-index.jsonl` sidecars are the intentionally small job
indexes used by the video pipeline.

## Output layout and commit rule

Each committed batch has two files:

```text
batches/batch-<20-hex-key>.tar.zst
batches/batch-<20-hex-key>.manifest.jsonl
```

The archive contains at most 100 results and normally stops before adding a
result that would take input plus result bytes over 2 GiB before compression.
A single result larger than 2 GiB is still allowed. Its member layout is:

```text
slp_with_video/<sample>/input.slp
slp_with_video/<sample>/metadata.json
slp_with_video/<sample>/video.mp4       # absent only for an explicit skip
```

The matching manifest is uploaded last and is the commit record. Treat an
archive as committed only when both the positive-size archive and its
positive-size manifest exist. An archive without its manifest is not dataset
data. The batch key is the first 20 hex characters of the SHA-256 of the
newline-joined source references in sample order.

A normal manifest row is:

```json
{"schemaVersion":1,"status":"complete","sample":6533,"sourceReference":"shard-05.tar.zst::files/015bb27b397dbe67__example.slp","sourceSha256":"<64 hex>","archive":"hal-fox-captain-falcon-battlefield/recordings-252x208-20fps-slippi-pts-v3/batches/batch-<key>.tar.zst"}
```

An explicit skip adds:

```json
{"artifact":"skipped","skipReason":"no_playable_frames"}
```

Normal rows do not have an `artifact` field. `sample` is a deterministic,
zero-based position after lexicographically sorting every canonical source
reference. It is convenient within this frozen corpus but is not an intrinsic
replay ID. Use `sourceReference` as the durable identity and join key.
`sourceSha256` identifies the exact retained `input.slp`.

## Video and frame interpretation

Every non-skipped video satisfies this contract:

| Property | Value |
|---|---|
| Container / codec | MP4 / H.264 (`libx264`) |
| Pixel format | `yuv420p` |
| Dimensions | 252 × 208 |
| Frame rate | constant `20/1` FPS |
| Streams | one video stream, no audio |
| First packet timestamp | 0 |
| Renderer | CPU only |

The replay timeline is 60 FPS. The video selects every third source frame,
including both endpoints when the endpoint lands on the sampling grid. For
video frame index `i` starting at zero:

```text
slippi_frame(i) = firstSelectedSlpFrame + 3 * i
video_time(i)   = i / 20 seconds
```

Equivalently:

```text
lastSelected = first + floor((lastSource - first) / 3) * 3
videoFrames  = (lastSelected - first) / 3 + 1
```

`firstSelectedSlpFrame` is normally `-39`. Use the values in each sample's
metadata rather than assuming it. The exact stored-video duration is always
`video.frames / 20`; for corpus totals, sum `video.frames` first and divide
once. Do not use `match.duration.seconds` as MP4 duration. It is the semantic
60 Hz playable duration and uses different endpoint semantics. Across this
corpus its sum is 346 seconds shorter than the stored-video total.

Important `metadata.json` fields:

| Field | Interpretation |
|---|---|
| `source.reference` | Same durable replay identity as manifest `sourceReference`. |
| `file.bytes`, `file.sha256` | Size and SHA-256 of the retained `input.slp`. |
| `match.stage`, `match.rules` | Parsed stage and rules. Stage should be Battlefield / ID 31. |
| `match.frames` | SLP frame bounds, stored count, 60 FPS rate, and playable bound. |
| `match.duration` | Replay-semantic duration, not exact MP4 duration. |
| `match.end`, `match.winner` | Parsed end method and game-end placement; unresolved/no-contest games can have no winner. |
| `players` | Character, port, identity when present, starting/final state, aggregate stats, and action counts. |
| `players[].rank` | Label inferred from the source filename only; `mmr` is null. It is not verified rank. |
| `aggregate` | Replay-level stock, conversion, and combo counts. |
| `video.frames`, `video.frameRate` | Authoritative video length and rate. |
| `video.firstSelectedSlpFrame` | SLP frame represented by video frame zero. |
| `video.lastSourceSlpFrame` | Authoritative render endpoint; may be before `match.frames.last` only for recorded tail recovery. |
| `video.lastSelectedSlpFrame` | Last SLP frame actually present in the MP4. |
| `video.sourceFrameStep` | `3`, the 60-to-20 FPS stride. |
| `video.croppedTailSourceFrames` | Zero to two unsampled source frames after the final selected frame. |
| `video.renderSeconds` | CPU wall-clock conversion time, not content duration. |
| `video.gameplaySeconds` | Exact stored duration, equal to `video.frames / 20`. |
| `video.inputBytes` | Intermediate raw AVI size, not input SLP size. |
| `video.outputBytes` | MP4 byte size. |
| `video.cpuOnly` | `true` for this corpus. |
| `availability` | Explicit notes about unavailable true MMR and wall-clock time. |

The timestamp, display name, connect code, and other identity fields can be
null because an SLP may not contain them. This dataset does not contain a
separate per-frame game-state table: it contains pixels, the original replay,
and replay-level metadata. Parse the embedded `input.slp` when training needs
controller or state features, apply `replayNormalization` first when present,
then align source frame `firstSelectedSlpFrame + 3 * i` with video frame `i`.

Some source files contain concatenated games. In those cases the renderer uses
the first complete game and records `replayNormalization` with policy
`first-complete-game-v1`, selected segment/frame bounds, and the normalized
render-input hash. The archive still retains the original, unmodified
`input.slp`; `file.sha256` and manifest `sourceSha256` identify that original.
State extraction for one of these records must apply the recorded segment
selection before aligning it to the video.

## Exceptional records

Eight source files have no playable frame interval. Their archive entry has
`input.slp` plus a minimal `metadata.json` containing only
`{"skipReason":"no_playable_frames"}` and no MP4.

| Sample | Source reference |
|---:|---|
| 2035 | `shard-02.tar.zst::files/661d4696f4fc15d6__diamond-platinum-e716493849200da0fa5da992.slp` |
| 2040 | `shard-02.tar.zst::files/67345e5d5615ed8f__diamond-platinum-8e2c5428db27e9105e13e65e.slp` |
| 2053 | `shard-02.tar.zst::files/698846dfba9eeb65__platinum-platinum-29042898cc5078c54cd0da6e.slp` |
| 2811 | `shard-02.tar.zst::files/d71a8f33b7b2862a__diamond-platinum-345f3c80658cb6931a48bda4.slp` |
| 2869 | `shard-02.tar.zst::files/e0b1ce7617dd5578__platinum-platinum-52d8579beac1a29342471365.slp` |
| 6548 | `shard-05.tar.zst::files/04308a65bc667fce__diamond-platinum-083d8d2b399bde34ea19a0f0.slp` |
| 7808 | `shard-05.tar.zst::files/d51b8c9ec5dbc4f5__master-diamond-beb87b38852076e42a3a24ea.slp` |
| 10054 | `shard-06.tar.zst::files/e1b21c555c08881a__diamond-platinum-32b7db289905d92a6b445a17.slp` |

Sample 10317 has a separately recorded stable-tail recovery. Its source
declared frame 9204, but Dolphin reproducibly stopped progressing after frame
7200. Policy `stable-render-prefix-v1` accepted the independently validated
stable prefix after a 60-second stall and records 2,004 discarded source
frames. The resulting MP4 has 2,414 frames and lasts 120.700 seconds. Do not
silently compare its video endpoint to the original declared endpoint; inspect
`renderTailRecovery`.

## Reading the dataset efficiently

Set paths once:

```bash
export SMASH_REMOTE=smash-drive
export SMASH_RCLONE_CONFIG="$HOME/.config/rclone/rclone.conf"
export SMASH_SOURCE=hal-fox-captain-falcon-battlefield
export SMASH_TARGET="$SMASH_SOURCE/recordings-252x208-20fps-slippi-pts-v3"
```

List without downloading payloads:

```bash
rclone lsf "$SMASH_REMOTE:$SMASH_TARGET/batches" \
  --files-only --config "$SMASH_RCLONE_CONFIG"
```

The manifests are small enough to copy locally and are the fastest way to
build an index:

```bash
rclone copy "$SMASH_REMOTE:$SMASH_TARGET/batches" ./manifests \
  --include '*.manifest.jsonl' --config "$SMASH_RCLONE_CONFIG"
```

Inspect an archive without writing the archive to disk:

```bash
export BATCH=batch-<key>
rclone cat "$SMASH_REMOTE:$SMASH_TARGET/batches/$BATCH.tar.zst" \
  --config "$SMASH_RCLONE_CONFIG" | zstd -dc | tar -tf -
```

Stream one member to stdout or a destination file:

```bash
export SAMPLE=6533
rclone cat "$SMASH_REMOTE:$SMASH_TARGET/batches/$BATCH.tar.zst" \
  --config "$SMASH_RCLONE_CONFIG" | zstd -dc | \
  tar -xOf - "slp_with_video/$SAMPLE/metadata.json"
```

`tar.zst` is sequential, not random-access. A one-member stream can still
transfer bytes up to that member. On training infrastructure, copy each batch
once to fast scratch, iterate all of its samples, then delete it; do not reopen
the same Drive object once per sample. Do not bulk-stage this corpus on the
weak laptop.

An individual manifest marks its batch committed and claims the references in
that batch. Logical corpus coverage requires deduplicating the complete
manifest set and comparing it with all 10,327 source-index references. The
manifests do not include archive size/hash or per-video hash. Strict byte
validation should stream each archive, compare every `input.slp` SHA-256 with
`sourceSha256`, ffprobe every MP4 against the video contract, compare
`video.outputBytes`, and sum `video.frames`.

## CPU-only regeneration and resume

The production corpus is complete; regeneration is only needed after an
intentional format change. The entry point is:

```bash
python3 -m core.data.frame-recorder.runner
```

Required external state is an authenticated Daytona CLI, an rclone config with
access to `smash-drive`, a local Melee v1.02 ISO, and the active Daytona
snapshots. No credential or ISO is committed here. The connector creates the
10 GB `smash-frame-assets-v1` volume when absent and checksum-seeds the ISO.
The default production shape is 23 GPU-free workers with 4 render processes
each, a GPU-free 2-vCPU coordinator, and one serial rate-limited Drive upload
stream. The code rejects any snapshot or sandbox reporting a nonzero GPU count.
The validated defaults are worker snapshot `smash-cpu-renderer-e7711b1-v3`,
coordinator snapshot `smash-cpu-renderer-e7711b1-v3-2cpu-repair`, and asset-seed
snapshot `smash-cpu-renderer-e7711b1-1cpu-v1`.

Core settings:

```bash
export SMASH_STORAGE_PROVIDER=gdrive
export SMASH_GDRIVE_CONFIG="$HOME/.config/rclone/rclone.conf"
export SMASH_GDRIVE_REMOTE=smash-drive
export SMASH_GDRIVE_ROOT=hal-fox-captain-falcon-battlefield
export SMASH_GDRIVE_RECORDING_DIR="$SMASH_GDRIVE_ROOT/recordings-252x208-20fps-slippi-pts-v3"
export SMASH_MELEE_ISO=/path/to/Super-Smash-Bros-Melee-v1.02.iso
export SMASH_WORKER_COUNT=23
export SMASH_PROCESSES_PER_SANDBOX=4
```

`SMASH_SAMPLE_LIMIT` bounds a smoke run. Snapshot, region, and volume overrides
are defined in `core/data/frame-recorder/daytona_connector.py`. The validated
encoding defaults are x264 `veryfast` at CRF 18. Changing encoding settings
changes artifact bytes and requires a new target directory/version.

Resume is manifest-driven. A rerun scans committed manifest/archive pairs,
removes their source references from the work set, and processes only missing
references. The coordinator streams each source shard once; workers never
download sources independently. Result archives are bounded and manifests are
still written last, so interruption cannot make a partial upload look
committed. Normal cleanup deletes owned sandboxes and the run queue while
retaining the shared ISO volume. Do not set `SMASH_KEEP_DAYTONA_RESOURCES=1`
for routine production runs.

As of this audit, the configured `smash-drive` remote uses rclone's shared
Google Drive OAuth client ID. rclone warns that this client is being retired
during 2026. Configure a private Drive client ID before any future large run to
avoid interruption or shared-project rate limits.

## Repository references

- Selection/extraction: `experiments_dump/hal_matchup_extract/process_archive.py`
- Original Drive inputs: `experiments_dump/hal_matchup_extract/run_on_runpod.py`
- CPU renderer and frame mapping: `core/data/frame-recorder/replay_renderer.py`
- Daytona/Drive orchestration: `core/data/frame-recorder/daytona_connector.py`
- Production measurements and audit summary: `experiments_dump/fast_replay_probe/daytona-cpu-findings.md`
