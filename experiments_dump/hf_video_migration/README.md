# Hugging Face video migration

This migration is deliberately split into two commands:

1. `provision` downloads the complete pinned dataset revision, converts every raw AVI to H.264,
   verifies frame counts, and prepares source/H.264 spotchecks.
2. `cutover` is accepted only after the prepared report passes. It deletes the remote AVIs,
   super-squashes the old large-file history, uploads the MP4s, and verifies all remote pairs.

The RunPod host is retained until `cleanup` sees a verified `complete` report. The current dataset
requires about 15 TB of temporary disk because its 2,741 AVIs occupy 12.78 TB.

RunPod currently caps CPU local disks at 320-480 GB and standard network volumes at 4 TB. The
full-snapshot path therefore requires a support-enabled volume larger than 14.09 TB. Provisioning
checks the mounted capacity and immediately deletes any silently under-allocated pod.

```bash
python3 experiments_dump/hf_video_migration/run_on_runpod.py provision
python3 experiments_dump/hf_video_migration/run_on_runpod.py status
python3 experiments_dump/hf_video_migration/run_on_runpod.py fetch
python3 experiments_dump/hf_video_migration/run_on_runpod.py cutover
python3 experiments_dump/hf_video_migration/run_on_runpod.py cleanup
```
