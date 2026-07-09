from __future__ import annotations

import os
import re
import tempfile
from dataclasses import dataclass
from enum import Enum, IntEnum
from pathlib import PurePosixPath


class Map(IntEnum):
    UNKNOWN = -1
    FOUNTAIN_OF_DREAMS = 2
    POKEMON_STADIUM = 3
    PRINCESS_PEACHS_CASTLE = 4
    KONGO_JUNGLE = 5
    BRINSTAR = 6
    CORNERIA = 7
    YOSHIS_STORY = 8
    ONETT = 9
    MUTE_CITY = 10
    RAINBOW_CRUISE = 11
    JUNGLE_JAPES = 12
    GREAT_BAY = 13
    HYRULE_TEMPLE = 14
    BRINSTAR_DEPTHS = 15
    YOSHIS_ISLAND = 16
    GREEN_GREENS = 17
    FOURSIDE = 18
    MUSHROOM_KINGDOM_I = 19
    MUSHROOM_KINGDOM_II = 20
    VENOM = 22
    POKE_FLOATS = 23
    BIG_BLUE = 24
    ICICLE_MOUNTAIN = 25
    ICETOP = 26
    FLAT_ZONE = 27
    DREAM_LAND_N64 = 28
    YOSHIS_ISLAND_N64 = 29
    KONGO_JUNGLE_N64 = 30
    BATTLEFIELD = 31
    FINAL_DESTINATION = 32
    TARGET_TEST_MARIO = 33
    TARGET_TEST_CAPTAIN_FALCON = 34
    TARGET_TEST_YOUNG_LINK = 35
    TARGET_TEST_DONKEY_KONG = 36
    TARGET_TEST_DR_MARIO = 37
    TARGET_TEST_FALCO = 38
    TARGET_TEST_FOX = 39
    TARGET_TEST_ICE_CLIMBERS = 40
    TARGET_TEST_KIRBY = 41
    TARGET_TEST_BOWSER = 42
    TARGET_TEST_LINK = 43
    TARGET_TEST_LUIGI = 44
    TARGET_TEST_MARTH = 45
    TARGET_TEST_MEWTWO = 46
    TARGET_TEST_NESS = 47
    TARGET_TEST_PEACH = 48
    TARGET_TEST_PICHU = 49
    TARGET_TEST_PIKACHU = 50
    TARGET_TEST_JIGGLYPUFF = 51
    TARGET_TEST_SAMUS = 52
    TARGET_TEST_SHEIK = 53
    TARGET_TEST_YOSHI = 54
    TARGET_TEST_ZELDA = 55
    TARGET_TEST_MR_GAME_AND_WATCH = 56
    TARGET_TEST_ROY = 57
    TARGET_TEST_GANONDORF = 58


class Character(IntEnum):
    UNKNOWN = -1
    CAPTAIN_FALCON = 0
    DONKEY_KONG = 1
    FOX = 2
    MR_GAME_AND_WATCH = 3
    KIRBY = 4
    BOWSER = 5
    LINK = 6
    LUIGI = 7
    MARIO = 8
    MARTH = 9
    MEWTWO = 10
    NESS = 11
    PEACH = 12
    PIKACHU = 13
    ICE_CLIMBERS = 14
    JIGGLYPUFF = 15
    SAMUS = 16
    YOSHI = 17
    ZELDA = 18
    SHEIK = 19
    FALCO = 20
    YOUNG_LINK = 21
    DR_MARIO = 22
    ROY = 23
    PICHU = 24
    GANONDORF = 25
    MASTER_HAND = 26
    WIREFRAME_MALE = 27
    WIREFRAME_FEMALE = 28
    GIGA_BOWSER = 29
    CRAZY_HAND = 30
    SANDBAG = 31
    POPO = 32


class Rank(Enum):
    UNKNOWN = "unknown"
    UNRANKED = "unranked"
    BRONZE_1 = "bronze_1"
    BRONZE_2 = "bronze_2"
    BRONZE_3 = "bronze_3"
    SILVER_1 = "silver_1"
    SILVER_2 = "silver_2"
    SILVER_3 = "silver_3"
    GOLD_1 = "gold_1"
    GOLD_2 = "gold_2"
    GOLD_3 = "gold_3"
    PLATINUM_1 = "platinum_1"
    PLATINUM_2 = "platinum_2"
    PLATINUM_3 = "platinum_3"
    DIAMOND_1 = "diamond_1"
    DIAMOND_2 = "diamond_2"
    DIAMOND_3 = "diamond_3"
    MASTER_1 = "master_1"
    MASTER_2 = "master_2"
    MASTER_3 = "master_3"
    GRANDMASTER = "grandmaster"


@dataclass(frozen=True)
class SlpMeta:
    map: Map
    character1: Character
    character2: Character
    match_duration_s: int
    char1_winner: bool | None
    rank: Rank


@dataclass(frozen=True)
class HfLocation:
    repo: str
    prefix: str = ""

    @classmethod
    def parse(cls, value: str) -> "HfLocation":
        parts = value.removeprefix("hf://").strip("/").split("/")
        if len(parts) < 2:
            raise ValueError("HF location must look like owner/repo or owner/repo/prefix")
        return cls(repo="/".join(parts[:2]), prefix="/".join(parts[2:]))

    def relative(self, path: str) -> str:
        return path.removeprefix(f"{self.prefix}/") if self.prefix else path

    def join(self, path: str) -> str:
        return "/".join(part for part in (self.prefix, path) if part)


STAGE_CODES = {
    "BF": Map.BATTLEFIELD,
    "PS": Map.POKEMON_STADIUM,
    "YS": Map.YOSHIS_STORY,
    "DL": Map.DREAM_LAND_N64,
    "FD": Map.FINAL_DESTINATION,
    "FOD": Map.FOUNTAIN_OF_DREAMS,
    "YI": Map.YOSHIS_ISLAND,
}

CHARACTER_ALIASES = {
    "CPTFALCON": Character.CAPTAIN_FALCON,
    "FALCON": Character.CAPTAIN_FALCON,
    "DK": Character.DONKEY_KONG,
    "GAMEANDWATCH": Character.MR_GAME_AND_WATCH,
    "PUFF": Character.JIGGLYPUFF,
    "ICS": Character.ICE_CLIMBERS,
    "DOC": Character.DR_MARIO,
    "YLINK": Character.YOUNG_LINK,
    "GANON": Character.GANONDORF,
}


class FilterSource:
    def filter_hf_for_map_and_one_char(
        self,
        hf_location_input: str,
        hf_location_output: str,
        map: Map,
        character: Character,
        *,
        max_files: int | None = None,
        dry_run: bool = False,
        private: bool = True,
        token: str | None = None,
    ) -> dict:
        """Copy HF SLPs whose filename says they match this map and character."""
        from huggingface_hub import HfApi, hf_hub_download

        source = HfLocation.parse(hf_location_input)
        output = HfLocation.parse(hf_location_output)
        hf_token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        api = HfApi(token=hf_token)

        if not dry_run:
            api.create_repo(output.repo, repo_type="dataset", token=hf_token, private=private, exist_ok=True)

        summary = {"scanned": 0, "matched": 0, "copied": 0, "examples": []}

        for source_path in self._slp_paths(api, source, hf_token):
            summary["scanned"] += 1
            if not self._path_matches(source_path, map, character):
                continue

            summary["matched"] += 1
            target_path = output.join(source.relative(source_path))
            if len(summary["examples"]) < 10:
                summary["examples"].append({"source": source_path, "target": target_path})

            if not dry_run:
                with tempfile.TemporaryDirectory(prefix="slp-filter-") as temp_dir:
                    local_path = hf_hub_download(
                        repo_id=source.repo,
                        repo_type="dataset",
                        token=hf_token,
                        filename=source_path,
                        local_dir=temp_dir,
                    )
                    api.upload_file(
                        repo_id=output.repo,
                        repo_type="dataset",
                        token=hf_token,
                        path_or_fileobj=local_path,
                        path_in_repo=target_path,
                    )
                summary["copied"] += 1

            if max_files is not None and summary["matched"] >= max_files:
                break

        return summary

    def _slp_paths(self, api, location: HfLocation, token: str | None):
        for item in api.list_repo_tree(
            location.repo,
            repo_type="dataset",
            token=token,
            path_in_repo=location.prefix or None,
            recursive=True,
        ):
            path = getattr(item, "path", "")
            if path.lower().endswith(".slp"):
                yield path

    def _path_matches(self, path: str, map: Map, character: Character) -> bool:
        return self._stage_from_path(path) == map and character in self._characters_from_path(path)

    def _stage_from_path(self, path: str) -> Map | None:
        match = re.search(r"\(([^()]*)\)\.slp$", PurePosixPath(path).name)
        return STAGE_CODES.get(self._key(match.group(1))) if match else None

    def _characters_from_path(self, path: str) -> set[Character]:
        name = PurePosixPath(path).name
        player_text = re.sub(r"\s*\([^()]*\)\.slp$", "", name)
        player_text = player_text.split(" ", 1)[1] if " " in player_text else ""
        players = {self._character_from_text(part) for part in player_text.split("+")}
        players.discard(None)

        folder = PurePosixPath(path).parts[0]
        if folder == "ZELDA_SHEIK":
            players.update({Character.ZELDA, Character.SHEIK})
        else:
            folder_character = self._character_from_text(folder)
            if folder_character is not None:
                players.add(folder_character)
        return players

    def _character_from_text(self, text: str) -> Character | None:
        text_key = self._key(re.sub(r"\[[^\]]*\]|\([^)]*\)", " ", text))
        if text_key in CHARACTER_ALIASES:
            return CHARACTER_ALIASES[text_key]
        for character in sorted(Character, key=lambda c: len(self._key(c.name)), reverse=True):
            if character is Character.UNKNOWN:
                continue
            if f" {self._key(character.name)} " in f" {text_key} ":
                return character
        return None

    def _key(self, text: str) -> str:
        text = text.upper().replace("&", " AND ")
        return re.sub(r"\s+", " ", re.sub(r"[^A-Z0-9]+", " ", text)).strip()


FileterSource = FilterSource
