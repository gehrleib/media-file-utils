#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MKV Track Filter Script

Purpose:
Automates the process of filtering unwanted audio and subtitle tracks from
Matroska (MKV) video files. It uses TMDb (The Movie Database) to identify the
original language of movies and applies user-defined preferences to select which
tracks to keep. Uses 'pycountry' for language code mapping. Calculates space saved.
Adjusts filename stem for '-GroupName' convention unless group is 'Radarr'/'Sonarr'.

Key Features:
- Filters based on language (user-preferred list + optional original language).
  Original language only added to preferred set for AUDIO filtering by default.
  Subtitles processes each preferred language independently, keeping Forced tracks and the
  best Regular track (non-SDH > SDH, then format preference). Falls back to 'und'.
- Removes commentary/description tracks (flags, keywords, heuristics).
- Selects best subtitle format (SSA/ASS > SRT > WebVTT > PGS > VobSub).
- Calculates and reports approximate disk space saved.
- Adjusts final filename stem to include ' -GroupName' if missing (excludes Radarr/Sonarr).
- Flexible input/output (single file, dir, recursive, output dir, overwrite).
- Dry run mode.

Requirements:
- Python 3.7+ (due to `BooleanOptionalAction` and f-string usage)
- MKVToolNix ('mkvmerge' in PATH)
- tmdbv3api library (`pip install tmdbv3api`)
- pycountry library (`pip install pycountry`)
- TMDb API Key (environment variable TMDB_API_KEY)
- MKV files with TMDb ID in filename (e.g., "{tmdb-12345}")
"""

import argparse
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

# --- Library Checks & Imports ---
try:
    import pycountry

    PYCOUNTRY_AVAILABLE = True
except ImportError:
    PYCOUNTRY_AVAILABLE = False

try:
    from tmdbv3api import Movie as TMDbMovie
    from tmdbv3api import TMDb

    TMDB_AVAILABLE = True
except ImportError:
    TMDB_AVAILABLE = False


# --- Constants ---
APP_NAME = "MKV Track Filter"
DEFAULT_LOG_DIR = Path("./logs")
MKVMERGE_COMMAND = "mkvmerge"

LANG_UND = "und"  # Undetermined language code

TYPE_VIDEO = "video"
TYPE_AUDIO = "audio"
TYPE_SUBTITLES = "subtitles"

# Keywords to identify commentary tracks by name (case-insensitive)
COMMENTARY_KEYWORDS = [
    "commentary",
    "director",
    "audio description",
    "visually impaired",
    "narration",
    "narrated",
    "descriptive",
]
# MKVToolNix JSON flags indicating commentary/description tracks
COMMENTARY_FLAGS = {"flag_commentary", "flag_visual_impaired"}

# Preferred order of subtitle formats (best first) used in filtering
SUBTITLE_FORMAT_ORDER = [
    "SSA/ASS",
    "SubRip/SRT",
    "WebVTT",
    "HDMV PGS",
    "VobSub",
]
# Keywords to identify SDH subtitles by name (case-insensitive)
SUBTITLE_SDH_KEYWORDS = ["sdh", "hearing impaired"]
# MKVToolNix JSON flag indicating hearing impaired (SDH) subtitle track
SUBTITLE_FLAG_HEARING_IMPAIRED = "flag_hearing_impaired"
# MKVToolNix JSON flag indicating forced subtitle track
SUBTITLE_FLAG_FORCED = "forced_track"

# Regex to find TMDb ID in filenames like {tmdb-12345}
TMDB_ID_PATTERN = re.compile(r"\{tmdb-(\d+)\}")
# Environment variable name for the API key
TMDB_API_KEY_ENV_VAR = "TMDB_API_KEY"


# --- Helper Functions ---
def setup_logging(log_level: str, log_dir: Path) -> logging.Logger:
    """Sets up console and file logging."""
    logger = logging.getLogger(APP_NAME)
    logger.setLevel(logging.DEBUG)  # Capture all levels internally
    if logger.hasHandlers():
        logger.handlers.clear()  # Avoid duplicate handlers

    log_dir.mkdir(parents=True, exist_ok=True)  # Ensure log dir exists
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"mkv_filter_{timestamp}.log"

    # File Handler - Detailed logs
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    fh.setFormatter(fh_formatter)
    logger.addHandler(fh)

    # Console Handler - Level set by user
    ch = logging.StreamHandler(sys.stdout)
    try:
        console_level = getattr(logging, log_level.upper(), logging.INFO)
    except AttributeError:
        console_level = logging.INFO
        print(
            f"Warning: Invalid log level '{log_level}'. Defaulting to INFO.",
            file=sys.stderr,
        )
    ch.setLevel(console_level)
    ch_formatter = logging.Formatter("%(levelname)s: %(message)s")
    ch.setFormatter(ch_formatter)
    logger.addHandler(ch)

    logger.info(
        f"Logging initialized. Console Level: {log_level.upper()}. "
        f"Log file: {log_file}"
    )
    return logger


# --- Main Processing Class ---
class MkvTrackFilter:
    """Handles the overall process of finding, analyzing, and filtering MKV files."""

    def __init__(self, args: argparse.Namespace):
        """Initializes the filter with args and sets up logging."""
        self.args = args
        self.logger = setup_logging(args.log_level, args.log_dir)
        self.mkvmerge_path: Optional[Path] = None
        self.tmdb: Optional[TMDb] = None
        self.tmdb_movie: Optional[TMDbMovie] = None
        self.stats = {
            "processed": 0,
            "skipped": 0,
            "errors": 0,
            "remuxed": 0,
            "overwritten": 0,
            "space_saved_bytes": 0,
        }

        # Check required libraries early
        if not TMDB_AVAILABLE:
            self.logger.warning(
                "'tmdbv3api' not found. TMDb lookups disabled."
            )
        if not PYCOUNTRY_AVAILABLE:
            self.logger.critical(
                "'pycountry' not found. Language mapping impossible."
            )
            sys.exit(1)

        # Parse preferred languages from argument string, ensuring 3-letter codes
        langs_raw = args.preferred_langs.split(",")
        valid_langs = {
            lang.strip().lower()
            for lang in langs_raw
            if lang.strip() and len(lang.strip()) == 3
        }
        invalid_langs = {
            lang.strip()
            for lang in langs_raw
            if lang.strip() and len(lang.strip()) != 3
        }
        if invalid_langs:
            self.logger.warning(
                "Ignoring invalid preferred language codes "
                f"(must be 3 letters): {invalid_langs}"
            )
        self.user_preferred_langs = valid_langs

        self.logger.info(
            f"User preferred languages: {self.user_preferred_langs or '{None}'}"
        )
        self.logger.info(
            "Add original language to preferred set (for audio): "
            f"{args.add_original_lang}"
        )

    def _human_readable_size(self, size_bytes: int) -> str:
        """Converts bytes to a human-readable string (KB, MB, GB)."""
        if size_bytes < 0:
            size_bytes = 0
        if size_bytes == 0:
            return "0 B"
        size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
        i = 0
        size_float = float(size_bytes)
        # Find appropriate unit
        while size_float >= 1024 and i < len(size_name) - 1:
            size_float /= 1024.0
            i += 1
        # Format and remove trailing zeros if .00
        s = "{:.2f}".format(size_float)
        s = s.rstrip("0").rstrip(".") if "." in s else s
        return f"{s} {size_name[i]}"

    def _initialize_tools(self) -> bool:
        """Finds mkvmerge executable and initializes TMDb API connection."""
        # Find mkvmerge
        mkvmerge_path_str = shutil.which(MKVMERGE_COMMAND)
        if not mkvmerge_path_str:
            self.logger.critical(f"'{MKVMERGE_COMMAND}' not found in PATH.")
            return False
        self.mkvmerge_path = Path(mkvmerge_path_str)
        self.logger.info(f"Found {MKVMERGE_COMMAND} at: {self.mkvmerge_path}")

        # Initialize TMDb if possible
        if TMDB_AVAILABLE:
            api_key = os.getenv(TMDB_API_KEY_ENV_VAR)
            if not api_key:
                self.logger.warning(
                    f"TMDb API key ({TMDB_API_KEY_ENV_VAR}) not set. "
                    "Lookup disabled."
                )
                self.tmdb, self.tmdb_movie = None, None
            else:
                try:
                    self.tmdb = TMDb()
                    self.tmdb.api_key = api_key
                    self.tmdb_movie = TMDbMovie()
                    self.logger.info("TMDb API initialized.")
                except Exception as e:
                    self.logger.error(
                        f"Failed initialize TMDb API: {e}",
                        exc_info=self.args.log_level == "DEBUG",
                    )
                    self.tmdb, self.tmdb_movie = None, None  # Proceed without
        else:
            self.tmdb, self.tmdb_movie = None, None  # Library not installed

        # pycountry availability checked in __init__
        if not PYCOUNTRY_AVAILABLE:
            return False
        return True

    def _find_mkv_files(self, input_path: Path) -> List[Path]:
        """Finds MKV files based on input path and recursion flag."""
        mkv_files: List[Path] = []
        if input_path.is_file():
            if input_path.suffix.lower() == ".mkv":
                mkv_files.append(input_path)
            else:
                self.logger.warning(f"Input '{input_path}' is not an MKV file.")
        elif input_path.is_dir():
            scan_type = (
                "recursively" if self.args.recursive else "non-recursively"
            )
            self.logger.info(f"Scanning directory: {input_path} ({scan_type})")
            pattern = "**/*.mkv" if self.args.recursive else "*.mkv"
            mkv_files = sorted(list(input_path.glob(pattern)))
            self.logger.info(f"Found {len(mkv_files)} MKV files.")
        else:
            self.logger.error(f"Input path '{input_path}' not found.")
        return mkv_files

    def _get_tmdb_language(self, tmdb_id: int) -> Optional[str]:
        """
        Queries TMDb for original language, returns 3-letter ISO 639-2 code.
        Uses pycountry for mapping. Returns None on failure.
        """
        if not self.tmdb or not self.tmdb_movie:
            self.logger.debug("TMDb API N/A.")
            return None
        # pycountry checked in init

        try:
            self.logger.debug(f"Querying TMDb for ID: {tmdb_id}")
            movie_details = self.tmdb_movie.details(tmdb_id)
            if not movie_details or not hasattr(
                movie_details, "original_language"
            ):
                self.logger.warning(f"TMDb: No details/lang for ID: {tmdb_id}")
                return None

            lang_code_2 = movie_details.original_language
            if not lang_code_2:
                self.logger.warning(f"TMDb empty lang code ID: {tmdb_id}")
                return None

            # Handle direct 3-letter codes or invalid format from TMDb
            if len(lang_code_2) == 3:
                return lang_code_2.lower()
            if len(lang_code_2) != 2:
                self.logger.warning(f"TMDb invalid lang code '{lang_code_2}'.")
                return None

            # Use pycountry for 2 -> 3 letter mapping
            try:
                lang_obj = pycountry.languages.get(alpha_2=lang_code_2.lower())
                if lang_obj:
                    # Prefer Bibliographic ('ger'), fallback Terminology ('deu')
                    if (
                        hasattr(lang_obj, "bibliographic")
                        and lang_obj.bibliographic
                    ):
                        code_3 = lang_obj.bibliographic
                        self.logger.debug(
                            f"Mapped '{lang_code_2}'->'{code_3}' "
                            "(pycountry biblio)."
                        )
                        return code_3
                    elif hasattr(lang_obj, "alpha_3") and lang_obj.alpha_3:
                        code_3 = lang_obj.alpha_3
                        self.logger.debug(
                            f"Mapped '{lang_code_2}'->'{code_3}' "
                            "(pycountry alpha_3)."
                        )
                        return code_3
                    else:
                        self.logger.warning(
                            f"pycountry found '{lang_code_2}' but lacks "
                            "3-letter code."
                        )
                        return None
                else:
                    self.logger.warning(
                        f"pycountry could not find lang code '{lang_code_2}'."
                    )
                    return None
            except Exception as e:
                self.logger.error(
                    f"pycountry lookup error for '{lang_code_2}': {e}"
                )
                return None
        except Exception as e:
            self.logger.error(
                f"TMDb query error ID {tmdb_id}: {e}",
                exc_info=self.args.log_level == "DEBUG",
            )
            return None

    def _parse_mkv_json(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Runs 'mkvmerge -J' and parses the JSON output."""
        if not self.mkvmerge_path:
            self.logger.error("mkvmerge path missing.")
            return None
        cmd = [str(self.mkvmerge_path), "-J", str(file_path)]
        self.logger.debug(f"Executing: {' '.join(map(str, cmd))}")
        try:
            # Run mkvmerge identify command
            process = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
                encoding="utf-8",
                errors="replace",
                timeout=300,
            )
            if process.returncode != 0:
                self.logger.error(
                    f"mkvmerge -J failed '{file_path.name}' "
                    f"Code:{process.returncode}. Stderr: {process.stderr[:1000]}..."
                )
                return None
            # Parse JSON output
            try:
                return json.loads(process.stdout)
            except json.JSONDecodeError as e:
                self.logger.error(
                    f"JSON parse failed '{file_path.name}': {e}. "
                    f"Output: {process.stdout[:1000]}..."
                )
                return None
        except subprocess.TimeoutExpired:
            self.logger.error(f"mkvmerge -J timeout '{file_path.name}'.")
            return None
        except FileNotFoundError:
            self.logger.critical(f"'{self.mkvmerge_path}' vanished!")
            sys.exit(1)
        except Exception as e:
            self.logger.error(
                f"mkvmerge -J error '{file_path.name}': {e}",
                exc_info=self.args.log_level == "DEBUG",
            )
            return None

    def _is_commentary(self, track: Dict[str, Any], track_type: str) -> bool:
        """Checks if track is commentary. Logs details at DEBUG level."""
        properties = track.get("properties", {})
        track_name_full = properties.get("track_name", "")
        track_name_lower = track_name_full.lower()
        track_id = track.get("id", "N/A")
        lang = properties.get("language", LANG_UND).lower()
        codec = track.get("codec", "Unknown")

        reason = None
        log_details = ""

        # Check Flags
        commentary_flags_found = [f for f in COMMENTARY_FLAGS if properties.get(f)]
        if commentary_flags_found:
            reason = f"Flag ({', '.join(commentary_flags_found)})"

        # Check Keywords
        if not reason:
            matched_keywords = [
                kw for kw in COMMENTARY_KEYWORDS if kw in track_name_lower
            ]
            if matched_keywords:
                reason = f"Keyword ('{', '.join(matched_keywords)}')"

        # Check Heuristics (Audio Only)
        if (
            not reason
            and track_type == TYPE_AUDIO
            and self.args.heuristic_commentary
        ):
            channels = properties.get("audio_channels")
            if channels is not None and channels <= 2:
                if any(
                    c in codec.lower()
                    for c in ["ac-3", "e-ac-3", "dts", "aac", "opus"]
                ):
                    reason = f"Heuristic (codec:{codec}, ch:{channels})"

        # Log if commentary detected
        if reason:
            log_prefix = f"Ignoring commentary {track_type} track {track_id}"
            core_info = (
                f"Lang='{lang}', Codec='{codec}', Name='{track_name_full}'"
            )
            if track_type == TYPE_AUDIO:
                channels = properties.get("audio_channels", "N/A")
                log_details = f"{core_info}, Ch={channels}"
            else:
                log_details = core_info
            self.logger.debug(f"{log_prefix}: {log_details} (Reason: {reason})")
            return True

        return False  # Not commentary

    def _filter_audio_tracks(
        self, tracks: List[Dict[str, Any]], preferred_languages: Set[str]
    ) -> List[int]:
        """Selects audio tracks based on preferred languages & commentary."""
        self.logger.debug(
            f"Filtering audio tracks. Preferred: {preferred_languages or '{None}'}"
        )
        valid_tracks_info = []
        commentary_tracks_ids = []
        preferred_lang_audio = []
        und_audio = []

        for track in tracks:
            if track.get("type") != TYPE_AUDIO:
                continue
            track_id = track["id"]
            properties = track.get("properties", {})
            lang = properties.get("language", LANG_UND).lower()
            if self._is_commentary(track, TYPE_AUDIO):
                commentary_tracks_ids.append(track_id)
                continue
            track_info = {"id": track_id, "lang": lang}
            valid_tracks_info.append(track_info) # Store all valid non-commentary
            if lang in preferred_languages:
                preferred_lang_audio.append(track_info)
            elif lang == LANG_UND:
                und_audio.append(track_info)
            # else: other languages are implicitly in valid_tracks_info

        # Selection logic: Preferred > UND > Any other valid > Commentary Fallback
        audio_tracks_to_keep_ids = []
        if preferred_lang_audio:
            ids = [t["id"] for t in preferred_lang_audio]
            self.logger.debug(f"Keeping {len(ids)} preferred language audio.")
            audio_tracks_to_keep_ids = ids
        elif und_audio:
            ids = [t["id"] for t in und_audio]
            self.logger.debug(f"Keeping {len(ids)} Undetermined audio.")
            audio_tracks_to_keep_ids = ids
        elif valid_tracks_info: # Fallback to *all* valid tracks if others empty
            ids = [t["id"] for t in valid_tracks_info]
            self.logger.warning(
                f"No preferred/und audio. Keeping {len(ids)} other "
                "non-commentary audio."
            )
            audio_tracks_to_keep_ids = ids
        elif commentary_tracks_ids: # Only commentary exists
            ids = commentary_tracks_ids
            self.logger.warning(
                f"Only commentary audio found. Keeping {len(ids)} track(s)."
            )
            audio_tracks_to_keep_ids = ids
        else:
            self.logger.info("No audio tracks found/kept.")

        self.logger.debug(
            f"Final audio tracks selected: {sorted(audio_tracks_to_keep_ids)}"
        )
        return sorted(audio_tracks_to_keep_ids)

    def _filter_subtitle_tracks(
        self, tracks: List[Dict[str, Any]], preferred_languages: Set[str]
    ) -> List[int]:
        """Selects subtitle tracks, processing each preferred language independently."""
        self.logger.debug(
            "Filtering subtitle tracks. Preferred languages: "
            f"{preferred_languages or '{None}'}"
        )
        final_kept_ids: Set[int] = set()

        # --- Step 1: Process each preferred language ---
        for pref_lang in preferred_languages:
            self.logger.debug(f"Processing preferred language: '{pref_lang}'")
            forced_this_lang = []
            regular_non_sdh_this_lang = []
            regular_sdh_this_lang = []

            # Categorize tracks for this lang
            for track in tracks:
                properties = track.get("properties", {})
                lang = properties.get("language", LANG_UND).lower()
                if track.get("type") != TYPE_SUBTITLES or lang != pref_lang:
                    continue

                track_id = track["id"]
                track_name_full = properties.get("track_name", "")
                codec = track.get("codec", "Unknown")

                if self._is_commentary(track, TYPE_SUBTITLES):
                    continue # Logged inside _is_commentary

                is_sdh = bool(
                    properties.get(SUBTITLE_FLAG_HEARING_IMPAIRED)
                ) or any(
                    kw in track_name_full.lower() for kw in SUBTITLE_SDH_KEYWORDS
                )
                is_forced = properties.get(SUBTITLE_FLAG_FORCED, False)
                track_info = {
                    "id": track_id, "lang": lang, "codec": codec,
                    "is_sdh": is_sdh, "name": track_name_full,
                }

                if is_forced:
                    forced_this_lang.append(track_info)
                    continue # Handle forced separately

                # Log non-forced, non-commentary track being processed
                reason_str = " (Reason: SDH)" if is_sdh else ""
                self.logger.debug(
                    f"Processing subtitle track {track_id} for '{pref_lang}': "
                    f"Codec='{codec}', Name='{track_name_full}'{reason_str}"
                )
                if is_sdh:
                    regular_sdh_this_lang.append(track_info)
                else:
                    regular_non_sdh_this_lang.append(track_info)

            # Process results for this language
            kept_forced_ids_this_lang = {t["id"] for t in forced_this_lang}
            if kept_forced_ids_this_lang:
                self.logger.debug(
                    f"Keeping forced track(s) for '{pref_lang}': "
                    f"{sorted(list(kept_forced_ids_this_lang))}"
                )
                final_kept_ids.update(kept_forced_ids_this_lang)

            # Select best regular track group (non-SDH > SDH)
            selected_regular_group: List[Dict[str, Any]] = []
            group_desc = ""
            if regular_non_sdh_this_lang:
                selected_regular_group = regular_non_sdh_this_lang
                group_desc = "non-SDH"
            elif regular_sdh_this_lang:
                selected_regular_group = regular_sdh_this_lang
                group_desc = "SDH"

            # Apply format filter if a regular group was selected
            if selected_regular_group:
                self.logger.debug(
                    f"Selected regular {group_desc} group for '{pref_lang}' "
                    f"({len(selected_regular_group)}). Filtering..."
                )
                best_regular_ids_this_lang = set()
                best_format_level = len(SUBTITLE_FORMAT_ORDER)
                for track_info in selected_regular_group:
                    track_id = track_info["id"]; codec = track_info["codec"]
                    try: level = SUBTITLE_FORMAT_ORDER.index(codec)
                    except ValueError: level = len(SUBTITLE_FORMAT_ORDER)

                    if level < best_format_level:
                        best_regular_ids_this_lang = {track_id}
                        best_format_level = level
                    elif level == best_format_level:
                        best_regular_ids_this_lang.add(track_id)

                if best_regular_ids_this_lang:
                    self.logger.debug(
                        f"Keeping best regular {group_desc} track(s) for "
                        f"'{pref_lang}': {sorted(list(best_regular_ids_this_lang))}"
                    )
                    final_kept_ids.update(best_regular_ids_this_lang)
                else:
                    self.logger.warning(
                        f"No regular {group_desc} tracks kept for '{pref_lang}' "
                        "after format filter."
                    )
            else:
                self.logger.debug(f"No regular tracks found for '{pref_lang}'.")

        # --- Step 2: Fallback to Undetermined ('und') if NO preferred tracks kept ---
        if not final_kept_ids:
            self.logger.info("No preferred subs kept. Checking 'und' non-SDH.")
            und_non_sdh_tracks = []
            for track in tracks:
                properties = track.get("properties", {})
                lang = properties.get("language", LANG_UND).lower()
                if track.get("type") == TYPE_SUBTITLES and lang == LANG_UND:
                    if not self._is_commentary(track, TYPE_SUBTITLES):
                        is_sdh = bool(
                            properties.get(SUBTITLE_FLAG_HEARING_IMPAIRED)
                        ) or any(
                            kw in properties.get("track_name", "").lower()
                            for kw in SUBTITLE_SDH_KEYWORDS
                        )
                        is_forced = properties.get(SUBTITLE_FLAG_FORCED, False)
                        if not is_forced and not is_sdh:
                            und_non_sdh_tracks.append({
                                "id": track["id"],
                                "codec": track.get("codec", "Unknown")
                            })

            if und_non_sdh_tracks: # Apply format filter
                self.logger.debug(f"Found {len(und_non_sdh_tracks)} 'und' non-SDH tracks.")
                best_und_ids = set(); best_format_level = len(SUBTITLE_FORMAT_ORDER)
                for track_info in und_non_sdh_tracks:
                    track_id = track_info["id"]; codec = track_info["codec"]
                    try: level = SUBTITLE_FORMAT_ORDER.index(codec)
                    except ValueError: level = len(SUBTITLE_FORMAT_ORDER)
                    if level < best_format_level: best_und_ids = {track_id}; best_format_level = level
                    elif level == best_format_level: best_und_ids.add(track_id)
                if best_und_ids:
                    self.logger.debug(f"Keeping best 'und' track(s): {sorted(list(best_und_ids))}")
                    final_kept_ids.update(best_und_ids)
                else: self.logger.warning("No 'und' tracks kept after format filtering.")
            else: self.logger.info("No suitable 'und' non-SDH tracks found.")

        # --- Step 3: Return combined IDs ---
        final_sorted_list = sorted(list(final_kept_ids))
        self.logger.debug(f"Final subtitle tracks selected: {final_sorted_list}")
        if not final_sorted_list:
            self.logger.info("No subtitle tracks kept.")
        return final_sorted_list

    def _build_mkvmerge_command(
        self, input_file: Path, output_file: Path,
        video_tracks_to_keep: List[int], audio_tracks_to_keep: List[int],
        subtitle_tracks_to_keep: List[int]
    ) -> List[str]:
        """Constructs the mkvmerge command line arguments."""
        if not self.mkvmerge_path: raise RuntimeError("mkvmerge path missing.")
        cmd = [str(self.mkvmerge_path), "-o", str(output_file)]
        # Video tracks
        if video_tracks_to_keep:
            cmd.extend(["--video-tracks", ",".join(map(str, video_tracks_to_keep))])
        else:
            cmd.append("--no-video")
            self.logger.warning(f"No video selected for '{input_file.name}'.")
        # Audio tracks
        if audio_tracks_to_keep:
            cmd.extend(["--audio-tracks", ",".join(map(str, audio_tracks_to_keep))])
        else:
            cmd.append("--no-audio")
            self.logger.warning(f"No audio selected for '{input_file.name}'.")
        # Subtitle tracks
        if subtitle_tracks_to_keep:
            cmd.extend(["--subtitle-tracks", ",".join(map(str, subtitle_tracks_to_keep))])
        else:
            cmd.append("--no-subtitles")
            self.logger.debug(f"No subtitles selected for '{input_file.name}'.")
        # Input file
        cmd.append(str(input_file))
        return cmd

    def _adjust_filename_stem(self, stem: str) -> str:
        """Adds ' -' before last word if missing, unless word is 'Radarr'/'Sonarr'."""
        original_stem = stem
        parts = stem.rsplit(None, 1)
        if len(parts) == 2:
            prefix, last_word_raw = parts
            cleaned_prefix = prefix.rstrip()
            chars_to_strip = ')]}' # Characters to strip from end for comparison
            comparable_last_word = last_word_raw.rstrip(chars_to_strip)
            self.logger.debug(
                f"Adjust Check: Prefix='{cleaned_prefix}', LastRaw='{last_word_raw}', "
                f"Comparable='{comparable_last_word}'"
            )
            if not cleaned_prefix.endswith(' -'): # Only adjust if ' -' is missing
                if comparable_last_word.lower() in ["radarr", "sonarr"]:
                    self.logger.debug(f"Last word '{comparable_last_word}', skipping adjustment.")
                    return original_stem # Do not adjust
                else:
                    new_stem = f"{cleaned_prefix} -{last_word_raw}" # Use raw word
                    self.logger.debug(f"Adjusting stem: '{original_stem}' -> '{new_stem}'")
                    return new_stem
            else: return original_stem # Already conforms
        else: return original_stem # No space / single word

    def _determine_output_path(self, input_file: Path) -> Optional[Path]:
        """
        Determines the correct initial output path location/name based on args.
        Handles path creation and checks for existing files if not overwriting.
        Returns the Path object for the initial output file, or None if skipping.
        Note: Final filename might be adjusted later by _adjust_filename_stem.
        """
        input_file = input_file.resolve() # Work with absolute paths

        # Determine base output path based on mode
        if self.args.output_dir:
            # Use specified output directory, keep original filename initially
            output_path = self.args.output_dir.resolve() / input_file.name
            # Create output directory if needed (only on actual run)
            if not self.args.dry_run:
                try:
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                except OSError as e:
                    self.logger.error(
                        f"Create dir FAIL '{output_path.parent}': {e}. "
                        f"Skip '{input_file.name}'."
                    )
                    return None
        elif self.args.overwrite:
            # Overwrite mode: initial target path is the input path itself.
            # The actual remux might use a temporary file.
            output_path = input_file
        else:
            # Default mode: output next to input, add '.filtered' before extension.
            output_path = input_file.with_suffix(".filtered" + input_file.suffix)

        # Determine path to display during dry run if different from actual target
        display_output_path = output_path
        is_overwrite_dry_run_local = (
            self.args.dry_run
            and not self.args.output_dir
            and output_path == input_file # Check if target is same as input
        )
        if is_overwrite_dry_run_local:
            # Show that dry run *would* create a distinct file if not for -o
            display_output_path = input_file.with_suffix(
                ".filtered" + input_file.suffix
            )

        # Check if the *actual* target output file exists, only if *not* overwriting.
        if not self.args.overwrite and output_path.exists():
            self.logger.warning(
                f"Output '{output_path}' exists. Skip '{input_file.name}' "
                "(use --overwrite or -o)."
            )
            return None

        # Return path (might be adjusted display path for dry run logs)
        if self.args.dry_run and display_output_path != output_path:
            self.logger.debug(
                f"Dry run target:'{output_path}', display:'{display_output_path}'"
            )
            return display_output_path # Return display path for logging consistency
        else:
            # For actual runs, or dry runs where display path matches target path
            self.logger.debug(f"Determined output path: {output_path}")
            return output_path

    def _process_single_file(self, file_path: Path) -> None:
        """Processes a single MKV file through all filtering steps."""
        self.logger.info(f"--- Processing file: {file_path.name} ---")
        self.stats["processed"] += 1
        file_processed_successfully = False
        original_size = 0
        original_stem = file_path.stem

        try:
            # === 1. Basic File Checks ===
            if not file_path.is_file():
                self.logger.error(f"Not found: {file_path}")
                self.stats["errors"] += 1; return
            try:
                original_size = file_path.stat().st_size
                if original_size == 0:
                    self.logger.warning(f"Empty: {file_path.name}. Skip.")
                    self.stats["skipped"] += 1; return
            except OSError as e:
                 self.logger.error(f"Stats error '{file_path.name}': {e}. Skip.")
                 self.stats["errors"] += 1; return

            # === 2. Determine Output Path (initial location/name) ===
            output_file_path = self._determine_output_path(file_path)
            if output_file_path is None:
                self.stats["skipped"] += 1; return # Reason logged in function

            # === 3. Determine IDEAL final filename stem ===
            final_stem = self._adjust_filename_stem(original_stem)
            is_overwrite_same_file = (
                self.args.overwrite and
                not self.args.output_dir and
                output_file_path.resolve() == file_path.resolve()
            )
            # Calculate the final intended path based on mode and adjusted stem
            final_path = (
                file_path.with_stem(final_stem)
                if is_overwrite_same_file
                else output_file_path.with_stem(final_stem)
            )

            # === 4. Extract TMDb ID & Get Original Language ===
            tmdb_match = TMDB_ID_PATTERN.search(file_path.stem); original_language = None
            if tmdb_match:
                tmdb_id = int(tmdb_match.group(1))
                self.logger.debug(f"Found TMDb ID: {tmdb_id}")
                original_language = self._get_tmdb_language(tmdb_id) # Needs pycountry
                if original_language: self.logger.info(f"Original language: '{original_language}'")
                else: self.logger.warning(f"Failed get lang for TMDb ID {tmdb_id}.")
            else:
                self.logger.warning(f"No TMDb ID in: {file_path.name}.")

            # === 5. Determine Effective Preferred Languages (Separately for Audio/Subs) ===
            preferred_langs_for_audio = set(self.user_preferred_langs)
            preferred_langs_for_subtitles = set(self.user_preferred_langs)
            if (self.args.add_original_lang and original_language and
                    original_language not in preferred_langs_for_audio):
                self.logger.debug(f"Adding '{original_language}' to preferred set for AUDIO.")
                preferred_langs_for_audio.add(original_language)
            if not preferred_langs_for_audio: self.logger.warning(f"No preferred languages for AUDIO processing in {file_path.name}.")
            if not preferred_langs_for_subtitles: self.logger.warning(f"No preferred languages for SUBTITLE processing in {file_path.name}.")

            # === 6. Get MKV Track Info ===
            mkv_info = self._parse_mkv_json(file_path)
            if not mkv_info or "tracks" not in mkv_info:
                self.logger.error(f"No valid track info: {file_path.name}. Skip.")
                self.stats["errors"] += 1; return
            all_tracks = mkv_info.get("tracks", [])
            if not all_tracks:
                self.logger.warning(f"No tracks found: {file_path.name}. Skip.")
                self.stats["skipped"] += 1; return

            # === 7. Filter Tracks ===
            initial_video_ids = {t["id"] for t in all_tracks if t.get("type") == TYPE_VIDEO}
            initial_audio_ids = {t["id"] for t in all_tracks if t.get("type") == TYPE_AUDIO}
            initial_subtitle_ids = {t["id"] for t in all_tracks if t.get("type") == TYPE_SUBTITLES}
            video_tracks_to_keep = sorted(list(initial_video_ids)) # Keep all video
            audio_tracks_to_keep = self._filter_audio_tracks(all_tracks, preferred_langs_for_audio)
            subtitle_tracks_to_keep = self._filter_subtitle_tracks(all_tracks, preferred_langs_for_subtitles)
            kept_video_ids = set(video_tracks_to_keep); kept_audio_ids = set(audio_tracks_to_keep); kept_subtitle_ids = set(subtitle_tracks_to_keep)

            # === 8. Check if Changes Needed ===
            if (initial_video_ids == kept_video_ids and
                    initial_audio_ids == kept_audio_ids and
                    initial_subtitle_ids == kept_subtitle_ids):
                self.logger.info(f"No changes needed: {file_path.name}. Skip.")
                self.stats["skipped"] += 1; return
            self.logger.info(
                f"Keep Tracks: V={video_tracks_to_keep}, A={audio_tracks_to_keep}, "
                f"S={subtitle_tracks_to_keep}"
            )

            # === 9. Build Command & Handle Overwrite Temp File ===
            temp_output_path = None; mkvmerge_target_path = output_file_path
            if is_overwrite_same_file:
                 temp_output_path = file_path.with_suffix('.mkvmerge_temp' + file_path.suffix)
                 mkvmerge_target_path = temp_output_path
                 self.logger.debug(f"Overwrite: Using temp '{temp_output_path.name}'")
                 if temp_output_path.exists():
                     self.logger.warning(f"Removing stale temp: {temp_output_path}")
                     try: temp_output_path.unlink()
                     except OSError as e: self.logger.error(f"Rm stale temp FAIL '{temp_output_path}': {e}. Skip '{file_path.name}'."); self.stats["errors"] += 1; return
            mkvmerge_cmd = self._build_mkvmerge_command(
                file_path, mkvmerge_target_path, video_tracks_to_keep,
                audio_tracks_to_keep, subtitle_tracks_to_keep
            )

            # === 10. Execute mkvmerge or Simulate ===
            self.logger.info(f"Intended final path: '{final_path}'")

            if self.args.dry_run:
                display_cmd = mkvmerge_cmd[:]
                display_cmd[2] = str(output_file_path) # Show initial target path
                self.logger.info(f"[DRY RUN] Would execute: {' '.join(map(str,display_cmd))}")
                # Show potential rename/delete actions
                if temp_output_path: # Overwrite mode dry run
                    if final_path != file_path:
                         self.logger.info(f"[DRY RUN] Would rename temp to: '{final_path.name}'")
                         self.logger.info(f"[DRY RUN] Would delete original: '{file_path.name}'")
                    else: # Overwrite but no name change
                         self.logger.info(f"[DRY RUN] Would replace original: '{file_path.name}'")
                elif output_file_path.stem != final_stem: # Default / -o mode rename dry run
                    self.logger.info(f"[DRY RUN] Would rename output to: '{final_path.name}'")
                return # End dry run for this file

            # --- Actual Execution ---
            self.logger.info(f"Executing: {' '.join(map(str, mkvmerge_cmd))}")
            start_remux_time = time.time()
            process = None
            path_after_mkvmerge = mkvmerge_target_path # Path where mkvmerge wrote
            final_path_after_ops = path_after_mkvmerge # Track final path after all ops

            try:
                process = subprocess.run(
                    mkvmerge_cmd, capture_output=True, text=True, check=False,
                    encoding='utf-8', errors='replace', timeout=3600
                )
                remux_duration = time.time() - start_remux_time

                if process.returncode == 0: # Remux SUCCESS
                    self.logger.info(
                        f"Remux OK -> '{path_after_mkvmerge.name}' ({remux_duration:.2f}s)."
                    )
                    file_processed_successfully = True

                    # === Post-Remux Renaming / Overwrite Logic (NEW OVERWRITE BEHAVIOR) ===
                    if temp_output_path: # --overwrite mode used temp file
                        try:
                            # Rename temp file to the *final adjusted name*
                            self.logger.debug(
                                f"Overwrite: Renaming temp '{temp_output_path.name}' "
                                f"to final '{final_path.name}'"
                            )
                            temp_output_path.rename(final_path)
                            final_path_after_ops = final_path # Update final path tracker

                            # If rename succeeded AND final name is DIFFERENT from original, delete original
                            if final_path != file_path:
                                try:
                                    self.logger.debug(f"Overwrite: Deleting original '{file_path.name}'")
                                    file_path.unlink()
                                    self.logger.info(
                                        f"Overwrite OK: Original '{file_path.name}' removed, "
                                        f"saved as '{final_path.name}'."
                                    )
                                except OSError as e_unlink:
                                    # Log warning but don't mark as error if only delete failed
                                    self.logger.warning(
                                        f"Overwrite Warning: Saved '{final_path.name}', but "
                                        f"couldn't delete original '{file_path}'. Error: {e_unlink}"
                                    )
                            else: # Renamed temp successfully but to the same original filename
                                 self.logger.info(
                                     f"Overwrite OK: '{file_path.name}' replaced "
                                     "(no name change)."
                                 )
                            self.stats["overwritten"] += 1 # Count as processed in overwrite mode

                        except OSError as e_rename:
                            self.logger.error(
                                f"Overwrite FAIL: Could not rename '{temp_output_path}' "
                                f"to '{final_path}'. Error: {e_rename}\n"
                                "Temp file may remain."
                            )
                            self.stats["errors"] += 1
                            file_processed_successfully = False # Mark failure

                    else: # Default or -o mode
                        if path_after_mkvmerge != final_path: # Rename if stem was adjusted
                            try:
                                self.logger.info(
                                    f"Renaming output: '{path_after_mkvmerge.name}' -> "
                                    f"'{final_path.name}'"
                                )
                                path_after_mkvmerge.rename(final_path)
                                final_path_after_ops = final_path # Update final path
                            except OSError as e:
                                self.logger.error(
                                    f"Rename FAIL: Could not rename "
                                    f"'{path_after_mkvmerge}' to '{final_path}'. Error: {e}"
                                )
                                self.stats["errors"] += 1
                                file_processed_successfully = False # Keep original output name
                        # else: No rename needed, final_path_after_ops is correct

                    # === Calculate Space Saved (using final path) ===
                    if file_processed_successfully:
                        try:
                            filtered_size = final_path_after_ops.stat().st_size
                            if original_size > 0 and filtered_size >= 0:
                                saved_this_file = original_size - filtered_size
                                if saved_this_file > 0:
                                    self.stats["space_saved_bytes"] += saved_this_file
                                    self.logger.info(
                                        "Space saved: "
                                        f"{self._human_readable_size(saved_this_file)}"
                                    )
                                elif saved_this_file < 0:
                                    self.logger.warning(
                                        "File size increased by "
                                        f"{self._human_readable_size(abs(saved_this_file))}."
                                    )
                        except OSError as e:
                            self.logger.error(
                                f"Stat failed for final file "
                                f"'{final_path_after_ops.name}': {e}"
                            )

                else: # Remux FAILED
                    self.logger.error(
                        f"mkvmerge FAIL '{file_path.name}' "
                        f"Code:{process.returncode} ({remux_duration:.2f}s). "
                        f"Stderr: {process.stderr[:1000]}..."
                    )
                    self.stats["errors"] += 1
                    file_processed_successfully = False

            # --- Handle Execution Exceptions ---
            except subprocess.TimeoutExpired:
                remux_duration = time.time() - start_remux_time
                self.logger.error(
                    f"mkvmerge TIMEOUT ({remux_duration:.0f}s) for '{file_path.name}'."
                )
                self.stats["errors"] += 1; file_processed_successfully = False
            except FileNotFoundError:
                self.logger.critical(f"'{self.mkvmerge_path}' vanished!")
                sys.exit(1)
            except Exception as e:
                self.logger.error(
                    f"mkvmerge error '{file_path.name}': {e}",
                    exc_info=self.args.log_level == 'DEBUG'
                )
                self.stats["errors"] += 1; file_processed_successfully = False
            finally: # Cleanup temp file if needed
                 if not file_processed_successfully and temp_output_path and temp_output_path.exists():
                     self.logger.info(f"Cleaning up temporary file: {temp_output_path}")
                     try:
                         temp_output_path.unlink()
                     except OSError as e:
                         self.logger.error(
                             f"Failed to clean up temporary file "
                             f"'{temp_output_path}': {e}"
                         )
        except Exception as e: # Catch errors in this file's processing logic
            self.logger.error(
                f"Error processing '{file_path.name}': {e}",
                exc_info=self.args.log_level == 'DEBUG'
            )
            self.stats["errors"] += 1
        finally: # Update stats and log separator
            if file_processed_successfully:
                self.stats["remuxed"] += 1
            self.logger.debug("-" * 20)

    def run(self) -> int:
        """Main execution flow: initialize, find files, process, summarize."""
        start_time = time.time()
        self.logger.info(f"--- {APP_NAME} Started ---")

        # Warnings for dry-run/overwrite
        if self.args.dry_run:
            self.logger.warning(" DRY RUN MODE ".center(60, "="))
            self.logger.warning(" No files will be modified ".center(60))
            self.logger.warning("=" * 60)
        is_overwrite_active = self.args.overwrite and not self.args.output_dir
        if is_overwrite_active:
            self.logger.warning(" OVERWRITE MODE ".center(60, "!"))
            self.logger.warning(" Original files WILL BE REPLACED/REMOVED ".center(60))
            self.logger.warning("!" * 60)
        elif self.args.overwrite and self.args.output_dir:
            self.logger.info("Note: --overwrite ignored with -o.")

        # Initialize tools
        if not self._initialize_tools():
            self.logger.critical("Tool init failed. Exiting.")
            return 1

        # Find files
        mkv_files = self._find_mkv_files(self.args.input_path)
        if not mkv_files:
            self.logger.warning(f"No MKV files found in '{self.args.input_path}'.")
            return 0

        # Process files
        self.logger.info(f"Starting processing for {len(mkv_files)} MKV files...")
        for mkv_file in mkv_files:
            try:
                self._process_single_file(mkv_file)
            except Exception as e: # Catch unexpected errors during file loop
                self.logger.error(
                    f"CRITICAL error processing '{mkv_file.name}': {e}",
                    exc_info=True # Always include traceback for critical loop errors
                )
                self.stats["errors"] += 1

        # Final Summary
        end_time = time.time()
        duration = end_time - start_time
        self.logger.info("--- Processing Summary ---")
        self.logger.info(f"Total files scanned:           {self.stats['processed']}")
        self.logger.info(f"Files successfully processed:  {self.stats['remuxed']}")
        if is_overwrite_active:
            self.logger.info(f"Files overwritten/replaced:    {self.stats['overwritten']}")
        self.logger.info(f"Files skipped (no changes etc):{self.stats['skipped']}")
        self.logger.info(f"Errors encountered:            {self.stats['errors']}")
        if self.stats["space_saved_bytes"] > 0 and not self.args.dry_run:
            total_saved_readable = self._human_readable_size(
                self.stats["space_saved_bytes"]
            )
            self.logger.info(f"Approx. total space saved:     {total_saved_readable}")
        self.logger.info(f"Total execution time:          {duration:.2f} seconds")

        # Safely get and log the log file path
        log_file_path = None
        try:
            log_file_path = next(
                (h.baseFilename for h in self.logger.handlers
                 if isinstance(h, logging.FileHandler)), None
            )
        except Exception as e:
            self.logger.error(f"Could not determine log file path: {e}")
        if log_file_path:
            self.logger.info(f"Log file:                      {log_file_path}")

        self.logger.info(f"--- {APP_NAME} Finished ---")
        return 1 if self.stats["errors"] > 0 else 0


# --- Main Execution Guard ---
def main():
    """Parses arguments, creates the filter object, and runs the process."""
    # Check essential library before parsing args
    if not PYCOUNTRY_AVAILABLE:
        print(
            "Error: Required library 'pycountry' not found. "
            "Install with `pip install pycountry`", file=sys.stderr
        )
        sys.exit(1)

    # Setup argument parser
    parser = argparse.ArgumentParser(
        description=f"{APP_NAME}: Filter MKV tracks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # Argument Groups
    io_group = parser.add_argument_group('Input/Output Options')
    behavior_group = parser.add_argument_group('Processing Behavior')
    lang_group = parser.add_argument_group('Language Selection')
    log_group = parser.add_argument_group('Logging Options')

    # Define Arguments
    io_group.add_argument(
        "-i", "--input-path", type=Path, default=".", metavar="PATH",
        help="Input MKV file or directory."
    )
    io_group.add_argument(
        "-r", "--recursive", action="store_true",
        help="Scan directories recursively."
    )
    io_group.add_argument(
        "-o", "--output-dir", type=Path, default=None, metavar="DIR",
        help="Output to separate directory (disables --overwrite)."
    )
    behavior_group.add_argument(
        "-d", "--dry-run", action="store_true",
        help="Simulate only, no file changes."
    )
    behavior_group.add_argument(
        "--overwrite", action="store_true",
        help="Replace original file with potentially renamed filtered version "
             "(use caution, ignored if -o used)."
    )
    behavior_group.add_argument(
        "--heuristic-commentary", action="store_true",
        help="Enable audio commentary detection heuristics."
    )
    lang_group.add_argument(
        "--preferred-langs", type=str, default="eng", metavar="CODES",
        help="Comma-separated list of preferred 3-letter language codes "
             "(ISO 639-2/T). Ex: eng,jpn"
    )
    lang_group.add_argument(
        "--add-original-lang", action=argparse.BooleanOptionalAction, default=True,
        help="Add movie's original language (from TMDb) to preferred list "
             "FOR AUDIO tracks."
    )
    log_group.add_argument(
        "--log-dir", type=Path, default=DEFAULT_LOG_DIR, metavar="DIR",
        help="Directory for log files."
    )
    log_group.add_argument(
        "--log-level", default="INFO", metavar="LEVEL",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Console logging level."
    )

    # Informational warning if TMDb API key likely missing
    if TMDB_AVAILABLE and not os.getenv(TMDB_API_KEY_ENV_VAR):
        print(
            f"Warning: Env var {TMDB_API_KEY_ENV_VAR} not set. "
            "TMDb lookup disabled.", file=sys.stderr
        )

    # Parse arguments
    args = parser.parse_args()

    # Create processor and run
    processor = MkvTrackFilter(args)
    exit_code = processor.run()
    sys.exit(exit_code)


# Run main() if script is executed directly
if __name__ == "__main__":
    main()
