## Usage

### General

Most scripts can be run using `python script_name.py [options]`. **Use the `--help` flag for detailed options for each script.**

## Scripts Included

### MKV Track Filter (`mkv_filter_script.py`)

* **Description:** Intelligently removes unwanted audio/subtitle tracks based on language (preferred list, original language via TMDb), commentary flags/keywords, SDH status, forced flags, and format preference. Calculates space saved.
* **Requirements:**
    * External Tools: MKVToolNix (`mkvmerge` command must be in system PATH)
    * Python Libraries: `tmdbv3api`, `pycountry`
    * Environment Variables: `TMDB_API_KEY` (Obtain from [themoviedb.org](https://www.themoviedb.org/). See Configuration section below for setup.)
* **Usage Examples:**
    * **Dry Run (Recommended First):** Scan a directory recursively, show what would happen (DEBUG log level recommended for detail):
        ```bash
        python mkv_filter_script.py -i "/path/to/movies" -r -d --log-level DEBUG
        ```
    * **Process Directory (Output to New Folder):** Keep English audio/subs, add original audio lang, keep forced English, output to `/path/to/output`:
        ```bash
        python mkv_filter_script.py -i "/path/to/movies" -r -o "/path/to/output" --preferred-langs eng
        ```
    * **Process Directory (Overwrite Originals):** Keep English and Spanish audio/subs (plus original audio lang), keep forced Eng/Spa, **overwrite original files (USE WITH CAUTION!)**:
        ```bash
        # ENSURE YOU HAVE BACKUPS BEFORE RUNNING THIS!
        python mkv_filter_script.py -i "/path/to/movies" -r --preferred-langs eng,spa --overwrite
        ```
* **Configuration:**
    * **TMDb API Key:** Requires a TMDb API key set as the `TMDB_API_KEY` environment variable for original language lookup. Obtain one from [themoviedb.org](https://www.themoviedb.org/).
        * **How to Set Environment Variables (Permanent - Recommended):**
            * **Windows:** Search for "Environment Variables", select "Edit environment variables for your account", click "New..." under "User variables", enter `TMDB_API_KEY` as name and your key as value. Click OK. **Restart any open terminals.** (Alternatively, use `setx TMDB_API_KEY "YourKey"` in cmd/powershell and restart the terminal).
            * **macOS/Linux (Bash/Zsh):** Add `export TMDB_API_KEY="YourKey"` to your shell profile file (e.g., `~/.zshrc`, `~/.bashrc`, `~/.profile`). Save the file and run `source ~/.your_profile_file` or restart your terminal.
        * *(For temporary testing in the current terminal only: Use `set VAR=VAL` (Win CMD), `$env:VAR="VAL"` (PowerShell), or `export VAR="VAL"` (macOS/Linux))*
