"""Settings for the gingado library."""

from pathlib import Path
## GENERAL SETTINGS

# Path to the directory cached datasets are stored in
CACHE_DIRECTORY = './gingado'

# Path to the on-disk cache used for SDMX HTTP responses
SDMX_HTTP_CACHE_PATH = Path(CACHE_DIRECTORY) / "sdmx_http_cache"

# Default time-to-live for cached SDMX HTTP responses, in seconds
SDMX_HTTP_CACHE_EXPIRE_AFTER = 24 * 60 * 60

## CB SPEECHES SETTINGS

# Base URL of CB speeches files (should end in a slash)
CB_SPEECHES_BASE_URL = 'https://www.bis.org/speeches/'

# Base name of the zip files on the website, used to create the full URL for downloading the file
CB_SPEECHES_ZIP_BASE_FILENAME = 'speeches'

# Base path used for storing the speeches files on disk
CB_SPEECHES_CSV_BASE_FILENAME = 'cb_speeches'

## MONPOL STATEMENTS SETTINGS

# Base URL of CB speeches files (should end in a slash)
MONPOL_STATEMENTS_BASE_URL = 'https://raw.githubusercontent.com/bis-med-it/gingado/main/assets/'

# Base path used for storing the speeches files on disk
MONPOL_STATEMENTS_CSV_BASE_FILENAME = 'monpol_statements'
