"""Settings for the gingado library."""

## GENERAL SETTINGS

# Path to the directory cached datasets are stored in
CACHE_DIRECTORY = './gingado'

## CB SPEECHES SETTINGS

# Base URL of CB speeches files (should end in a slash)
CB_SPEECHES_BASE_URL = 'https://www.bis.org/speeches/'

# Base name of the zip files on the website, used to create the full URL for downloading the file
CB_SPEECHES_ZIP_BASE_FILENAME = 'speeches'

# Base path used for storing the speeches files on disk
CB_SPEECHES_CSV_BASE_FILENAME = 'cb_speeches'

## MONOPOL STATEMENTS SETTINGS

# Base URL of CB speeches files (should end in a slash)
MONOPOL_STATEMENTS_BASE_URL = 'https://raw.githubusercontent.com/bis-med-it/gingado/main/assets/'

# Base path used for storing the speeches files on disk
MONOPOL_STATEMENTS_CSV_BASE_FILENAME = 'monpol_statements'
