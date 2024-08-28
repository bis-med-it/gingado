# Open Source Software Licence Compliance Note

The BIS has independently authored the software program: `gingado`. It relies on multiple third-party software modules listed in `requirements.txt`.

`gingado` is licensed under the Apache License Version 2.0 and is therefore provided with no warranty. To comply with the terms of the licences covering the third-party components, `gingado` must be installed with the considerations below, any other installation method may not be compliant with the relevant third-party licences.

## Installation considerations

For a licence compliant installation, `gingado` must be installed using the package installer for Python (pip) using the --no-binary flag. An example installation command is:

`pip install gingado --no-binary :all:`

## Further information

1. Please note that usage of the --no-binary flag will increase the complexity of the installation (such as requiring building modules for some components from source). Please refer to third-party documentation for additional guidance; and

2. Please be aware that compliance materials may be placed into temporary directories by pip; and

3. When resolving dependencies for `gingado`, pip may automatically use a later version of a dependency. For convenience, the BIS has provided `verified-requirements.txt` which contains fixed version numbers to prevent this behaviour. An example installation using this file is: `pip install -r verified-requirements.txt --no-binary :all:`
