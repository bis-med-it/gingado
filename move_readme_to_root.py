"""This script moves the README.md file from the docs folder into the project
root. This is required, since the index.qmd file can only be rendered into
files within the output folder. The post render option of the index.qmd file
will then call this script to move it."""
import os
import shutil

root_path = os.path.dirname(os.path.abspath(__file__))
source_path = os.path.join(root_path, 'docs', 'README.md')
destination_path = os.path.join(root_path, 'README.md')

if os.path.exists(source_path):
    # Move the file
    try:
        shutil.move(source_path, destination_path)
    except Exception as e:
        print(f"Error moving README.md file to project root: {e}")