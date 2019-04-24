# jupyter-git
Lightweight tool for easier jupyter notebook version control.

OBS: WIP

## Description
This tool will strip your notebooks of (unnecessary) metadata, write only the essential metadata + all code blocks to new, easily readale text files in a hidden .gitnb folder for smooth version control.
To ignore a specific notebook with gitignore, specify notebookname.py in your .gitignore.


## Usage
- option `--write` will write notebook data to .gitnb. Run the script with this option, then commit the .gitnb folder and proceed as normal.
- option `--read` will read notebook data from .gitnb into .ipynb files. OBS: This will overwrite existing notebooks with the same name.

### TODO:
- Turn it into a git extension
