# jupyter-git
Lightweight tool for easier jupyter notebook version control.
OBS: WIP

## Description
This tool will strip your notebooks of (unnecessary) metadata, write only the essential metadata + all code blocks to new json files in a hidden .nbgit folder for smooth version control.
To ignore a specific notebook with gitignore, specify notebookname.nbgit in your .gitignore.


## Usage
    - option `--write` will write notebook data to .nbgit. Run the script with this option, then commit the .nbgit folder and proceed as normal.
    - option `--read` will read notebook data from .nbgit into .ipynb files. OBS: This will overwrite existing notebooks with the same name.