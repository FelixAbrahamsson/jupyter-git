import json
import glob
import os
import argparse

NB_GIT_DIR = '.nbgit/'
EXT = '.nbgit'


def read_nb_files():

    nb_files = glob.glob('*.ipynb')
    gitignore_files = []
    if os.path.exists('.gitignore'):
        with open('.gitignore', 'r') as f:
            for line in f:
                line = line.strip()
                if line.split('.')[-1] == EXT.strip('.'):
                    gitignore_files.append(line)

    nb_files = [f for f in nb_files 
                if f.replace('.ipynb', EXT) 
                not in gitignore_files]

    return nb_files


def strip_nb(data):

    new_json = {
        'cells' : [],
        'nbformat_minor' : data['nbformat_minor'],
        'nbformat' : data['nbformat'],
    }
    new_json['metadata'] = {
        'kernelspec' : data['metadata']['kernelspec'],
        'language_info' : data['metadata']['language_info'],
    }

    for cell in data['cells']:
        
        new_cell = {
            'cell_type' : cell['cell_type'],
            'source' : cell['source'],
        }
        
        new_json['cells'].append(new_cell)

    return new_json


def write_nbgit():

    fnames = read_nb_files()
    if len(fnames) > 0:
        if not os.path.exists(NB_GIT_DIR):
            os.makedirs(NB_GIT_DIR)

    for fname in fnames:

        with open(fname, 'r') as f:
            nb_json = json.load(f)

        stripped_json = strip_nb(nb_json)

        new_fname = NB_GIT_DIR + fname.replace('.ipynb', EXT)

        with open(new_fname, 'w') as f:
            json.dump(stripped_json, f)


def read_nbgit():

    for fname in os.listdir(NB_GIT_DIR):

        with open(NB_GIT_DIR + fname, 'r') as f:
            data = json.load(f)

        for cell in data['cells']:

            cell['metadata'] = {}
        
            if cell['cell_type'] == 'code':
                cell['execution_count'] = None
                cell['outputs'] = []

        nb_fname = fname.replace(EXT, '.ipynb')
        with open(nb_fname, 'w') as f:
            json.dump(data, f)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--write', action='store_true')
    parser.add_argument('--read', action='store_true')

    args = parser.parse_args()

    if args.write and args.read:
        print("Can't read and write at the same time.")
    elif not args.write and not args.read:
        print("Must either read or write.")
    else:
        if args.write:
            write_nbgit()
        if args.read:
            read_nbgit()