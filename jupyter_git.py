import json
import glob
import os
import argparse
import re

NB_GIT_DIR = '.gitnb/'
EXT = '.py'
BLOCK_SEPARATORS = {
    'code' : '\n###_CODEBLOCK_###\n',
    'markdown' : '\n###_MARKDONWBLOCK_###\n',
}
SEPS_TO_BLOCK = {sep : name for name, sep in BLOCK_SEPARATORS.items()}


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

def get_output_text(data):
    
    metadata = {
        'nbformat_minor' : data['nbformat_minor'],
        'nbformat' : data['nbformat'],
    }
    metadata['metadata'] = {
        'kernelspec' : data['metadata']['kernelspec'],
        'language_info' : data['metadata']['language_info'],
    }
    
    output = str(metadata) + '\n'
    
    for cell in data['cells']:
        
        output += BLOCK_SEPARATORS[cell['cell_type']]
        
        for line in cell['source']:
            output += line
        
    return output


def write_gitnb():

    fnames = read_nb_files()
    if len(fnames) > 0:
        if not os.path.exists(NB_GIT_DIR):
            os.makedirs(NB_GIT_DIR)

    for fname in fnames:

        with open(fname, 'r') as f:
            nb_json = json.load(f)

        nb_text = get_output_text(nb_json)

        new_fname = NB_GIT_DIR + fname.replace('.ipynb', EXT)

        with open(new_fname, 'w') as f:
            f.write(nb_text)


def read_gitnb():

    for fname in os.listdir(NB_GIT_DIR):

        with open(NB_GIT_DIR + fname, 'r') as f:
            lines = f.readlines()
            
        metadata = json.loads(lines[0].replace("'", "\""))
        out_json = metadata
        out_json['cells'] = []
        
        if len(lines) > 1:
            
            separator_re = r'(' + r'|'.join([ sep for sep in BLOCK_SEPARATORS.values()]) + r')'
            text = ''.join(lines)
            split = re.split(separator_re, text)
            if split[0] not in SEPS_TO_BLOCK:
                split = split[1:]
            
            end = (len(split) // 2) * 2
            blocks = [(SEPS_TO_BLOCK[split[i]], split[i+1]) for i in range(0, end, 2)]
            
            for cell_type, source in blocks:
                
                cell = {
                    'metadata' : {},
                    'cell_type' : cell_type,
                    'source' : source,
                }
                
                if cell_type == 'code':
                    
                    cell['execution_count'] = None
                    cell['outputs'] = []
                
                out_json['cells'].append(cell)
            
        nb_fname = (fname).replace(EXT, '.ipynb')
        with open(nb_fname, 'w') as f:
            json.dump(out_json, f)
            
    return out_json


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
            write_gitnb()
        if args.read:
            read_gitnb()