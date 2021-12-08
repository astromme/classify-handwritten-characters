"""
Creates `character.index` which gives each character a unique number
equal to the row number in the index file. This index is critical information
for both training and inference time and should be committed to the source
code. Typically this file will never need to be regenerated.
"""


from libgnt.character_index import write_character_index
from tqdm import tqdm
import sys
import os

def main():
    max_characters = 10000
    dir = sys.argv[1]
    character_index = []

    for dir, sub_dirs, files in os.walk(dir):
        character_index.extend(sub_dirs)

    print(character_index[:100])


    print(f'{len(character_index)} characters in index')
    write_character_index(character_index)

if __name__ == '__main__':
    main()
