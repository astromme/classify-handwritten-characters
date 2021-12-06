"""
Creates `character.index` which gives each character a unique number
equal to the row number in the index file. This index is critical information
for both training and inference time and should be committed to the source
code. Typically this file will never need to be regenerated.
"""


from libgnt.character_index import write_character_index
from libgnt.gnt import samples_from_directory
from tqdm import tqdm
import sys

def main():
    max_characters = 10000
    directories = sys.argv[1:]
    character_index = []

    for directory in tqdm(directories):
        for bitmap, character in tqdm(samples_from_directory(directory)):
            try:
                label = character_index.index(character)
            except ValueError:
                if len(character_index) == max_characters:
                    continue # ignore this sample

                character_index.append(character)
                label = len(character_index) - 1

    print(f'{len(character_index)} characters in index')
    write_character_index(character_index)

if __name__ == '__main__':
    main()
