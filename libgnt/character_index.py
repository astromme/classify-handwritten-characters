_default_filename = 'characters.index'

def _read_character_index(filename=_default_filename):
    with open(filename, 'r') as f:
        character_index = [s.strip() for s in f.readlines()]
        return character_index

character_index = _read_character_index()

def write_character_index(character_index, filename=_default_filename):
    with open(filename, 'w') as f:
        for c in character_index:
            f.write(f'{c}\n')

    character_index = _read_character_index()
