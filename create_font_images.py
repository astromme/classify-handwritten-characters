import sys
import os

from PIL import Image, ImageDraw, ImageFont
from libgnt.character_index import character_index
from tqdm import tqdm

IMAGE_WIDTH = IMAGE_HEIGHT = 100
FONT_SIZE = 80
PADDING_X = (IMAGE_WIDTH - FONT_SIZE) / 2
PADDING_Y = (IMAGE_HEIGHT - FONT_SIZE) / 2
FONT_FILE = "/System/Library/Fonts/STHeiti Medium.ttc"
FONT_NAME = "STHeitiMedium"

def create_character_image(character):
    image = Image.new('RGB', (IMAGE_WIDTH, IMAGE_HEIGHT), color=(255, 255, 255))
    font = ImageFont.truetype(FONT_FILE, FONT_SIZE)
    canvas = ImageDraw.Draw(image)
    canvas.text((PADDING_X, PADDING_Y), character, font=font, fill=(0, 0, 0))
    return image

def write_character_image(image, character, train_dir):
    directory = f'{train_dir}/{character}'
    filename = f'{directory}/{FONT_NAME}.png'
    try:
        os.makedirs(directory)
    except FileExistsError:
        pass
    image.save(filename)

def main():
    if len(sys.argv) < 2:
        print(f'usage: {sys.argv[0]} train_dir')
        print(f'Creates training images using system fonts')
        sys.exit()

    train_dir = sys.argv[1]

    for character in tqdm(character_index):
        if character.strip() == '':
            pass

        img = create_character_image(character)
        write_character_image(img, character, train_dir)

if __name__ == '__main__':
    main()
