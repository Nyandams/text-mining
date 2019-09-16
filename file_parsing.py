from verse import Verse
import re
import pandas as pd


def parse_file(file):
    try:
        with open(file, 'r') as fp:
            verse_list = []
            new_verse = True
            text = ''
            author = ''

            for line in fp:
                if new_verse:
                    words = line.split()
                    for word in words:
                        split = re.search("[a-zA-Z]*", word).group()
                        if split != '':
                            author = split
                    new_verse = False

                elif line in ['\n', '\r\n']:
                    verse_list.append(Verse(text=text, author=author))
                    new_verse = True
                    text = ''

                else:
                    text += line
            return verse_list

    except IOError:
        print("Error while loading the file")
        return None


verses = parse_file('./data/pg17.txt')
