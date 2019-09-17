from verse import Verse
import re
import csv


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


def create_csv(verses_list):
    with open('data/mormon.csv', 'w', newline="") as csv_file:
        filewriter = csv.writer(csv_file, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for verse in verses_list:
            row = []
            row.append(verse.text)
            row.append(verse.author)
            filewriter.writerow(row)


verses = parse_file('./data/pg17.txt')
create_csv(verses)
