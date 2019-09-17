class Verse:
    def __init__(self, text, author):
        """
        :param text: the content of the verse
        :type text: str
        :param author: the author of the verse
        :type author: str
        """
        self.text = text
        self.author = author

    def __str__(self):
        return 'Verse: (\n author:{},\n text:{})'.format(self.author, self.text)

    def __repr__(self):
        return '\n' + self.__str__()
