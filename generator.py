"""Gerüst für eine Implementierung des Dissociated-Press-Generators."""

from nltk.corpus import brown


class Generator:
    def __init__(self, n: int):
        self.n = n  # bigram order
        self.model = None  # the language model to apply

    def train(self, words: list):
        """Trains the model on the basis of a corpus.

        Args:
        words: A list of strings – the corpus to train on.
        """
        pass

    def generate(self, text_length: int):
        """Applies the model to generate a text of the given length.

        Args:
        text_length: The length of the text to be generated.

        Returns:
        A string with the generated text.
        """
        return ''


if __name__ == '__main__':
    brown_words = list(brown.words())

    generator = Generator(2)  # set up a generator for a bigram model
    generator.train(brown_words)  # train the bigram model on the Brown corpus
    text_1 = generator.generate(100)  # generate a text of 100 words
    text_2 = generator.generate(100)  # generate another text of 100 words

    print(text_1)
    print()
    print(text_2)
