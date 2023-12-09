import matplotlib.pyplot as plt
import re

from nltk.corpus import brown
from collections import Counter


with open(r"src\jungle_book.txt", encoding="utf8") as jungle_book:
    jungle_book_str = jungle_book.read()
    jungle_book_cleaned = re.sub('\W+', ' ', jungle_book_str)
    jungle_book_tokens = jungle_book_cleaned.split()

with open(r"src\bibel_utf8_noref.txt", encoding="utf8") as bible:
    bible_str = bible.read()
    bible_cleaned = re.sub('\W+', ' ', bible_str)
    bible_tokens = bible_cleaned.split()


def plot(token_list: list[str]):
    token_occurrences = Counter(token_list).most_common()
    num_types = len(token_occurrences)

    plt.plot(*zip(*token_occurrences),
             color="brown")

    plt.plot([(x + 1) for x in range(num_types)],
             [num_types / (x + 1) for x in range(num_types)],
             color="gray")


if __name__ == "__main__":
    match input("0 for Brown, 1 for JungleBook, 2 for Bible:\n"):
        case "0": plot(brown.words())
        case "1": plot(jungle_book_tokens)
        case "2": plot(bible_tokens)
        case _: exit()

    plt.yscale('log'), plt.xscale('log')
    plt.show()
