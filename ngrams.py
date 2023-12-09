####################################
#     ngrams.py                    #
# Sprachmodellierung mit N-Grammen #
#                                  #
# Marius Gerdes + some better naming #
# 8.11.2015                        #
####################################

from collections import Counter
import random


class NGrams(object):
    """Korpus basierte n-gramm Spachmodellierung.""
    NGrams erwartet einen Korpus als Liste von strings und erstellt dann
    eine bedingte Wahrscheinlichkeitsverteilung P(w_n | w_1, w_2, ..., w_n-1)
    anhand von maximum-likelihood Schaetzung.

    Beispielbenutzung:
    from ngrams import *
    ws = "the good the bad and the ugly".split()
    bigrams = NGrams(ws, 2)
    bigrams.ngrams()
    [('bad', 'and'), ('and', 'the'), ('the', 'good'), ('the', 'ugly'), ('good', 'the'), ('the', 'bad')]
    bigrams["the"]
    {'ugly': 0.3333333333333333, 'bad': 0.3333333333333333, 'good': 0.3333333333333333}
    bigrams.random_sample_for(("the"))
    'good'

    trigrams = NGrams(ws, 3)
    trigrams[("the","good")]["the"]
    1.0
    trigrams.random_sample_for(("the","good"))
    'the'
    """

    def __init__(self, ws, n=2):
        """Erschafft ein NGram Modell aus einer liste von Strings.
        ws - Eine Liste von strings
        n - Die Ordnung des NGram Modells als integer
        """
        if n < 1:
            raise RuntimeError("Error during NGramModel creation: Cannot create NGramModel of order 0 or lower.")

        self._n = n
        # wir merken uns beobachtete n-gramme
        self._ngrams = set()

        # wir zaehlen die Anzahl beobachteter n-grams
        # in einem dictionary von countern
        self._cond_freq_dist = self._count(ws, n)

        # bedingte Wahrscheinlichkeitsverteilung als
        # dictionary von dictionaries
        self._cond_prob_dist = self._ml_estimate(self._cond_freq_dist)

    def ngrams(self):
        """Gibt alle beobachteten N-Grams als liste zurueck."""
        return list(self._ngrams)

    def keys(self):
        """Gibt alle gueltigen Werte fuer KEY in self[KEY] oder
        self.random_sample_for(KEY) zurueck.
        """
        return self._cond_prob_dist.keys()

    def items(self):
        """Gibt alle Wahrscheinlichkeitsverteilungen fuer beobachtete Kontexte
        als (KONTEXT, DICTIONARY) Paar zurueck.
        """
        return self._cond_prob_dist.items()

    def random_sample_for(self, context):
        """Gibt ein zufaelliges Wort fuer einen gegebenen Kontext zurueck. Die
        Wahrscheinlichkeit des Rueckgabewerts richtet sich nach der
        Wahrscheinlichkeitsverteilung fuer CONTEXT.

        context - Ein (n-1)-Tupel von strings (Kann bei n=2 als string statt
            als 1-Tupel von strings gegeben werden)
        return - Ein zufaelliger string, der mit context beobachtet wurde.
        """
        threshold = random.random()

        for (word, p) in self[context].items():
            threshold -= p
            if threshold <= 0.0:
                return word

        if threshold <= 0.00001:
            # ohoh hier sollten wir nicht hinkommen
            # wir erlauben einen leichten rundungsfehler
            return word

        raise RuntimeError("Error during random sample creation: Probabilities do not sum to 1. Is there a rounding error?")

    def _count(self, ws, n):
        cfd = {}
        for i in range((len(ws)-n)+1):
            ngram = tuple(ws[i:i+n])
            self._ngrams.add(ngram)
            nth_word = ngram[-1]
            context = ngram[:-1]

            if not (context in cfd):
                cfd[context] = Counter()

            cfd[context][nth_word] += 1

        return cfd

    def _ml_estimate(self, cfd):
        cpd = {}

        for context in cfd.keys():
            context_count = float(sum(cfd[context].values()))
            cpd[context] = {}

            for next_word in cfd[context].keys():
                cpd[context][next_word] = cfd[context][next_word] / context_count

        return cpd

    def __getitem__(self, context):
        """Index operator []. self[context] gibt die Wahrscheinlichkeits-
        verteilung fuer den gegebnen Kontext wieder.

        context - Bei n-grammen ein (n-1)-Tupel aus strings.
           (kann bei n=2 (bigrams) als string statt als 1-Tupel angegeben
           werden)
        return - Ein Dictionary als string -> float mapping.
        """
        if self._n == 2:
            context = [context]
        return self._cond_prob_dist[tuple(context)]

    def __contains__(self, context):
        """Erlaubt Benutzung von 'in' keyword:
        ('the', 'good') in trigrams -> True
        Dies verhaelt sich so, dass wenn self[CONTEXT] einen Wert hat, CONTEXT
        in self = True ist.

        context - Ein (n-1)Tupel als vermeintlich beobachteter Kontext.
            (Bei n=2 auch als string akzeptierbar)
        return - Bool; je nach dem ob context beobachtet wurde.
        """
        if self._n == 2:
            context = [context]
        return tuple(context) in self._cond_prob_dist
