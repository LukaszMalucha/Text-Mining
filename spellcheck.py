# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 20:06:52 2020

@author: LukaszMalucha
"""

from spellchecker import SpellChecker

spell = SpellChecker()

# find those words that may be misspelled
misspelled = spell.unknown(['something', 'is', 'hapenning', 'here'])

for word in misspelled:
    # Get the one `most likely` answer
    print(spell.correction(word))

    # Get a list of `likely` options
    print(spell.candidates(word))