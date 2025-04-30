import random
import constants

class Card:
    def __init__(self, type, suit):
        self.type = type
        self.suit = suit
        self.identifier = f'{self.type}-{self.suit}'
        self.points = constants.cards_points[self.type]
        try:
            self.power = constants.CARD_POWER[self.identifier]
        except KeyError:
            raise ValueError(f"Unknown card id {self.identifier!r}")
        self.category = constants.trick_category[self.identifier]


def create_deck():
    cards = []
    for type in constants.types:
        for suit in constants.suits:
            card = Card(type, suit)
            cards.append(card)
            cards.append(card)

    random.shuffle(cards)

    return cards

