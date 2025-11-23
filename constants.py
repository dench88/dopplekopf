# constants.py

# ---- Players ----
PLAYERS = ("RUSTY", "SUSIE", "HARLEM", "ALICE")

# If you ever want a default points dict:
PLAYER_POINTS_DEFAULT = {p: 0 for p in PLAYERS}


# ---- Card structure ----
types = ["9", "10", "K", "A", "J", "Q"]      # ranks
suits = ["diamonds", "hearts", "spades", "clubs"]

# Points per rank
cards_points = {"9": 0, "10": 10, "K": 4, "A": 11, "J": 2, "Q": 3}

# ---- Trump definition (single source of truth) ----
TRUMPS = [
    "9-diamonds", "K-diamonds", "10-diamonds", "A-diamonds",
    "J-diamonds", "J-hearts", "J-spades", "J-clubs",
    "Q-diamonds", "Q-hearts", "Q-spades", "Q-clubs",
    "10-hearts",
]
TRUMP_SET = set(TRUMPS)


# ---- Total ordering of cards by power ----
# This is your master list: lowest → highest power.
CARDS_POWER = [
    "9-hearts", "9-spades", "9-clubs",
    "K-hearts", "K-spades", "K-clubs",
    "10-spades", "10-clubs",
    "A-hearts", "A-spades", "A-clubs",
    "9-diamonds", "K-diamonds", "10-diamonds", "A-diamonds",
    "J-diamonds", "J-hearts", "J-spades", "J-clubs",
    "Q-diamonds", "Q-hearts", "Q-spades", "Q-clubs",
    "10-hearts",
]

# identifier -> integer power
CARD_POWER = {ident: i for i, ident in enumerate(CARDS_POWER)}


# ---- Trick category (suit vs trump) ----
TRICK_CATEGORY = {
    ident: ("trumps" if ident in TRUMP_SET else ident.split("-")[1])
    for ident in CARDS_POWER
}

# Non-trump “colour” cards, if you still care:
COLOUR_CARDS = [c for c in CARDS_POWER if c not in TRUMP_SET]

#------------------------

# types = ['9', '10', 'K', 'A', 'J', 'Q']

# suits = ['diamonds', 'hearts', 'spades', 'clubs']

# TRUMPS = ['9-diamonds', 'K-diamonds', '10-diamonds', 'A-diamonds',
#                 'J-diamonds', 'J-hearts', 'J-spades', 'J-clubs',
#                 'Q-diamonds', 'Q-hearts', 'Q-spades', 'Q-clubs',
#                 '10-hearts'
#                ]

# cards_points = {'9': 0, '10': 10, 'K': 4, 'A': 11, 'J': 2, 'Q': 3}

# CARDS_POWER = ['9-hearts', '9-spades', '9-clubs',
#                 'K-hearts', 'K-spades', 'K-clubs',
#                 '10-spades', '10-clubs',
#                 'A-hearts', 'A-spades', 'A-clubs',
#                '9-diamonds', 'K-diamonds', '10-diamonds', 'A-diamonds',
#                 'J-diamonds', 'J-hearts', 'J-spades', 'J-clubs',
#                 'Q-diamonds', 'Q-hearts', 'Q-spades', 'Q-clubs',
#                 '10-hearts'
#                ]
# CARD_POWER = {ident: i for i, ident in enumerate(CARDS_POWER)}

# TRICK_CATEGORY = {'9-hearts': 'hearts', '9-spades': 'spades', '9-clubs': 'clubs',
#                 'K-hearts': 'hearts', 'K-spades': 'spades', 'K-clubs': 'clubs',
#                 '10-spades': 'spades', '10-clubs': 'clubs',
#                 'A-hearts': 'hearts', 'A-spades': 'spades', 'A-clubs': 'clubs',
#                '9-diamonds': 'trumps', 'K-diamonds': 'trumps', '10-diamonds': 'trumps', 'A-diamonds': 'trumps',
#                 'J-diamonds': 'trumps', 'J-hearts': 'trumps', 'J-spades': 'trumps', 'J-clubs': 'trumps',
#                 'Q-diamonds': 'trumps', 'Q-hearts': 'trumps', 'Q-spades': 'trumps', 'Q-clubs': 'trumps',
#                 '10-hearts': 'trumps'
#                   }
# COLOUR_CARDS = list(set(cards_power) - set(trumps))

# players = {"RUSTY": [], "SUSIE": [], "HARLEM": [], "ALICE": []}

# player_points = {"RUSTY": 0, "SUSIE": 0, "HARLEM": 0, "ALICE": 0}

# print(colour_cards)