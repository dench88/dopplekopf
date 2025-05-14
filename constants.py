
types = ['9', '10', 'K', 'A', 'J', 'Q']

suits = ['diamonds', 'hearts', 'spades', 'clubs']

trumps = ['9-diamonds', 'K-diamonds', '10-diamonds', 'A-diamonds',
                'J-diamonds', 'J-hearts', 'J-spades', 'J-clubs',
                'Q-diamonds', 'Q-hearts', 'Q-spades', 'Q-clubs',
                '10-hearts'
               ]

cards_points = {'9': 0, '10': 10, 'K': 4, 'A': 11, 'J': 2, 'Q': 3}

cards_power = ['9-hearts', '9-spades', '9-clubs',
                'K-hearts', 'K-spades', 'K-clubs',
                '10-spades', '10-clubs',
                'A-hearts', 'A-spades', 'A-clubs',
               '9-diamonds', 'K-diamonds', '10-diamonds', 'A-diamonds',
                'J-diamonds', 'J-hearts', 'J-spades', 'J-clubs',
                'Q-diamonds', 'Q-hearts', 'Q-spades', 'Q-clubs',
                '10-hearts'
               ]
CARD_POWER = {ident: i for i, ident in enumerate(cards_power)}

trick_category = {'9-hearts': 'hearts', '9-spades': 'spades', '9-clubs': 'clubs',
                'K-hearts': 'hearts', 'K-spades': 'spades', 'K-clubs': 'clubs',
                '10-spades': 'spades', '10-clubs': 'clubs',
                'A-hearts': 'hearts', 'A-spades': 'spades', 'A-clubs': 'clubs',
               '9-diamonds': 'trumps', 'K-diamonds': 'trumps', '10-diamonds': 'trumps', 'A-diamonds': 'trumps',
                'J-diamonds': 'trumps', 'J-hearts': 'trumps', 'J-spades': 'trumps', 'J-clubs': 'trumps',
                'Q-diamonds': 'trumps', 'Q-hearts': 'trumps', 'Q-spades': 'trumps', 'Q-clubs': 'trumps',
                '10-hearts': 'trumps'
                  }
colour_cards = list(set(cards_power) - set(trumps))

players = {"RUSTY": [], "SUSIE": [], "HARLEM": [], "ALICE": []}

player_points = {"RUSTY": 0, "SUSIE": 0, "HARLEM": 0, "ALICE": 0}

# print(colour_cards)