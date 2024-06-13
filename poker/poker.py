from libs.PokerOddsCalc.table import HoldemTable
from typing import List


def parse_cards(cards: List[str]):
    parsed_cards = []
    for card in cards:
        number, col = card.split(",")
        if number == "10":
            number = "T"
        elif number == "11":
            number = "J"
        elif number == "12":
            number = "Q"
        elif number == "13":
            number = "K"
        elif number == "14":
            number = "A"

        parsed_cards.append("".join([number, col.lower()]))

    return parsed_cards


def count_probability(my_cards: List[str], community_cards: List[str], num_players: int):
    holdem_game = HoldemTable(num_players=num_players, deck_type='full')

    my_cards_parsed = parse_cards(my_cards)
    community_cards_parsed = parse_cards(community_cards)

    holdem_game.add_to_hand(1, my_cards_parsed)
    holdem_game.add_to_community(community_cards_parsed)
    holdem_game.next_round()
    
    result = holdem_game.simulate()

    return [result["Player 1 Win"], result["Player 1 Tie"]]
