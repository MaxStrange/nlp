"""
This is the API for the part of the program that does the inference.
"""
import analyzer
import data


def get_relationship(rel_as_yam):
    """
    Creates a data.Relationship object from the given files.
    Used for inference, not training.
    """
    betrayal = False # Not needed for inference
    from_player = rel_as_yam[0]['a_to_b']['from_country']
    to_player = rel_as_yam[0]['a_to_b']['to_country']
    seasons = []
    for s in rel_as_yam:
        season = 0 if s['season'] == "Spring" else 0.5
        year = int(s['year']) + season
        interaction = None # Not needed for inference
        betrayer = "a_to_b" # Not needed for inference
        victim = "b_to_a" # Not needed for inference
        messages_betrayer = [analyzer.analyze_message(m) for m in s[betrayer]['messages']]
        messages_victim = [analyzer.analyze_message(m) for m in s[victim]['messages']]
        messages = {"betrayer": messages_betrayer, "victim": messages_victim}
        sdict = {"season": year, "interaction": interaction, "messages": messages}
        seasons.append(sdict)

    relationship = data.Relationship({"idx": 0, "game": 0, "betrayal": betrayal, "people": [from_player, to_player], "seasons": seasons})
    return relationship

