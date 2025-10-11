from magi.decision.aggregator import choose_verdict ,majority_weighted ,parse_vote ,PersonaVote 
from magi.decision.schema import PersonaOutput 


def test_majority_weighted_prefers_high_confidence ():
    votes =[
    PersonaVote (name ="melchior",action ="approve",confidence =0.2 ),
    PersonaVote (name ="balthasar",action ="reject",confidence =0.9 ),
    PersonaVote (name ="casper",action ="approve",confidence =0.1 ),
    ]
    assert majority_weighted (votes )=="reject"


def test_parse_vote_defaults_to_revise_when_no_tag ():
    persona =PersonaOutput (name ="melchior",text ="Need more data",confidence =0.5 )
    vote =parse_vote (persona )
    assert vote .action =="revise"


def test_choose_verdict_uses_tags ():
    personas =[
    PersonaOutput (name ="melchior",text ="[APPROVE] looks good",confidence =0.7 ),
    PersonaOutput (name ="balthasar",text ="[REJECT] too risky",confidence =0.8 ),
    PersonaOutput (name ="casper",text ="[REVISE] add guardrails",confidence =0.9 ),
    ]
    assert choose_verdict (personas )=="revise"
