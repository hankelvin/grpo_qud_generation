import pyrankvote
from pyrankvote import Candidate, Ballot
from pyrankvote.helpers import CompareMethodIfEqual
assert CompareMethodIfEqual.MostSecondChoiceVotes == 'MostSecondChoiceVotes'

def remove_duplicates(rankllm_sys_vote):
    # remove duplicates
    dedup_ranks = []
    for x in rankllm_sys_vote: 
        if x not in dedup_ranks: 
            dedup_ranks.append(x) 
    return dedup_ranks

def compute_stv(cfg, rankllm_choices, num_seats = 1):
    all_results = {}

    # get the set of districts 
    districts = set()
    for cdict in rankllm_choices.values(): 
        districts.update(cdict.keys())

    # put each district to vote
    for i, district in enumerate(districts):
        all_results[district] = do_one_line_stv(rankllm_choices, district, num_seats)
        
    return all_results
    
def do_one_line_stv(rankllm_choices, district, num_seats):
    candidates  = {} # These are what's being voted on 
    ballots     = [] # These are the ballots that contain the votes for this "district"
    # this is running through all rankllm systems
    for rankllm_sys, choices in rankllm_choices.items():
        rankllm_sys_vote    = choices[district]
        dedup_ranks         = remove_duplicates(rankllm_sys_vote)
        
        ##### add to candidates if it's not there yet #####
        for cand in dedup_ranks: 
            if cand not in candidates:
                candidates[cand] = Candidate(f'{cand}')
        ###################################################
        
        voter_ballot = Ballot(ranked_candidates = [candidates[r] for r in dedup_ranks])
        ballots.append(voter_ballot)
    
    try:
        # instant runoff for single seat, but pick the one with 2nd most votes in case of tie
        results = pyrankvote.instant_runoff_voting(candidates = list(candidates.values()), 
                                    ballots = ballots, 
                                    compare_method_if_equal = "MostSecondChoiceVotes")
        runoff_winners = [r.name for r in results.get_winners()]
        assert len(runoff_winners) == 1, len(runoff_winners)

        
        results_all_cands = pyrankvote.single_transferable_vote(candidates = list(candidates.values()), 
                                        ballots = ballots, number_of_seats = len(candidates)-1)
        winners_all_cands = [r.name for r in results_all_cands.get_winners()]
        # NOTE: add losing candidate
        loser = list(set(candidates).difference(set(winners_all_cands)))
        assert len(loser) == 1
        winners_all_cands.extend(loser)

    except: 
        all_voters_set      = set([c2.name for c in ballots for c2 in c.ranked_candidates])
        all_candidates_set  = set(all_voters_set)
        assert len(all_voters_set) == len(all_candidates_set) == num_seats
        runoff_winners = list(set(all_candidates_set))
        winners_all_cands = 'ALL VOTED THE SAME'
        print('ALL VOTED THE SAME')

    line                            = {}
    line['runoff_winners']          = runoff_winners
    line['stv_winners_all_cands']   = winners_all_cands 
    # NOTE: ordinal scores, highest score for 1st-ranked
    max_score                       = len(winners_all_cands)
    line['stv_scoring']             = {sysname: max_score - rank_pos \
                                        for rank_pos, sysname in enumerate(winners_all_cands)}
    
    return line