import json
import random
import os
import numpy as np

def trans_score2probability(results):
    cnt = 0
    for res in results:
        # begin to assist info 
        if 'assist_info' in list(res.keys()):
            # print(list(res['assist_info'].keys()))
            assist_info_keys = list(res['assist_info'].keys())
            for assist_info_key in assist_info_keys:
                assist_candidates_dict = res['assist_info'][assist_info_key]['assist_candidates']
                values = list(assist_candidates_dict.values())
                if values[0] > 0 and values[1] > 0:
                    continue
                print(values)
                probabilities = np.exp(values) / np.sum(np.exp(values))
                prob_res = {}
                keys = list(assist_candidates_dict.keys())
                for i in range(len(keys)):
                    prob_res[keys[i]] = probabilities[i]
                res['assist_info'][assist_info_key]['assist_candidates'] = prob_res
                cnt += 1
        else:
            res['assist_info'] = {}
    print(cnt)
    return results

# def llm_prob(query_reses):
#     error_count = 0
#     for res in query_reses:
#         try:
#             candidates_dict = list(res['candidates_dict'].keys())
#             assist_info = res['assist_info']
#             assist_queries = list(res['assist_info'].keys())
#             for assist_query in assist_queries:
#                 assist_info[assist_query]['llm_prob'] = {}
#                 assist_candidates = list(assist_info[assist_query]['assist_candidates'].keys())
#                 candidates_1_probability = assist_info[assist_query]['statements_prob_norm'][assist_candidates[0]][0] * assist_info[assist_query]['assist_candidates'][assist_candidates[0]] \
#                                         + assist_info[assist_query]['statements_prob_norm'][assist_candidates[1]][0] * assist_info[assist_query]['assist_candidates'][assist_candidates[1]]
#                 candidates_2_probability = assist_info[assist_query]['statements_prob_norm'][assist_candidates[0]][1] * assist_info[assist_query]['assist_candidates'][assist_candidates[0]] \
#                                         + assist_info[assist_query]['statements_prob_norm'][assist_candidates[1]][1] * assist_info[assist_query]['assist_candidates'][assist_candidates[1]]
#                 assist_info[assist_query]['llm_prob'][candidates_dict[0]] = candidates_1_probability
#                 assist_info[assist_query]['llm_prob'][candidates_dict[1]] = candidates_2_probability
#             pass
#         except Exception as e:
#             # print('error', e)
#             error_count = error_count + 1
#             continue
#     print('error count:', error_count)
#     return query_reses
#     pass

def llm_prob(query_reses):
    error_count = 0
    for res in query_reses:
        res['balanced_answer'] = res['pred_ans']
        if 'equal_answer' in res:
            if res["equal_answer"] == True:
                continue
        candidates_dict = list(res['candidates_dict'].keys())
        assist_info = res['assist_info']
        assist_queries = list(res['assist_info'].keys())
        for assist_query in assist_queries:
            try:
                assist_info[assist_query]['llm_prob'] = {}
                assist_candidates = list(assist_info[assist_query]['assist_candidates'].keys())
                candidates_1_probability = assist_info[assist_query]['statements_prob_norm'][assist_candidates[0]][0] * assist_info[assist_query]['assist_candidates'][assist_candidates[0]] \
                                        + assist_info[assist_query]['statements_prob_norm'][assist_candidates[1]][0] * assist_info[assist_query]['assist_candidates'][assist_candidates[1]]
                candidates_2_probability = assist_info[assist_query]['statements_prob_norm'][assist_candidates[0]][1] * assist_info[assist_query]['assist_candidates'][assist_candidates[0]] \
                                        + assist_info[assist_query]['statements_prob_norm'][assist_candidates[1]][1] * assist_info[assist_query]['assist_candidates'][assist_candidates[1]]
                assist_info[assist_query]['llm_prob'][candidates_dict[0]] = candidates_1_probability
                assist_info[assist_query]['llm_prob'][candidates_dict[1]] = candidates_2_probability
            except Exception as e:
                # print('error', e)
                error_count = error_count + 1
                continue
    print('error count:', error_count)
    return query_reses
    pass

def merge(result_file_base):
    files = os.listdir(result_file_base)    
    results = []
    for f in files:
        if not os.path.isdir(result_file_base + '/' + f):
            results += json.load(open(result_file_base + '/' + f, "r"))
    return results


def trans_and_postprocess(results):
    print('res len:', len(results))
    # results = trans_score2probability(results)
    results = llm_prob(results)
    return results


