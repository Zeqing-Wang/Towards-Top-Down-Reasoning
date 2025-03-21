import argparse
import copy
import json
from tqdm import tqdm
import re
import sys
import re
import json
import os
import vqa_filter
import datetime
from VQAEval import VQAEval
from collections import Counter
import copy
from postprocess_res import trans_and_postprocess
def evaluate_prd_acc(res):
    count = 0
    for r in res:
        if r['pred_ans'].lower() == r['gt_ans'].lower():
            count = count + 1
    return count/len(res)

def evaluate_balance_acc(res):
    count = 0
    for r in res:
        if r['balanced_answer'].lower() == r['gt_ans'].lower():
            count = count + 1
    return count/len(res)

def process_overflow(res):
    # origin candidates
    for r in res:
        if 'assist_info' not in r:
            continue
        if r['pred_ans'] != max(r['candidates_dict'], key=r['candidates_dict'].get):
            candidate_keys = list(r['candidates_dict'].keys())
            r['candidates_dict'][candidate_keys[0]], r['candidates_dict'][candidate_keys[1]] = r['candidates_dict'][candidate_keys[1]], r['candidates_dict'][candidate_keys[0]]
            pass
    # assist query candiates
        for assist_query in list(r['assist_info'].keys()):
            if r['assist_info'][assist_query]['assist_answer'] != max(r['assist_info'][assist_query]['assist_candidates'], key=r['assist_info'][assist_query]['assist_candidates'].get):
                assist_candidate_keys = list(r['assist_info'][assist_query]['assist_candidates'].keys())
                r['assist_info'][assist_query]['assist_candidates'][assist_candidate_keys[0]], r['assist_info'][assist_query]['assist_candidates'][assist_candidate_keys[1]] = r['assist_info'][assist_query]['assist_candidates'][assist_candidate_keys[1]], r['assist_info'][assist_query]['assist_candidates'][assist_candidate_keys[0]]
                
    return res

def auto_produce_balanced_answer(res, origin_candidates_confidence, assist_candidates_confidence, strategy, consistency, greater_than_self):
    for r in res:
        origin_candidates = list(r['candidates_dict'].keys())
        if len(origin_candidates)!=2:
            r['balanced_answer'] = r['pred_ans']
            continue
        if 'assist_info' not in r:
            r['balanced_answer'] = r['pred_ans']
            continue
        if origin_candidates[0] in origin_candidates[1] or origin_candidates[1] in origin_candidates[0]:
            r['balanced_answer'] = r['pred_ans']
            continue
        if r['candidates_dict'][r['pred_ans']] > origin_candidates_confidence:
            r['balanced_answer'] = r['pred_ans']
            continue
        
        if r['is_integration']:
            # empty because need to judge each assist query
            answer_list = []
            if strategy == 'add_first':
                # add the origin prd answer first
                answer_list.extend([r['pred_ans']])
                # make the voting pool
                voting_pool = {}
                voting_pool[r['pred_ans']] = r['candidates_dict'][r['pred_ans']]
                # judge each assist query
                for assist_query in list(r['assist_info'].keys()):
                    if 'rights_alloc_res' not in r['assist_info'][assist_query]:
                        continue
                    if greater_than_self:
                        if r['candidates_dict'][r['pred_ans']] > r['assist_info'][assist_query]['assist_candidates'][r['assist_info'][assist_query]['assist_answer']]:
                            continue
                    if r['assist_info'][assist_query]['assist_candidates'][r['assist_info'][assist_query]['assist_answer']] > assist_candidates_confidence:
                        answer_list.extend(r['assist_info'][assist_query]['assist_query_balanced_answers'])  #  = answer_list + r['assist_info'][assist_query]['assist_query_balanced_answers']
                        rights_alloc_keys = list(r['assist_info'][assist_query]['rights_alloc_res'])
                        for rights_alloc_key in rights_alloc_keys:
                            if rights_alloc_key in voting_pool:
                                voting_pool[rights_alloc_key] += r['assist_info'][assist_query]['rights_alloc_res'][rights_alloc_key]
                            else:
                                voting_pool[rights_alloc_key] = r['assist_info'][assist_query]['rights_alloc_res'][rights_alloc_key]
                        # do not use consistency
                r['voting_pool'] = voting_pool
                r['balanced_answer'] = max(voting_pool, key=voting_pool.get)
                
            if strategy == 'add_after':
                voting_pool = {}
                for assist_query in list(r['assist_info'].keys()):
                    if 'rights_alloc_res' not in r['assist_info'][assist_query]:
                        continue
                    if greater_than_self:
                        if r['candidates_dict'][r['pred_ans']] > r['assist_info'][assist_query]['assist_candidates'][r['assist_info'][assist_query]['assist_answer']]:
                            continue
                    if r['assist_info'][assist_query]['assist_candidates'][r['assist_info'][assist_query]['assist_answer']] > assist_candidates_confidence:
                        # add consistency 
                        answer_list.extend(r['assist_info'][assist_query]['assist_query_balanced_answers']) 
                        rights_alloc_keys = list(r['assist_info'][assist_query]['rights_alloc_res'])

                        for rights_alloc_key in rights_alloc_keys:
                            if rights_alloc_key in voting_pool:
                                voting_pool[rights_alloc_key] += r['assist_info'][assist_query]['rights_alloc_res'][rights_alloc_key]
                            else:
                                voting_pool[rights_alloc_key] = r['assist_info'][assist_query]['rights_alloc_res'][rights_alloc_key]
                if len(answer_list) == 0:
                    answer_list.extend([r['pred_ans']])
                
                r['voting_pool'] = voting_pool
                if len(voting_pool) == 0:
                    r['balanced_answer'] = r['pred_ans']
                else:
                    # judge if it has mutil top 1 answer
                    max_value = max(voting_pool.values())
                    max_value_count = list(voting_pool.values()).count(max_value)
                    if max_value_count > 1:
                        r['balanced_answer'] = r['pred_ans']
                    else:
                        r['balanced_answer'] = max(voting_pool, key=voting_pool.get)

        else:
                pass
        if type(r['balanced_answer']) == 'list':
            print(r['balanced_answer'])
            assert False, 'List Type'
    return res
    pass

def cal_single_acc(prd_ans, gt_ans):
    # implention of the acc cal method of VQA_RAD like
    if prd_ans.lower() == gt_ans.lower():
        return 1
    else:
        return 0
    pass

def best_acc_search(res):
    # input the res
    # search the upper limit of current assist query
    # both the add_firt and the add_after strategy
    
    # add_first
    overall_score_add_first = 0
    for r in res:

            
        current_best_acc = cal_single_acc(r['pred_ans'], r['gt_ans'])
        if 'assist_info' in r:
        # judge each assist query
            for assist_query in list(r['assist_info'].keys()):
                answer_list = []
                answer_list.extend([r['pred_ans']])
                answer_list.extend(r['assist_info'][assist_query]['assist_query_balanced_answers']) 
                counter = Counter(answer_list)
                max_count = counter.most_common(1)[0][1]
                max_count_elements = sum(count == max_count for count in counter.values())
                has_multiple_most_common = max_count_elements > 1
                if has_multiple_most_common:
                    # confusion
                    r['balanced_answer'] = r['pred_ans']
                else:
                    most_common_element = Counter(answer_list).most_common(1)[0][0]
                    r['balanced_answer'] = most_common_element
                # cal the acc current
                current_best_acc = max(current_best_acc, cal_single_acc(r['balanced_answer'], r['gt_ans']))
        else:
            r['balanced_answer'] = r['pred_ans']
        overall_score_add_first = overall_score_add_first + current_best_acc
            
    # add after
    overall_score_add_after = 0
    for r in res:
        if 'assist_info' in r:
            current_best_acc = 0
            
            for assist_query in list(r['assist_info'].keys()):
                answer_list = []
                answer_list.extend(r['assist_info'][assist_query]['assist_query_balanced_answers']) 
                if len(answer_list) == 0:
                    answer_list.extend([r['pred_ans']])
                counter = Counter(answer_list)
                max_count = counter.most_common(1)[0][1]
                max_count_elements = sum(count == max_count for count in counter.values())
                has_multiple_most_common = max_count_elements > 1
                if has_multiple_most_common:
                    answer_list.extend([r['pred_ans']])
                    most_common_element = Counter(answer_list).most_common(1)[0][0]
                    r['balanced_answer'] = most_common_element
                else:
                    most_common_element = Counter(answer_list).most_common(1)[0][0]
                    r['balanced_answer'] = most_common_element
                current_best_acc = max(current_best_acc, cal_single_acc(r['balanced_answer'],r['gt_ans']))
            # print(r['balanced_answer'],r['pred_ans'])
        else:
            r['balanced_answer'] = r['pred_ans']
            current_best_acc = cal_single_acc(r['balanced_answer'],r['gt_ans'])
        overall_score_add_after = overall_score_add_after + current_best_acc
        # print('Current Overall Score: ', overall_score_add_after)
        # assert False
    # print('Current Overall Score: ', overall_score_add_after, 'len: ',len(res))
    print('Add First Best Acc: ', overall_score_add_first/len(res))
    print('Add After Best Acc: ', overall_score_add_after/len(res))
    if type(r['balanced_answer']) == 'list':
        print(r['balanced_answer'])
        assert False, 'List Type'
        
    add_first_best_acc = overall_score_add_first/len(res)
    add_after_best_acc = overall_score_add_after/len(res)
    return add_first_best_acc, add_after_best_acc
    pass
    
def judge_effect(res):
    # positive negative no_effect
    postive_num = 0
    negative_num = 0
    no_effect_num = 0
    
    for r in res:
        # Origin Score
        origin_score = cal_single_acc(r['pred_ans'], r['gt_ans'])
        # Balance Score
        balance_score = cal_single_acc(r['balanced_answer'], r['gt_ans'])
        if origin_score == balance_score:
            no_effect_num = no_effect_num + 1
            r ['judge_effect'] = 'no_effect'
        elif origin_score > balance_score:
            negative_num = negative_num + 1
            r ['judge_effect'] = 'negative'
        elif origin_score < balance_score:
            postive_num = postive_num + 1
            r ['judge_effect'] = 'postive'
        else:
            print(origin_score, balance_score)
            assert False, 'Judge Fall'
        
    return postive_num, negative_num, no_effect_num, res
        
    
          
def filter_prob_same_trend(res):
    # filter the non help assist query
    for r in res:
        for assist_query in list(r['assist_info'].keys()):
            # firstly, judge if the len euqal 2
            if len(list(r['assist_info'][assist_query]['statements_prob_norm'].keys())) != 2:
                continue
            
            # both greater or smaller
            llm_prob_keys = list(r['assist_info'][assist_query]['statements_prob_norm'].keys())
            if r['assist_info'][assist_query]['statements_prob_norm'][llm_prob_keys[0]][0] > r['assist_info'][assist_query]['statements_prob_norm'][llm_prob_keys[0]][1] \
                and r['assist_info'][assist_query]['statements_prob_norm'][llm_prob_keys[1]][0] > r['assist_info'][assist_query]['statements_prob_norm'][llm_prob_keys[1]][1]:
                    r['assist_info'][assist_query]['assist_query_balanced_answers'] = []
                
            if r['assist_info'][assist_query]['statements_prob_norm'][llm_prob_keys[0]][0] < r['assist_info'][assist_query]['statements_prob_norm'][llm_prob_keys[0]][1] \
                and r['assist_info'][assist_query]['statements_prob_norm'][llm_prob_keys[1]][0] < r['assist_info'][assist_query]['statements_prob_norm'][llm_prob_keys[1]][1]:
                    r['assist_info'][assist_query]['assist_query_balanced_answers'] = []
            pass
        
    return res
    pass

def filter_confidence_smaller_than_orgin(res):
    for r in res:
        prob_ori = r['candidates_dict'][list(r['candidates_dict'].keys())[0]]
        
        
        for assist_query in list(r['assist_info'].keys()):    
            # firstly, judge if the len euqal 2
            if len(list(r['assist_info'][assist_query]['statements_prob_norm'].keys())) != 2:
                continue
            r['assist_info'][assist_query]

            pass

def get_args():
    parser = argparse.ArgumentParser(description='step_eval_mutil_process_api_zoo')
    parser.add_argument('--dataset', type=str, default='winoground')    # gqa scienceqa vqa vqa_rad art_vqa winoground a_okvqa
    parser.add_argument('--model_name', type=str, default='deepseek-ai/deepSeek-llm-67b-chat')
    parser.add_argument('--prob2word_type', type=str, default='5')
    parser.add_argument('--prompt_maker_name', type=str, default='only_if_prompt_make')
    parser.add_argument('--output_path', type=str, default='')
    parser.add_argument('--strategy', type=str, default='add_after')
    parser.add_argument('--prompt_version', type=str, default='short_prompt_v1')
    # ['deepseek-ai/deepSeek-llm-67b-chat','deepseek-ai/deepseek-v2-chat','alibaba/Qwen1.5-7B-Chat', 'alibaba/Qwen1.5-14B-Chat', 'alibaba/Qwen1.5-32B-Chat', 'alibaba/Qwen1.5-110B-Chat', 'alibaba/Qwen2-72B-Instruct', 'alibaba/Qwen2-57B-A14B-Instruct', 'alibaba/Qwen2-7B-Instruct']
    # ['glm-4', 'glm-3-turbo']
    return parser.parse_args()

if __name__ == '__main__':
    dt = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    report_feature = 'hard_filter' # loose_filter  hard_filter
    
    # Simplify Del Keys
    del_keys = ['add_caption_statement', 'equal_answer']
    
    # Strategy
    # candidates_confidence: origin answer confidence and assist confidence choose
    select_strategy = ['candidates_confidence',''] # confidence_larger_than_origin
    origin_candidates_confidence = 0.5 # if large than x, then directly use it, start from 0.5
    assist_candidates_confidence = 0.5 # if large than x, then add to the vote stage, start from 0.5
    search_grid = 0.01 # grid search
    strategy = 'add_after' # add_first add_after
    consistency = False
    greater_than_self = False
    best_origin_candidates_confidence = 0
    best_assist_candidates_confidence = 0
    # make output_path 
    file_name_with_extension = os.path.basename(file_path)
    file_name, extension = os.path.splitext(file_name_with_extension)
    output_name = 'wo_tcw_{}_{}_{}_{}.json'.format(file_name, report_feature,strategy,dt)
    output_file_path = os.path.join(analysis_root_dir, output_name)

    args = get_args()
    strategy = args.strategy
    dataset = args.dataset
    model_name = args.model_name
    prob2word_type = args.prob2word_type
    prompt_maker_name = args.prompt_maker_name
    output_path = args.output_path
    prompt_version = args.prompt_version
    # make output_path 
    
    
    file_path = '{}/{}/{}/{}/prob2word_type_{}_prompt_maker_name_{}/step_2.json'.format(output_path,model_name, prompt_version, dataset, prob2word_type, prompt_maker_name)
    output_file_path = '{}/{}/{}/{}/prob2word_type_{}_prompt_maker_name_{}/final.json'.format(output_path,model_name, prompt_version, dataset, prob2word_type, prompt_maker_name)
    
    

    res = json.load(open(file_path, "r"))
    res = process_overflow(res)

    
    # this is the full file result
    for r in res:
        r['equal_answer'] = False
        r['is_integration'] = True
    baseline = evaluate_prd_acc(res)
    
    # This is the filtered result
    r_filter = []
    # to considering the future process, this step use the 
    for r in res:
        can_balance = False
        if 'assist_info' not in r:
            continue
        assist_queries = list(r['assist_info'].keys())
        for assit_query in assist_queries:
            if report_feature == 'loose_filter':
                if 'statements_prob_norm' in r['assist_info'][assit_query] and len(r['assist_info'][assit_query]['statements_prob_norm'])!=0:
                # if 'statements_prob_norm' in r['assist_info'][assit_query]:
                    can_balance = True
                    pass
            if report_feature == 'hard_filter':
                if len(r['assist_info'][assit_query]['assist_query_balanced_answers']) != 0:
                    can_balance = True
                    pass

            pass
        if can_balance:
            r_filter.append(r)
    # print('debug len:',len(r_filter))
    # assert False
    r_filter_baseline = evaluate_prd_acc(r_filter)
    r_max_acc = r_filter_baseline 
    
    print('Full Best Acc Search:')
    full_add_first_best_acc, full_add_after_best_acc = best_acc_search(res)
    print('Filtered Best Acc Search:')
    filtered_add_first_best_acc, filtered_add_after_best_acc = best_acc_search(r_filter)
    # assert False
    max_acc = baseline
    
    
    # Only Save Filter, because we need the filtered comparsion
    r_best_acc_res = r_filter
    
    
    while origin_candidates_confidence <=1:
        tqdm.write(f'Current value: {origin_candidates_confidence}')
        assist_candidates_confidence = 0.5
        while assist_candidates_confidence <=1:
            res = auto_produce_balanced_answer(res, origin_candidates_confidence, assist_candidates_confidence, strategy, consistency, greater_than_self)
            r_res = auto_produce_balanced_answer(r_filter, origin_candidates_confidence, assist_candidates_confidence, strategy, consistency, greater_than_self)
            balanced_acc = evaluate_balance_acc(res)
            r_balanced_acc = evaluate_balance_acc(r_res)
            print(balanced_acc)
            max_acc = max(balanced_acc, max_acc)
            
            if r_max_acc < r_balanced_acc:
                r_best_acc_res = copy.deepcopy(r_res)
                best_assist_candidates_confidence = assist_candidates_confidence
                best_origin_candidates_confidence = origin_candidates_confidence
            r_max_acc = max(r_balanced_acc, r_max_acc)
            assist_candidates_confidence = assist_candidates_confidence + search_grid
            # break
        origin_candidates_confidence = origin_candidates_confidence + search_grid
        # break
        
    print('File Name: ', file_path)
    print('Setting: \n strategy: ', strategy)
    print('\nGrid:', search_grid)
    
    print('Best_assist_candidates_confidence: ', best_assist_candidates_confidence)
    print('Best_origin_candidates_confidence: ', best_origin_candidates_confidence)
    
    print('Full Result Len:', len(res))
    print('Full Result Baseline:', baseline)
    print('Full Best Acc:', max_acc)
    
    print('Filtered queries: ',len(r_filter))
    print('Fitlered Baseline: ', r_filter_baseline)
    print('Fitlered Best Acc: ', r_max_acc)
    
    print('Reproduce Fitlered Best Acc: ', evaluate_balance_acc(r_best_acc_res))
    
    
    # making out put
    # report_dict 
    report_dict = {}
    # add contents
    report_dict['File_Name'] = file_name
    report_dict['Setting_strategy'] = strategy
    report_dict['Grid'] = search_grid
    
    
    report_dict['Best_assist_candidates_confidence'] = best_assist_candidates_confidence
    
    report_dict['Best_origin_candidates_confidence'] = best_origin_candidates_confidence
    
    report_dict['Full Result Len'] = len(res)
    report_dict['Full Result Baseline'] = baseline
    report_dict['Full Best Acc'] = max_acc
    
    report_dict['Full_add_first_search_best_acc'] = full_add_first_best_acc
    report_dict['Full_add_after_search_best_acc'] = full_add_after_best_acc
    
    
    report_dict['Filtered queries'] = len(r_filter)
    report_dict['Fitlered Baseline'] = r_filter_baseline
    report_dict['Fitlered Best Acc'] = r_max_acc
    
    report_dict['Filtered_add_first_search_best_acc'] = filtered_add_first_best_acc
    report_dict['Filtered_add_after_search_best_acc'] = filtered_add_after_best_acc

    # output to json file
    postive_num, negative_num, no_effect_num, r_judge_res = judge_effect(r_best_acc_res)
    report_dict['filter_postive_num'] = postive_num
    report_dict['filter_negative_num'] = negative_num
    report_dict['filter_no_effect_num'] = no_effect_num
    report_dict['r_best_acc_res'] = r_judge_res
    json.dump(report_dict, open(output_file_path, "w"), indent=4, ensure_ascii=False)
    
