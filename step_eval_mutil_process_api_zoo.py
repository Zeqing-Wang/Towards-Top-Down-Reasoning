## for 965, only need to get statement and probabily
from api_tools import request_api, request_api_zoo
import json
from tqdm import tqdm
from lavis.models import load_model_and_preprocess, model_zoo
from PIL import Image
import torch
from concurrent.futures import ThreadPoolExecutor
import concurrent
import numpy as np
import os
import argparse
import re
# model_name = 'alibaba/Qwen2-57B-A14B-Instruct' # 'alibaba/Qwen2-57B-A14B-Instruct', 'alibaba/Qwen2-72B-Instruct'

def vqa_answer(model, image, question):
    answers, candidates_dict = model.predict_answers(samples={"image": image, "text_input": question},
                                    answer_list=None,
                                    inference_method="generate",
                                    num_beams=5,
                                    max_len=10,
                                    min_len=1,
                                    num_ans_candidates=128,
                                    prompt='Question: {} Short answer:')
    # only bz = 1
    return answers[0], candidates_dict[0]

def vlm_assist_query(model, image, question):
    # generate method
    x = model.generate(samples={"image": image, "prompt": question},
                       num_captions=2)
    
    
    return x # [0] return should be a list




def get_assist_query_single(dataset, res, assist_query_method, prompt, model, vis_processors, txt_processors,idx):
    
        top_k = 2 # num of top k candidates
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # model, vis_processors, txt_processors = load_model_and_preprocess(name="blip2_t5", model_type="pretrain_flant5xl", is_eval=True, device=device)
        gpt_idx_now = 0
    # for res in tqdm(query_reses):
        try:
            if assist_query_method == 'llm':
                # this branch is for use llm to get the assist query
                if dataset == 'scienceqa':
                    question_line = 'Question: ' + res['question_text']
                else:
                    question_line = 'Question: ' + res['question']

                candidates_dict = res['candidates_dict']
                candidates_answer_list = list(candidates_dict.keys())
                candidate_answers_line = 'Candidate answers:'
                candidates_answer_list = candidates_answer_list[:2]
                caption_line = 'Caption: ' + res['captions'][0]
                for c_canswer in candidates_answer_list:
                    if c_canswer == candidates_answer_list[-1]:
                        candidate_answers_line = candidate_answers_line + ' ' + c_canswer
                    else:
                        candidate_answers_line = candidate_answers_line + ' ' + c_canswer + ','
                # print('answer line::->', candidate_answers_line)
                # get_assist_query_prompt = prompt + '\n\n'+ question_line + '\n\n' + caption_line + '\n\n' + candidate_answers_line + '\n\n' + 'Relevant issues:'
                get_assist_query_prompt = prompt + '\n\n'+ question_line + '\n\n' + caption_line + '\n\n' + candidate_answers_line + '\n\n' + 'Assistive questions:'
                print('final assist prompt:', get_assist_query_prompt)
                res_assist_query = request_api_zoo.request_api_uniform(get_assist_query_prompt, model_name= model_name, idx=0)
                print('geted_res:', res_assist_query)
                
                if res_assist_query == 'timeout':
                    gpt_idx_now = (gpt_idx_now + 1) % 2
                    res_assist_query = request_api_zoo.request_api_uniform(get_assist_query_prompt, model_name= model_name, idx=0)
                    
                # ready to get the re context
                
                # for short version
                
                
                match = re.search(r'<list>(.*?)</list>', res_assist_query)
                res_assist_query = match.group(1)
                # print('after re res:', res_assist_query)
                
                
                
                res_assist_query_eval = eval(res_assist_query)

            elif assist_query_method=='vlm':
                prompt = 'Reasoning Question : Is the banana ripe enough to eat?\n'+ \
                         'Perception Question : Is the banana yellow ?\n' + \
                         'Reasoning Question : Is it cold outside ?\n' + \
                         'Perception Question : Are any people wearing jackets ?\n' + \
                         'Reasoning Question : {}\n' + \
                         'Perception Question : '
                prompt = prompt.format(res['question'])
                print('input assist query prompt is:', prompt)
                if dataset == 'okvqa':
                    image_root_path = '/okvqa/'
                    image_path = image_root_path + res['image']
                if dataset =='vqa':
                    image_root_path = '/vqa_v2/val2014/COCO_val2014_000000'
                    image_id = str(res['image_id'])
                    image_id_filled = image_id.zfill(6)
                    image_name = image_id_filled + '.jpg'
                    image_path = image_root_path + image_name
                if dataset == 'vqa_rad':
                    image_root_path = '/vqa_rad/images/'
                    image_path = image_root_path + res['image']
                if dataset == 'art_vqa':
                    image_root_path = '/art_vqa/SemArt/Images/'
                    image_path = image_root_path + res['image']
                if dataset == 'winoground':
                    image_root_path = '/winoground/images/'
                    image_path = image_root_path + res['image']
                else:
                    assert False, 'no dataset alloc'
                
                raw_image = Image.open(image_path).convert("RGB")
                image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
                res_assist_query_eval = vlm_assist_query(model=model, image=image, question=prompt)
                # print(res_assist_query_eval, type(res_assist_query_eval))
                # assert False
            # image_root_path = '/vqa_v2/val2014/COCO_val2014_000000'  # '/mnt/SSD_4TB/wangzeqing/data/wzq/vqa_task/gqa/images/'
            # image_root_path = '/okvqa/'  #'/mnt/SSD_4TB/wangzeqing/data/wzq/vqa_task/gqa/images/'


            if dataset == 'okvqa':
                image_root_path = '/okvqa/'
                image_path = image_root_path + res['image']
            if dataset =='vqa':
                image_root_path = '/vqa_v2/val2014/COCO_val2014_000000'
                image_id = str(res['image_id'])
                image_id_filled = image_id.zfill(6)
                image_name = image_id_filled + '.jpg'
                image_path = image_root_path + image_name         
            if dataset == 'vqa_rad':
                image_root_path = '/vqa_rad/images/'
                image_path = image_root_path + res['image']
            if dataset == 'art_vqa':
                image_root_path = '/art_vqa/SemArt/Images/'
                image_path = image_root_path + res['image']
            if dataset == 'winoground':
                image_root_path = '/winoground/images/'
                image_path = image_root_path + res['image']
            if dataset =='a_okvqa':
                image_root_path = '/a_okvqa/'
                image_path = image_root_path + res['image_name']
            if dataset =='scienceqa':
                image_root_path = '/ScienceQA/data/test/'
                image_path = image_root_path + res['image_name'] 
            raw_image = Image.open(image_path).convert("RGB")
            image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
                
            assist_info = {}
            for assist_query in res_assist_query_eval:
                question = txt_processors["eval"](assist_query)
                assist_answer, assist_candidates= vqa_answer(model=model, image=image, question=question)
                # print('assist query:', assist_query)
                # print('assist answer:',assist_answer)
                # print('assist candidates:', assist_candidates)
                assist_answers_list = list(assist_candidates.keys())
                assist_answers_list = assist_answers_list[:2]
                
                assist_candidates_top2 = {assist_answers_list[0]:assist_candidates[assist_answers_list[0]],
                                          assist_answers_list[1]:assist_candidates[assist_answers_list[1]]}
                # print(assist_candidates_top2)
                # print(assist_candidates)
                assist_info[assist_query] = {}
                assist_info[assist_query]['assist_answer'] = assist_answer
                
                assist_info[assist_query]['assist_candidates'] = assist_candidates_top2
                
                # trans scores to probability
                assist_candidates_dict = assist_candidates_top2 #res['assist_info'][assist_info_key]['assist_candidates']
                values = list(assist_candidates_dict.values())
                probabilities = np.exp(values) / np.sum(np.exp(values))
                prob_res = {}
                keys = list(assist_candidates_dict.keys())
                for i in range(len(keys)):
                    prob_res[keys[i]] = probabilities[i]
                assist_info[assist_query]['assist_candidates'] = prob_res
                
            res['assist_info'] = assist_info
        except Exception as e:
            print('error in get assist query')
            print(e)
            # continue
        return res
    # return query_reses




def get_statements_json_format_4_single(dataset ,res, prompt, idx):
    # for res in tqdm(query_reses):
        try:
            assist_info = res['assist_info']
            assist_queries = list(assist_info.keys())
            query_candidates = list(res['candidates_dict'].keys())
            for assist_query in assist_queries:
                assist_query_info = assist_info[assist_query]
                assist_query_candidates = list(assist_query_info['assist_candidates'].keys())
                assist_info[assist_query]['statements_info'] = {}
                
                
                # from here get the statement info
                if dataset == 'scienceqa':
                    question_line = 'Question: ' + res['question_text']
                else:
                    question_line = 'Question: ' + res['question']
                answer_line = 'Answers: ' + query_candidates[0] + ', ' + query_candidates[1]
                priori_question_line = 'Priori Question: ' + assist_query
                priori_answer_line = 'Priori Answers: ' +  assist_query_candidates[0] + ',' + assist_query_candidates[1]
                get_statements_prompt = prompt + '\n\n' + question_line + '\n\n' + answer_line + '\n\n' + priori_question_line + '\n\n' + priori_answer_line + '\n\n' +'statements: '
                geted_statements = request_api_zoo.request_api_uniform(prompt=get_statements_prompt, model_name= model_name, idx=idx)
                # print('before re statements:', geted_statements)
                
                # for short version
                match = re.search(r'<convert>(.*?)</convert>|json\s+(.*?)\s*|```json\s+(.*?)\s+```|<convert>(.*?)', geted_statements, re.DOTALL)
                geted_statements = match.group(1)
                # for short version
                
                # print('after re statements:', geted_statements)
                assist_info[assist_query]['statements_info_ori'] = geted_statements
                
                eval_geted_statements = eval(geted_statements)
                for assist_query_answer in assist_query_candidates:
                    assist_query_key = 'Priori Answer: ' + assist_query_answer
                    statement_info_list = []
                    for query_answer in query_candidates:
                        question_key = 'Question Answer: ' + query_answer
                        # assist_info[assist_query]['statements_info_ori'][assist_query_answer] = geted_statements
                        # print('Question key: ', question_key)
                        # print('Assist key: ', assist_query_key)
                        # print('eval_statement: ', eval_geted_statements)
                        statement_info_list.append(eval_geted_statements[question_key][assist_query_key])
                        # print(geted_statements)
                    assist_info[assist_query]['statements_info'][assist_query_answer] = str(statement_info_list)
        except Exception as e:
            print(e)
            print('error in trans statement!!')
        return res




def get_probability_json_format_single_with_caption(res, prompt, idx, with_caption):
        while True:
            try:
                assist_info = res['assist_info']
                assist_queries = list(assist_info.keys())
                query_candidates = list(res['candidates_dict'].keys())
                if with_caption:
                    caption = res['captions'][0]
                for assist_query in assist_queries:
                    assist_answer_candidates = list(assist_info[assist_query]['assist_candidates'].keys())
                    assist_info[assist_query]['statements_prob'] = {}
                    assist_info[assist_query]['statements_prob_norm'] = {}
                    for assist_answer in assist_answer_candidates:
                        # print(assist_info[assist_query])
                        geted_statement = assist_info[assist_query]['statements_info'][assist_answer]
                        if geted_statement[0] == '\'':
                            geted_statement = '['+ geted_statement + ']'
                        eval_geted_statement = eval(geted_statement)
                        # format_temp = 'This is a scene {}. In above scene, {}'.format()
                        # re_format_statement = [caption+','+eval_geted_statement[0], caption+','+eval_geted_statement[1]]
                        if with_caption:
                            re_format_statement = ['This is a scene of '+caption+'. In the above scene, '+eval_geted_statement[0], 
                                                'This is a scene of '+caption+'. In the above scene, '+eval_geted_statement[1]]
                            eval_geted_statement = re_format_statement
                        else:
                            re_format_statement = [eval_geted_statement[0], eval_geted_statement[1]]
                            pass
                        res['add_caption_statement'] = re_format_statement
                        statement_line = 'Statements: ' + str(re_format_statement)
                        # print('statement line:---------: ', statement_line)
                        geted_probability_prompt = prompt + '\n\n' + statement_line + '\n\n' + 'Probability:'
                        # print('geted probability prompt', geted_probability_prompt)
                        idx = (idx + 1) % 4
                        geted_probability = request_api_zoo.request_api_uniform(prompt=geted_probability_prompt, model_name= model_name, idx=idx)
                    
                        # print('before re prob:', geted_probability)
                        # for short version
                        match = re.search(r'<dict>(.*?)</dict>|json\s+(.*?)\s+`', geted_probability, re.DOTALL)
                        geted_probability = match.group(1)
                        # for short version
                        # print('after re prob:', geted_probability)
                    
                    
                    
                        assist_info[assist_query]['statements_prob'][assist_answer] = geted_probability
                        try:
                            eval_geted_probability = eval(geted_probability)
                            sum_probability = eval_geted_probability[eval_geted_statement[0]] + eval_geted_probability[eval_geted_statement[1]]
                            assist_info[assist_query]['statements_prob_norm'][assist_answer] = [eval_geted_probability[eval_geted_statement[0]]/sum_probability, eval_geted_probability[eval_geted_statement[1]]/sum_probability]
                        except Exception as e:
                            continue
                break
            except Exception as e:
                print(e)
                print('error in res', res)
                break
        return res
    



def run_single(dataset, use_equal_answer, with_caption, res, assist_query_method, get_assist_query_prompt, get_statement_prompt, get_probability_prompt,model, vis_processors, txt_processors, idx):
    idx = (idx + 1) % 4
    # check equal answer
    if use_equal_answer:
        if res['equal_answer']:
            return res
    if len(res['candidates_dict'])<2:
        return res
    res = get_assist_query_single(dataset=dataset,res=res, assist_query_method=assist_query_method, prompt=get_assist_query_prompt, model=model, vis_processors=vis_processors, txt_processors=txt_processors, idx=idx)
    # res = get_statements_json_format_single(res=res, prompt=get_statement_prompt, idx=idx)
    res = get_statements_json_format_4_single(dataset = dataset, res=res, prompt=get_statement_prompt, idx=idx)
    # res = get_probability_json_format_single(res=res, prompt=get_probability_prompt, idx=idx)
    res = get_probability_json_format_single_with_caption(with_caption=with_caption, res=res, prompt=get_probability_prompt, idx=idx)
    # run_res.append(res)
    
    # count = count + 1
    return res
    
    
    pass

def run_mutil_process(dataset, use_equal_answer, with_caption, query_reses, max_workers_num, output_path, assist_query_method, get_assist_query_prompt, get_statement_prompt, get_probability_prompt):
    idx = 0
    run_res = []
    geted_res_id = []
    output_freq = 10
    count = 0
    
    # model prepare
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, vis_processors, txt_processors = load_model_and_preprocess(name="blip2_t5", model_type="pretrain_flant5xl", is_eval=True, device=device)
    executor = ThreadPoolExecutor(max_workers=max_workers_num)
    futures = [
        executor.submit(run_single,
                        dataset,
                        use_equal_answer, 
                        with_caption,
                        res,
                        assist_query_method,
                        get_assist_query_prompt, 
                        get_statement_prompt, 
                        get_probability_prompt, 
                        model, 
                        vis_processors, 
                        txt_processors, 
                        idx)
        for res in query_reses if res['question_id'] not in geted_res_id
    ]

    # multi process excute
    for future in tqdm(concurrent.futures.as_completed(futures), total=len(query_reses)):
        idx = (idx + 1) % 4
        idx = 0
        res = future.result()
        if res['question_id'] in geted_res_id:
            continue
        run_res.append(res)
        geted_res_id.append(res['question_id'])
        print('len::::::::::',len(run_res))
        if count % output_freq == 0:
            json.dump(run_res, open(output_path, "w") , indent=4, ensure_ascii=False)
        count = count + 1
    json.dump(run_res, open(output_path, "w"), indent=4, ensure_ascii=False)
    pass


def get_args():
    parser = argparse.ArgumentParser(description='step_eval_mutil_process_api_zoo')
    parser.add_argument('--dataset', type=str, default='winoground')    # gqa scienceqa vqa vqa_rad art_vqa winoground a_okvqa
    parser.add_argument('--model_name', type=str, default='deepseek-ai/deepSeek-llm-67b-chat')
    parser.add_argument('--assist_query_method', type=str, default='llm')   # vlm / llm
    parser.add_argument('--assist_query_prompt_path', type=str, default='')
    parser.add_argument('--with_caption', type=bool, default=True) # False True
    parser.add_argument('--use_equal_answer', type=bool, default=False) # False True
    parser.add_argument('--transfor_statement_prompt_path', type=str, default='')
    parser.add_argument('--probability_prompt_path', type=str, default='')
    parser.add_argument('--query_file_path', type=str, default='')
    parser.add_argument('--output_path', type=str, default='')
    parser.add_argument('--prompt_version', type=str, default='')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    dataset = args.dataset
    model_name = args.model_name
    with_caption = args.with_caption
    use_equal_answer = args.use_equal_answer
    assist_query_method = args.assist_query_method
    assist_query_prompt_path = args.assist_query_prompt_path
    transfor_statement_prompt_path = args.transfor_statement_prompt_path
    probability_prompt_path = args.probability_prompt_path
    query_file_path = args.query_file_path
    output_path = args.output_path
    prompt_version = args.prompt_version

    output_file_path = '{}/{}/{}/{}/step_1.json'.format(output_path, model_name, prompt_version, dataset)
    
    output_folder_path = output_file_path.rsplit('/',maxsplit=1)[0]

    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    queries = json.load(open(query_file_path, "r"))
    print('query len: ', len(queries))
    # assert False
    with open(assist_query_prompt_path) as f:
        assist_query_prompt = f.read().strip()
    with open(transfor_statement_prompt_path) as f:
        statement_prompt = f.read().strip()
    with open(probability_prompt_path) as f:
        probability_prompt = f.read().strip()

    run_mutil_process(dataset=dataset,
                      use_equal_answer = use_equal_answer, 
                      with_caption = with_caption, 
                      query_reses=queries, 
                      max_workers_num=8, 
                      output_path=output_file_path, 
                      assist_query_method = assist_query_method, 
                      get_assist_query_prompt=assist_query_prompt, 
                      get_statement_prompt=statement_prompt, 
                      get_probability_prompt=probability_prompt)
    
    pass