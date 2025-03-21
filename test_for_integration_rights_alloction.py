import argparse
from api_tools import request_api
import json
from tqdm import tqdm
from lavis.models import load_model_and_preprocess, model_zoo
from PIL import Image
import torch
import logging
import os

def trans_prob2word(prob):
    if prob < 0.2:
        return 'Impossible'
    if prob < 0.4:
        return 'Unlikely'
    if prob < 0.7:
        return 'Possible'
    if prob < 0.9:
        return 'Likely'
    if prob <=1:
        return 'Probable'
    assert False, 'Error In Trans Porb to Word.'

def trans_prob2word_2_type(prob):
    if prob <= 0.5:
        return 'Unlikely'
    if prob <=1:
        return 'Likely'
    assert False, 'Error In Trans Porb to Word.'

def trans_prob2word_enhanced(prob):
    if prob < 0.1:
        return 'Extremely Unlikely'
    elif prob < 0.3:
        return 'Unlikely'
    elif prob < 0.5:
        return 'Somewhat Unlikely'
    elif prob < 0.7:
        return 'Somewhat Likely'
    elif prob < 0.9:
        return 'Likely'
    elif prob <= 1:
        return 'Highly Likely'
    else:
        raise ValueError('Probability out of range [0, 1].')


    
def only_if_prompt_make(prior, describe, question, assist_query):
    # print(prior, describe, question)
    prompt ='Context: If the sky is not blue, the weather is rain. Possible.\n' + \
            'Question: what weather is likely ?\n' + \
            'Short answer: rain\n' +  \
            'Context: {} {}.\n' + \
            'Question: {}\n' + \
            'Short answer: '
    prompt = prompt.format(prior, describe, question)
    # print(prompt)
    return prompt
    pass


            
def vqa_answer(model, image, prompt):
    answers, candidates_dict = model.predict_answers(samples={"image": image, "text_input": prompt},
                                    answer_list=None,
                                    inference_method="generate",
                                    num_beams=5,
                                    max_len=10,
                                    min_len=1,
                                    num_ans_candidates=128,
                                    )
    # only bz = 1
    return answers[0] #, candidates_dict[0]
    pass

def integration(prob2word_type, assist_query_select, dataset ,res, model, vis_processors, txt_processors, integration_strategy, prompt_maker_name):
    prompt_maker = globals()[prompt_maker_name]
    
    for r in tqdm(res):
        r['is_integration'] = False
        if assist_query_select == 'choosen':
            if 'chosen_assist_query' in r:
                assist_query = r['chosen_assist_query']
                data = r['assist_info'][assist_query]
                # eval statements
                for key in data['statements_info']:
                    data['statements_info'][key] = eval(data['statements_info'][key])
                    pass
                extracted_pairs = []
                for key in data["statements_info"]:
                    for statement, probability in zip(data["statements_info"][key], data["statements_prob_norm"][key]):
                        extracted_pairs.append((statement, probability))
                
                # process the image
                if dataset in ['okvqa']:
                    image_root_path = '/okvqa/'  #'/mnt/SSD_4TB/wangzeqing/data/wzq/vqa_task/gqa/images/'
                    img_id = str(r['image'])
                    img_path = image_root_path + img_id
                elif dataset in ['gqa']:
                    img_id = str(r['image_name'])
                    image_root_path = '/gqa/images/'
                    img_path = image_root_path + img_id
                elif dataset in ['snlive']:
                    img_id = str(r['image_name'])
                    image_root_path = '/image/'
                    img_path = image_root_path + img_id
                    pass
                elif dataset in ['vqa']:
                    image_root_path = '/vqa_v2/val2014/COCO_val2014_000000'
                    image_id = str(r['image_id'])
                    image_id_filled = image_id.zfill(6)
                    image_name = image_id_filled + '.jpg'
                    img_path = image_root_path + image_name
                elif dataset in ['vqa_rad']:
                    image_path = '/vqa_rad/images/'
                    image = r['image']
                    img_path = image_path + image
                elif dataset in ['winoground']:
                    image_root_path = '/winoground/images/'
                    img_path = image_root_path + r['image']
                elif dataset in ['art_vqa']:
                    image_root_path = '/art_vqa/SemArt/Images/'
                    img_path = image_root_path + r['image']
                    pass
                raw_image = Image.open(img_path).convert("RGB")
                image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
                
                
                # begin to run extracted pairs
                for prior_pair in extracted_pairs:
                    if dataset in ['okvqa', 'vqa', 'vqa_rad', 'art_vqa']:
                        prior, describe, question = prior_pair[0], trans_prob2word(prior_pair[1]), r['question']
                    if dataset in ['gqa', 'snlive']:
                        prior, describe, question = prior_pair[0], trans_prob2word(prior_pair[1]), r['question_text']
                    # prompt = only_if_prompt_make(prior=prior, describe=describe, question=question)
                    prompt = prompt_maker(prior=prior, describe=describe, question=question, assist_query=assist_query)
                    # print(prompt)
                    prompt = txt_processors["eval"](prompt)
                    integration_answer = vqa_answer(model=model, image=image, prompt=prompt)
                    
                    if integration_answer in r['candidates_dict']:
                        if integration_strategy == 'first':
                            r['balanced_answer'] = [integration_answer]
                            r['is_integration'] = True
                            break
                        if integration_strategy == 'all':
                            
                            if r['is_integration'] == False:
                                
                                r['balanced_answer'] = [integration_answer]
                                # assert False, 'de2'
                                r['is_integration'] = True
                                
                            else:
                                balanced_answer_now = r['balanced_answer']

                                balanced_answer_now.append(integration_answer)

                                r['balanced_answer'] = balanced_answer_now

                                # assert False,'debug'
                            pass
                # assert False
            else: 
                # r['is_integration'] = False
                pass
            # break
        elif assist_query_select == 'no':
            if 'assist_info' not in r.keys():
                continue
            # print('debug 11')
            assist_queries = list(r['assist_info'].keys())
            for assist_query in assist_queries:
            # assist_query = r['chosen_assist_query']
                r['assist_info'][assist_query]['assist_query_balanced_answers'] = []
                try:
                    data = r['assist_info'][assist_query]
                    # eval statements
                    for key in data['statements_info']:
                        data['statements_info'][key] = eval(data['statements_info'][key])
                        pass
                    extracted_pairs = []
                    pairs_condifent = []
                    for key in data["statements_info"]:
                        for statement, probability in zip(data["statements_info"][key], data["statements_prob_norm"][key]):
                            extracted_pairs.append((statement, probability))
                            pairs_condifent.append(r['assist_info'][assist_query]['assist_candidates'][key])
                    # process the image
                    if dataset in ['okvqa']:
                        image_root_path = '/okvqa/' 
                        img_id = str(r['image'])
                        img_path = image_root_path + img_id
                    elif dataset in ['gqa']:
                        img_id = str(r['image_name'])
                        image_root_path = '/gqa/images/'
                        img_path = image_root_path + img_id
                    elif dataset in ['snlive']:
                        img_id = str(r['image_name'])
                        image_root_path = '/image/'
                        img_path = image_root_path + img_id
                        pass
                    elif dataset in ['vqa']:
                        image_root_path = '/vqa_v2/val2014/COCO_val2014_000000'
                        image_id = str(r['image_id'])
                        image_id_filled = image_id.zfill(6)
                        image_name = image_id_filled + '.jpg'
                        img_path = image_root_path + image_name
                    elif dataset in ['vqa_rad']:
                        image_path = '/vqa_rad/images/'
                        image = r['image']
                        img_path = image_path + image
                        pass
                    elif dataset in ['winoground']:
                        image_root_path = '/winoground/images/'
                        img_path = image_root_path + r['image']
                    elif dataset in ['art_vqa']:
                        image_root_path = '/art_vqa/SemArt/Images/'
                        img_path = image_root_path + r['image']
                    raw_image = Image.open(img_path).convert("RGB")
                    image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
                    
                    
                    # begin to run extracted pairs
                    r['assist_info'][assist_query]['rights_alloc_res'] = {}
                    r['assist_info'][assist_query]['rights_alloc_res_mid_result'] = {}
                    idx = 0
                    for prior_pair in extracted_pairs:
                        if dataset in ['okvqa', 'vqa', 'vqa_rad', 'winoground','art_vqa']:
                            if prob2word_type == '5':
                                prior, describe, question = prior_pair[0], trans_prob2word(prior_pair[1]), r['question']
                        if dataset in ['gqa', 'snlive']:
                            prior, describe, question = prior_pair[0], trans_prob2word(prior_pair[1]), r['question_text']
                        # prompt = only_if_prompt_make(prior=prior, describe=describe, question=question)
                        if prompt_maker_name == 'only_if_prompt_make_possibility':
                            describe = 'Possibility : {}'.format(prior_pair[1])
                            pass
                        prompt = prompt_maker(prior=prior, describe=describe, question=question, assist_query=assist_query)
                        print(prompt)
                        # assert False
                        prompt = txt_processors["eval"](prompt)
                        integration_answer = vqa_answer(model=model, image=image, prompt=prompt)
                        
                        if integration_answer in r['candidates_dict']:
                            r['assist_info'][assist_query]['assist_query_balanced_answers'].append(integration_answer)
                            # add to the 
                            if integration_answer in r['assist_info'][assist_query]['rights_alloc_res']:
                                r['assist_info'][assist_query]['rights_alloc_res'][integration_answer] += pairs_condifent[idx]
                            else:
                                r['assist_info'][assist_query]['rights_alloc_res'][integration_answer] = pairs_condifent[idx]
                                
                            r['assist_info'][assist_query]['rights_alloc_res_mid_result'][prior_pair[0]] = {integration_answer:pairs_condifent[idx]}
                        # add the rights alloction
                        idx += 1
                except Exception as e:
                    # print(e)
                    logging.exception(e)
                    pass
                            
    return res

def get_args():
    parser = argparse.ArgumentParser(description='step_eval_mutil_process_api_zoo')
    parser.add_argument('--dataset', type=str, default='winoground')    # gqa scienceqa vqa vqa_rad art_vqa winoground a_okvqa
    parser.add_argument('--model_name', type=str, default='deepseek-ai/deepSeek-llm-67b-chat')
    # ['deepseek-ai/deepSeek-llm-67b-chat','deepseek-ai/deepseek-v2-chat','alibaba/Qwen1.5-7B-Chat', 'alibaba/Qwen1.5-14B-Chat', 'alibaba/Qwen1.5-32B-Chat', 'alibaba/Qwen1.5-110B-Chat', 'alibaba/Qwen2-72B-Instruct', 'alibaba/Qwen2-57B-A14B-Instruct', 'alibaba/Qwen2-7B-Instruct']
    # ['glm-4', 'glm-3-turbo']
    parser.add_argument('--integration_strategy', type=str, default='all')   # all first
    parser.add_argument('--assist_query_select', type=str, default='no')    # no choosen
    parser.add_argument('--prob2word_type', type=str, default='5')  # 2, 3, 5
    parser.add_argument('--prompt_maker_name', type=str, default='only_if_prompt_make')  # only_if_prompt_make_no_exampler only_if_prompt_make , only_if_prompt_make_cot_considering , only_if_prompt_make_cot_consider_question only_if_prompt_make_possibility
    parser.add_argument('--file_name', type=str, default='')
    parser.add_argument('--output_file', type=str, default='')
    parser.add_argument('--output_path', type=str, default='')
    parser.add_argument('--prompt_version', type=str, default='short_prompt_v1')
    return parser.parse_args()


if __name__ == '__main__':

    args = get_args()
    dataset = args.dataset
    model_name = args.model_name
    integration_strategy = args.integration_strategy
    assist_query_select = args.assist_query_select
    prob2word_type = args.prob2word_type
    prompt_maker_name = args.prompt_maker_name
    file_name = args.file_name
    output_file = args.output_file
    output_path = args.output_path
    prompt_version = args.prompt_version

    file_name = '{}/{}/{}/{}/step_1.json'.format(output_path, model_name, prompt_version, dataset)
    output_file = '{}/{}/{}/{}/prob2word_type_{}_prompt_maker_name_{}/step_2.json'.format(output_path,model_name, prompt_version, dataset, prob2word_type, prompt_maker_name)
    
    # ensure the output file path 
    
    output_folder_check = '{}/{}/{}/{}/prob2word_type_{}_prompt_maker_name_{}'.format(output_path,model_name, prompt_version, dataset, prob2word_type, prompt_maker_name)
    if not os.path.exists(output_folder_check):
        os.makedirs(output_folder_check)
    
    
    res = json.load(open(file_name, "r"))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, vis_processors, txt_processors = load_model_and_preprocess(name="blip2_t5", model_type="pretrain_flant5xl", is_eval=True, device=device)
    
    result = integration(prob2word_type = prob2word_type, 
                         assist_query_select = assist_query_select, 
                         dataset = dataset, 
                         res = res, 
                         model = model, 
                         vis_processors = vis_processors, 
                         txt_processors = txt_processors, 
                         integration_strategy = integration_strategy, 
                         prompt_maker_name = prompt_maker_name)
    json.dump(result, open(output_file, "w"), indent=4, ensure_ascii=False)
    
    pass