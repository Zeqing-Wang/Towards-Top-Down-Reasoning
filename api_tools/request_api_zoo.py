
from retrying import retry
from zhipuai import ZhipuAI
import random

from openai import OpenAI



@retry(stop_max_attempt_number=7)
def request_api_uniform(prompt, model_name, idx):
    SiliconCloud_model_list = [
        'Qwen/Qwen2.5-32B-Instruct',
        'google/gemma-2-27b-it',
        'OpenGVLab/InternVL2-26B',
        'Pro/Qwen/Qwen2-7B-Instruct',
        'Pro/Qwen/Qwen2-1.5B-Instruct',
        'Qwen/Qwen2.5-14B-Instruct',
        'internlm/internlm2_5-20b-chat',
        'Pro/Qwen/Qwen2.5-7B-Instruct'
    ]
    
    zhipu_model_list = ['glm-4', 'glm-3-turbo']
    gpt_model_list = ['gpt-3.5-turbo','gpt-4o-mini']
    if model_name in SiliconCloud_model_list:
        client = OpenAI(api_key="INSERT YOUR KEY HERE", base_url="https://api.siliconflow.cn/v1")

        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {'role': 'user', 'content': prompt}
            ],
            stream=False,
            # seed=1005,
            temperature=0.7,
            max_tokens=512,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
        )
        # print(response.choices[0].message.content)
        return response.choices[0].message.content
        pass
    if model_name in zhipu_model_list:
        client = ZhipuAI(api_key="INSERT YOUR KEY HERE") # 填写您自己的APIKey
        response = client.chat.completions.create(
            model="glm-3-turbo",  # 填写需要调用的模型名称
            messages=[
                {"role": "user", "content": prompt}
            ],
        )
        # print(response.choices[0].message)
        return response.choices[0].message.content
        pass
    if model_name in gpt_model_list:
        gpt_key_list = ['INSERT YOUR KEY HERE']
        api_key = random.choice(gpt_key_list)

        client = OpenAI(api_key=api_key, base_url="INSERT YOUR API SERVER HERE")

        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {'role': 'user', 'content': prompt}
            ],
            stream=False,
            # seed=1005,
            temperature=0.7,
            max_tokens=512,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
        )
        return response.choices[0].message.content
        pass
    
if __name__ == '__main__':
    # test
    test_prompt = 'hello!' 
    model_name = 'gpt-4o-mini'  
    # all model name 
    res = request_api_uniform(test_prompt, model_name, idx=0)
    print(res)
