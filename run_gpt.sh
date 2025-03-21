# gpt-4o-mini  gpt-3.5-turbo

CUDA_VISIBLE_DEVICES=4 python step_eval_mutil_process_api_zoo.py\
 --model_name Pro/Qwen/Qwen2-1.5B-Instruct\
 --prompt_version short_prompt_v1\
 --dataset winoground\
 --assist_query_prompt_path assist_query.prompts\
 --transfor_statement_prompt_path transfor_statement.prompt\
 --probability_prompt_path probability.prompts\
 --output_path \
 --query_file_path test_baseline_with_caption.json


# for winoground
prompt_maker_names=(
    "only_if_prompt_make"
)


prob2word_types=('5')


for prompt_maker_name in "${prompt_maker_names[@]}"
do
    for prob2word_type in "${prob2word_types[@]}"
    do
        CUDA_VISIBLE_DEVICES=0 python test_for_integration_rights_alloction.py \
        --model_name Pro/Qwen/Qwen2-1.5B-Instruct \
        --dataset winoground \
        --prompt_version short_prompt_v1 \
        --output_path  \
        --prob2word_type "$prob2word_type" \
        --prompt_maker_name "$prompt_maker_name"

        CUDA_VISIBLE_DEVICES=0 python auto_metric_vqa_rad_rights_alloction.py \
        --model_name Pro/Qwen/Qwen2-1.5B-Instruct \
        --dataset winoground \
        --prompt_version short_prompt_v1 \
        --output_path  \
        --prob2word_type "$prob2word_type" \
        --strategy 'add_after' \
        --prompt_maker_name "$prompt_maker_name"
    done
done
