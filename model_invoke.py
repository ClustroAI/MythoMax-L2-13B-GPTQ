import json

from transformers import AutoModelForCausalLM, AutoTokenizer

model_name_or_path = "TheBloke/MythoMax-L2-13B-GPTQ"
revision = "gptq-8bit-128g-actorder_True"

# To use a different branch, change revision
# For example: revision="main"
model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                             device_map="auto",
                                             trust_remote_code=False,
                                             revision=revision)

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

def invoke(input_text):
    input_json = json.loads(input_text)
    prompt = input_json['prompt']
    user_input = input_json['user_input'] if 'user_input' in input_json else ""

    prompt_template = f'''Below is an instruction that describes a task. Write a response that 
    appropriately completes the request.

    ### Instruction:
    {prompt}

    ### Input:
    {user_input}

    ### Response:

    '''

    # Generator configs
    temperature = input_json['temperature'] if 'temperature' in input_json else 0.7
    top_p = input_json['top_p'] if 'top_p' in input_json else 0.95
    top_k = input_json['top_k'] if 'top_k' in input_json else 40
    max_new_tokens = input_json['max_new_tokens'] if 'max_new_tokens' in input_json else 512

    # inference
    input_ids = tokenizer(prompt_template, return_tensors='pt').input_ids.cuda()
    output = model.generate(
        inputs=input_ids, 
        temperature=temperature, 
        do_sample=True, 
        top_p=top_p, 
        top_k=top_k, 
        max_new_tokens=max_new_tokens
    )
    response = tokenizer.decode(output[0, input_ids.shape[1]:-1], skip_special_tokens=True)

    return response
