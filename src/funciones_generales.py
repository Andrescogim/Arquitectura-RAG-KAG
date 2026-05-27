

def build_prompt(prompt_base, info_adicional):
    
    prompt = prompt_base.format(**info_adicional)
    return prompt
