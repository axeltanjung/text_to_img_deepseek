import torch
from transformers import AutoModelForCausalLM
from janus.models import MultiModalityCasualLM, VLChatProcessor
from janus.utils.io import load_pil_images
import numpy as np
import PIL.Image
import os

model_path = "deepseek-ai/Janus-Pro-7B"
processor = VLChatProcessor.from_pretrained(model_path)
tokenizer = processor.tokenizer

vl_gpt = AutoModelForCausalLM.from_pretrained(
    model_path, trust_remote_code=True
).to(torch.bfloat16).cuda().eval()

conversation = [
    {
        "role": "<User>",
        "content": f"<image_placeholder>\n{question}",
        "images": [image],
    },
    {"role": "<Assistant>", "content": ""},
]

pil_images = load_pil_images(conversation)
prepare_input = processor(
    conversation = conversation, images = pil_images, force_batchify=True
).to(vl_gpt.device)

inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_input)

outputs = vl_gpt.languange_model.generate(
    inputs_embeds=inputs_embeds,
    attention_mask = prepare_input.attention_mask,
    max_new_token = 512,
    pad_token_id = tokenizer.eos_token_id,
    bos_token_id = tokenizer.bos_token_id,
    eos_token_id = tokenizer.eos_token_id,
)

answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)

print(answer)

# Assuming 'processor' is alreay defined
sft_format = processor.apply_sft_template_for_multi_turn_prompts(
    conversation = conversation,
    sft_format = processor.sft_format,
    system_prompt = "",
)
prompt = sft_format + processor.image_start_tag

@torch.inference_mode()
def generate(
    mmgpt,
    processor,
    prompt,
    temperature=1,
    parallel_size = 16,
    cfg_weight = 5,
    image_token_num_per_image = 576,
    img_size = 384,
    patch_size = 16
):
    # 1 Prepare conditional & unconditional tokens
    # 2 Iteratively sample image tokens with CFG
    # 3 Decode to final images and save them
generate(vl_gpt, processor, prompt)