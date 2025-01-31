import streamlit as st
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

# ------------------------------------------------------------------------------
# Create streamlit app
# ------------------------------------------------------------------------------
def main():
    st.title("Streamlit Demo")

    # Create tab
    tab1, tab2 = st.tabs(["Multimodal Understanding", "Text-to-Image Generation"])

    # Sidebar for image upload and parameter
    with st.sidebar():
        st.header("Image Upload")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

        st.header("Parameters")
        
        # Multimodal understanding parameters
        with st.expander("Multimodal Understanding", expanded=True):
            seed = st.number_input("Seed", min_value=0, value=42, step=1)
            top_p = st.slider("Top-p", min_value=0.0, max_value=1.0, value=0.9, step=0.05)
            temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.1, step=0.05)
        
        # Text-to-Image Generation parameters
        with st.expander("Text-to-Image Settings", expanded=True):
            seed_t2i = st.number_input("Seed", min_value=0, value=12345, step=1)
            cfg_weight = st.slider("CFG Weight", min_value=1.0, max_value=10.0, value=5.0, step=0.5)

    # Main content
    with tab1:
        st.subheader("Ask a question about the image")
        if uploaded_file:
            st.image(uploaded_file, use_column_width=True)
        question = st.text_input("Question", value = "Explain the image")

        if st.button("Chat"):
            if not uploaded_file:
                st.warning("Please upload an image before chatting")
            
            else:
                with st.spinner("Analyzing your image ..."):
                    asnwer = multimodal_understanding(
                        image = uploaded_file,
                        question = question,
                        seed = seed,
                        top_p = top_p,
                        temperature = temperature
                    )
                st.text_area("Response", value = answer, height = -150)

    with tab2:
        st.subheader("Generate image from text")
        prompt = st.text_input("Prompt", value = "A cute baby fox in the autumn leaves, digital arts, cinematic")

        if st.button("Generate Images"):
            with st.spinner("Generating image ... This may take a minute."):
                images = generate_imag(prompt = prompt, seed = seed_t2i, guidance = cfg_weight)
            st.write("Generated Images":)
            cols = st.columns(2)
            idx = 0
            for i in range(2): # 2 rows
                for i in range(2): # 2 cols
                    if idx < len(images):
                        with cols[j]:
                            st.image(images[idx], use_column_width=True)
                    idx += 1

        # Tips / Example Prompt
        with st.expander("Example Prompt"):
            st.write("1. A cyberpunk samurai meditating in a neon-lit Japanese garden, cherry blossoms falling.")
            st.write("2. A magical library with floating books, ethereal lighting, dust particles in the air, hyperrealistic detail.")
            st.write("3. A steampunk-inspired coffee machine with brass gears and copper pipes, Victorian era style, morning light.")

if __name__ == "__main__":
    main()

