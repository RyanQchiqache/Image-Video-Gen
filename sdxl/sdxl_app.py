import streamlit as st
import torch
from diffusers import DiffusionPipeline

# -------------------------
# Load models once (cached)
# -------------------------
@st.cache_resource
def load_pipelines():

    # DO NOT USE torch.set_default_device("cuda")
    # It breaks schedulers and causes numpy/cuda errors.

    base = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
    )
    base.to("cuda")  # move only the model to GPU

    refiner = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-refiner-1.0",
        torch_dtype=torch.float16,
        use_safetensors=True,
        text_encoder_2=base.text_encoder_2,
        vae=base.vae,
        variant="fp16",
    )
    refiner.to("cuda")  # same here

    return base, refiner


# Load both pipelines
base, refiner = load_pipelines()

# -------------------------
# Streamlit UI
# -------------------------
st.markdown("""
    <h1 style='text-align: center; color: #7DF9FF; font-size: 50px;'>
        SDXL Futuristic Generator 
    </h1>
    <p style='text-align: center; color: #A0A0A0; font-size: 18px;'>
        Create high-quality AI images with an ultra-clean interface.
    </p>
""", unsafe_allow_html=True)

with st.container():
    st.write("###  Prompt Settings")
    prompt = st.text_input("Prompt", "A dog jumping from a balcony")
    negative_prompt = st.text_input("Negative Prompt", "blurry, distorted, bad anatomy")

col1, col2, col3 = st.columns(3)

with col1:
    steps = st.slider("Steps", 10, 60, 40)

with col2:
    guidance_scale = st.slider("Guidance (CFG)", 1.0, 15.0, 7.5)

with col3:
    high_noise_frac = st.slider("Refiner Start %", 0.5, 0.95, 0.8)

st.write("###  Resolution")
resolution = st.selectbox(
    "Choose output size",
    ["1024x1024", "768x768", "512x512"],
    index=0,
)

W, H = map(int, resolution.split("x"))

generate = st.button(" Generate Image", use_container_width=True)

if generate:
    with st.spinner(" Generating image..."):

        # -------- Base pipeline → latents --------
        latents = base(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            denoising_end=high_noise_frac,
            output_type="latent",
            height=H,
            width=W,
        ).images

        # -------- Refiner pipeline → final image --------
        image = refiner(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=latents,
            num_inference_steps=int(steps * (1 - high_noise_frac)),
            denoising_start=high_noise_frac,
            guidance_scale=guidance_scale,
        ).images[0]

        # ---------- Display ----------
        st.markdown("<h3 style='text-align:center; color:#7DF9FF;'> Final Output</h3>", unsafe_allow_html=True)
        st.image(image, caption="Generated Image", use_container_width=True)

        # ---------- Save PNG ----------
        from io import BytesIO
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        png_bytes = buffer.getvalue()

        st.download_button(
            label="️ Download PNG",
            data=png_bytes,
            file_name="generated_image.png",
            mime="image/png",
            use_container_width=True
        )

