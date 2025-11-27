# SDXL Streamlit Generator

A **futuristic, GPU‑powered Stable Diffusion XL (SDXL)** image generator built with **Streamlit**, **PyTorch**, and **Diffusers**.

This project provides:

* SDXL Base + Refiner pipeline
* Fast GPU inference
* Futuristic UI (Streamlit)
* Negative prompts, CFG, steps, and resolution controls
* PNG download button

Because SDXL and Diffusers are sensitive to dependency versions, this README explains exactly how to set up the environment correctly to avoid import errors, segmentation faults, and HuggingFace Hub issues.

---

##  Features

* **Stable Diffusion XL** Base + Refiner
* Clean and futuristic UI
* Support for:

  * Prompt / Negative prompt
  * Inference steps
  * CFG scale
  * High-noise refiner handover
  * Multiple resolutions (512–1024)
* **Save output as PNG**
* Compatible with RTX GPUs (16GB+ recommended)

---

##  IMPORTANT: Correct Dependency Versions

Diffusers, HuggingFace Hub, and Transformers must be version‑aligned.

Your working configuration must be **exactly**:

```
diffusers==0.25.0
huggingface_hub==0.19.4
transformers==4.35.0
accelerate==0.21.0
safetensors>=0.4.0
xformers (optional but recommended)
```

These versions ensure:

* `DiffusionPipeline` imports correctly
* SDXL Base + Refiner work
* No "cached_download" import errors
* No NumPy/CUDA device mismatch errors
* No segmentation faults

If you install newer `huggingface_hub` (e.g., 0.35+), `diffusers 0.25.0` will break.

---

## Installation

Create a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install diffusers==0.25.0
pip install huggingface_hub==0.19.4
pip install transformers==4.35.0
pip install accelerate==0.21.0
pip install safetensors
pip install xformers
pip install streamlit
```

---

##  HuggingFace Token

You need a HuggingFace token with access to:

* `stabilityai/stable-diffusion-xl-base-1.0`
* `stabilityai/stable-diffusion-xl-refiner-1.0`

Login:

```bash
huggingface-cli login
```

---

## Running the App

```bash
streamlit run sdxl_app.py
```

Once models download the first time, they are cached in:

```
~/.cache/huggingface/hub
```

Future runs will be instant.

---

## Project Structure

```
.
├── sdxl_app.py
├── README.md
└── requirements.txt
```

---

## Troubleshooting

### "ImportError: cannot import name 'cached_download'"

You installed a new version of `huggingface_hub`.

**Fix:** downgrade:

```bash
pip install huggingface_hub==0.19.4 --force-reinstall
```

### "Segmentation fault" on pipeline load

Cause: wrong versions of Diffusers + HF Hub.
Fix: ensure versions exactly match the list above.

### Very slow model downloads

Install HF Transfer:

```bash
pip install hf-transfer
export HF_HUB_ENABLE_HF_TRANSFER=1
```

### CUDA → NumPy errors

Remove any usage of:

* `torch.set_default_device("cuda")`
* `scheduler.to("cuda")`

Schedulers must stay on CPU.

## Contributing

Pull requests are welcome! If improving UI, adding features, or optimizing performance, feel free to open an issue.


