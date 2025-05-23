# Surgeon Style Fingerprinting via Discrete Diffusion Models

This repository contains code for modeling personalized surgical behavior using discrete diffusion models in a multimodal Vision-Language-Action (VLA) framework. We develop surgeon-specific gesture prediction pipelines conditioned on surgical video, intent prompts, and privacy-aware surgeon embeddings.

## 🔍 Overview

Surgeons exhibit stylistic variations influenced by training and experience. To model these differences, we propose a discrete denoising diffusion model that reconstructs gesture sequences conditioned on:

- Visual features from surgical videos (ResNet)
- Task-level language prompts (BERT)
- Surgeon identity embeddings (learnable or LLM-based)

We evaluate personalization and privacy through gesture prediction accuracy and membership inference attacks.

## 📁 Project Structure

- `preprocessing.ipynb` — JIGSAWS feature extraction and gesture alignment
- `profile_train_w_o_privacy.py` — Baseline with `nn.Embedding` for surgeon ID
- `profile_train_privacy_w_o_skills.py` — Third-party LLM with ID-only prompts
- `profile_train_privacy_skills.py` — Third-party LLM with GRS-conditioned prompts
- `membership_inference_attack.py` — Privacy evaluation using membership inference

## 🧪 Experimental Settings

- Batch size: 32  
- Learning rate: \( 1 \times 10^{-3} \)  
- Epochs: 20  
- Loss: Cross-entropy on denoised gesture tokens  
- Optimizer: Adam  
- Frameworks: PyTorch, SentenceTransformers

## 🔐 Privacy and Fingerprinting

We assess privacy risks by evaluating how easily surgeon embeddings can be used for re-identification. Embedding strategies include:

- **Non-private**: Learnable embeddings
- **Third-party LLM (ID-only)**: Prompts like `"Surgeon ID: 3"`
- **Third-party LLM (ID + GRS)**: Prompts like `"Surgeon ID: 3, skill score: 3.75"`

## 📊 Key Results

We demonstrate that skill-conditioned language embeddings improve gesture prediction while raising privacy concerns under membership inference. This supports the need for balancing personalization and privacy in surgical AI systems.

## 📄 Paper

See [`paper_draft/`](paper_draft/) for the LaTeX source of the accompanying manuscript.

## 📎 Citation

Coming soon.

## 🔗 Project Page

GitHub Repository: [https://github.com/huixin-zhan-ai/Surgeon_style_fingerprinting](https://github.com/huixin-zhan-ai/Surgeon_style_fingerprinting)

---

