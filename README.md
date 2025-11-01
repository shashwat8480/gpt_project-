MiniGPT - 124M

A Transformer-based Large Language Model (LLM) built using the GPT-2 architecture with approximately 124 million parameters, fine-tuned for spam detection on the UCI SMS and Email Spam dataset.

🧩 Model Details : 

Architecture: GPT-2 style decoder-only Transformer

Parameters: ~124M

Layers: 12

Hidden Size: 768

Attention Heads: 12

Context Length: 1024 tokens

Tokenizer: Byte Pair Encoding (BPE)

Base Weights: OpenAI GPT-2 (public release)

Fine-tuned Task: Binary text classification (Spam / Not Spam)

⚙️ Features : 

End-to-end training pipeline (preprocessing → fine-tuning → evaluation)

Implements multi-head self-attention and GELU activation

Checkpointing support for resuming training

Compatible with Hugging Face tokenizer and dataset utilities

Supports both text generation and classification inference

Runs entirely locally — no external API required

Modular design for easy experimentation and dataset swapping
