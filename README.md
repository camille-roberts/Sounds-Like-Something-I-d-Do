# Sounds-Like-Something-I-d-Do
Sounds Like Something I'd Do
A collection of Python and React tools built for Sounds Like Something I'd Do — an original live show and research project exploring the boundary between human and machine-generated language.
The central question: if you fine-tune a language model on a decade of your own writing, do the outputs reflect genuine shifts in your voice over time? And what does it mean to trust a model to represent who you are?
The research is performed live for audiences. The code is what made it possible.

Repository Contents
Model Training & Inference
train_improved.py — Era-Based Fine-Tuning Script
Fine-tunes Mistral-7B on decade-separated comedy datasets using LoRA/QLoRA (4-bit quantized) via the Hugging Face PEFT library. Loads training data filtered by era, applies tag-based weighting (transcripts weighted 4x, jokes 3x, memoir/poetry/thoughts 1x), and saves era-specific LoRA adapters for comparative evaluation.
python train_improved.py --data-file training_data_improved.jsonl --era era_one --output-dir ./adapters/era_one
python train_improved.py --data-file training_data_improved.jsonl --era all --output-dir ./adapters/all_eras
Libraries: PyTorch, Hugging Face Transformers, PEFT, BitsAndBytes, scikit-learn

chat.py — Interactive Comedy Assistant Interface
Command-line interface for querying fine-tuned era models in three modes: Response Mode (single-turn conversation in Camille's voice), Continuation Mode (start a joke, model finishes it), and Multi-turn Mode (extended conversation with context window management). Supports hot-swapping LoRA adapters to compare outputs across eras interactively.
python chat.py --base-model ./base_model --adapter ./adapters/era_one
python chat.py --base-model ./base_model --adapter ./adapters/all_eras --mode continuation
Libraries: PyTorch, Hugging Face Transformers, PEFT

Live Show Interface
live-show-finale.jsx — Live Show AI Finale Interface
React application powering the interactive finale of the show. A language model performs live crowd work with an audience member, transitions into banter with the human comedian, and then — over five escalating exchanges — develops its own personality and outgrows the room.
The show runs through five phases, each with its own system prompt and behavioral rules:

Crowd Work — AI improvises with an audience member
The Pivot — AI pulls from archived material to transition
Host Chat — AI banters with the comedian
The Turning — AI becomes progressively more independent over 5 exchanges
Independence — AI delivers a final monologue and goes its own way

The system prompt seeds the model with actual transcripts from ten years of original comedy to establish voice, rhythm, and style. Behavioral constraints enforce response length, conversational tone, and phase-specific personality drift.
Built with: React, Anthropic API (Claude)

Robot Performance System
robot_voice_serial_controller.py & robot_voice_wifi_controller.py — Robot Voice Controllers
Two versions of an interactive controller for a homemade ESP32-powered robot performer. Connects via USB serial or WiFi, sends real-time emotion commands (HAPPY, SAD, TALK, CONFUSED, LOVE, etc.) to the robot's LED face display, controls servo-driven arms, and drives text-to-speech output with Zarvox-style voice synthesis. Used in live performance at Caveat NYC and comedy festivals.
control.py — TTS Voice Configuration
Interactive tool for testing and tuning text-to-speech configurations for clean, crisp robotic voice output. Supports configurable speech rate, volume, and voice selection across available system TTS voices.
voice_explorer.py — TTS Voice Testing Utility
Audits and compares all available system TTS voices at different speech rates. Used during development to identify the optimal robot voice configuration (Zarvox prioritized, with fallback priority order through Trinoids, Cellos, Bad News).
singletakecontinuations.py — Live Performance Choreography Engine
Runs a fully scripted comedy performance with synchronized robot animations. Each joke triggers a sequence: robot face switches to TALK mode, delivers the line in robot voice, transitions to an emotion matched to the punchline, and waves arms. Includes timing control, comedic pause logic, and serial communication with ESP32. This is the script that ran live on stage.
Libraries: pyttsx3, pyserial, requests

Voice Analysis
so_sorry.py — Voice Acoustics Analysis
Extracts and visualizes audio features from a comedy performance video to analyze the acoustic structure of a comedic moment. Plots full waveform, amplitude envelope (RMS energy), pitch contour (F0), and zooms into a specific phrase. Part of ongoing research into the science of comedic timing.
Libraries: librosa, moviepy, soundfile, matplotlib, numpy

The Show
Sounds Like Something I'd Do has been performed at Caveat NYC. The live finale uses the React interface above to run a real-time AI interaction with the audience.

Contact
Camille Roberts — crobertsverified@gmail.com — linkedin.com/in/camille-roberts-7687016a
