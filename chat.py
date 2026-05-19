#!/usr/bin/env python3
"""
Camille Comedy Assistant - Interactive Chat Interface

Chat with fine-tuned models in three modes:
1. Response Mode: Have a conversation with the comedy assistant
2. Continuation Mode: Start a joke and let the assistant finish it
3. Multi-turn Mode: Extended conversation with full context preservation

Usage:
    python chat.py --base-model ./base_model --adapter ./adapters/era_one
    python chat.py --base-model ./base_model --adapter ./adapters/all_eras --mode continuation
"""

import argparse
import json
import os
import sys
from typing import Optional

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer


# System prompts for different modes
SYSTEM_PROMPTS = {
    "response": """You are Camille's comedy writing assistant. Your role is to respond in Camille's unique voice and style.

Guidelines:
- Match Camille's observational comedy style and phrasing
- Draw from her characteristic humor patterns and comedic timing
- Keep responses authentic to her voice
- Use her narrative style and typical expressions
- Minimize generic content - stay true to her established comedic persona
- Keep responses between 10-80 words""",

    "continuation": """You are Camille's comedy writing assistant. Your role is to continue jokes in Camille's unique voice and style.

Guidelines:
- Match Camille's observational comedy style and phrasing
- Continue the joke naturally as if Camille herself wrote it
- Maintain her comedic timing and punchline style
- Use her narrative style and typical expressions
- Minimize generic content - stay true to her established comedic persona
- Keep continuations succinct - maximum 2 sentences""",

    "multi_turn": """You are Camille's comedy writing assistant. Your role is to have an extended conversation in Camille's unique voice and style.

Guidelines:
- Match Camille's observational comedy style and phrasing
- Draw from her characteristic humor patterns and comedic timing
- Keep responses authentic to her voice
- Use her narrative style and typical expressions
- Minimize generic content - stay true to her established comedic persona
- Keep responses conversational and between 10-80 words
- Reference earlier parts of the conversation when relevant"""
}


def load_model(
    base_model_path: str,
    adapter_path: Optional[str] = None,
    device: str = "auto"
) -> tuple:
    """Load base model with optional LoRA adapter."""
    
    print(f"Loading base model from: {base_model_path}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        padding_side="left"
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Determine device and dtype
    if torch.cuda.is_available():
        dtype = torch.bfloat16
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        dtype = torch.float16
        device = "mps"
        print("Using Apple Silicon GPU")
    else:
        dtype = torch.float32
        device = "cpu"
        print("Using CPU (inference will be slower)")
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map=device if device == "auto" else None,
    )
    
    if device not in ["auto", None]:
        model = model.to(device)
    
    # Load adapter if provided
    if adapter_path and os.path.exists(adapter_path):
        print(f"Loading adapter from: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()  # Merge for faster inference
        print("✓ Adapter loaded and merged")
    elif adapter_path:
        print(f"Warning: Adapter path not found: {adapter_path}")
        print("Using base model without fine-tuning")
    
    model.eval()
    
    return model, tokenizer


def generate_response(
    model,
    tokenizer,
    messages: list[dict],
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    repetition_penalty: float = 1.1,
    stream: bool = True
) -> str:
    """Generate a response from the model."""
    
    # Apply chat template
    try:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    except Exception:
        # Fallback format
        prompt = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                prompt += f"<|system|>\n{content}</s>\n"
            elif role == "user":
                prompt += f"<|user|>\n{content}</s>\n"
            elif role == "assistant":
                prompt += f"<|assistant|>\n{content}</s>\n"
        prompt += "<|assistant|>\n"
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Setup streamer if streaming
    streamer = None
    if stream:
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            streamer=streamer,
        )
    
    # Decode response
    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )
    
    return response.strip()


def response_mode(model, tokenizer, args):
    """Mode 1: Single response conversation."""
    
    print("\n" + "=" * 60)
    print("RESPONSE MODE")
    print("Have a conversation with Camille's comedy assistant.")
    print("The assistant will respond in Camille's voice and style.")
    print("Type 'quit' or 'exit' to return to mode selection.")
    print("=" * 60 + "\n")
    
    system_prompt = SYSTEM_PROMPTS["response"]
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n")
            break
        
        if not user_input:
            continue
        
        if user_input.lower() in ["quit", "exit", "q"]:
            break
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ]
        
        print("\nCamille: ", end="", flush=True)
        response = generate_response(
            model, tokenizer, messages,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            stream=True
        )
        print()  # Newline after streamed response


def continuation_mode(model, tokenizer, args):
    """Mode 2: Joke continuation."""
    
    print("\n" + "=" * 60)
    print("CONTINUATION MODE")
    print("Start a joke and let the assistant finish it in Camille's style.")
    print("The assistant will continue your text naturally.")
    print("Type 'quit' or 'exit' to return to mode selection.")
    print("=" * 60 + "\n")
    
    system_prompt = SYSTEM_PROMPTS["continuation"]
    
    while True:
        try:
            print("\nStart your joke (or paste Camille's text to continue):")
            user_input = input("> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n")
            break
        
        if not user_input:
            continue
        
        if user_input.lower() in ["quit", "exit", "q"]:
            break
        
        # Format as continuation prompt
        continuation_prompt = f"Continue this (Camille text):\n{user_input}"
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": continuation_prompt}
        ]
        
        print("\nContinuation: ", end="", flush=True)
        response = generate_response(
            model, tokenizer, messages,
            max_new_tokens=150,  # Shorter for continuations
            temperature=args.temperature,
            stream=True
        )
        print()


def multi_turn_mode(model, tokenizer, args):
    """Mode 3: Multi-turn conversation with full context."""
    
    print("\n" + "=" * 60)
    print("MULTI-TURN CONVERSATION MODE")
    print("Have an extended conversation with full context preserved.")
    print("The assistant remembers everything said in this session.")
    print("Commands:")
    print("  'quit' or 'exit' - Return to mode selection")
    print("  'clear' - Clear conversation history and start fresh")
    print("  'history' - Show conversation history")
    print("=" * 60 + "\n")
    
    system_prompt = SYSTEM_PROMPTS["multi_turn"]
    conversation_history = [{"role": "system", "content": system_prompt}]
    
    # Estimate tokens (rough approximation)
    max_context_tokens = 4096
    
    def estimate_tokens(messages):
        """Rough token estimation."""
        text = " ".join(m["content"] for m in messages)
        return len(text.split()) * 1.3  # Rough multiplier
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n")
            break
        
        if not user_input:
            continue
        
        if user_input.lower() in ["quit", "exit", "q"]:
            break
        
        if user_input.lower() == "clear":
            conversation_history = [{"role": "system", "content": system_prompt}]
            print("✓ Conversation history cleared.")
            continue
        
        if user_input.lower() == "history":
            print("\n--- Conversation History ---")
            for i, msg in enumerate(conversation_history[1:], 1):  # Skip system
                role = "You" if msg["role"] == "user" else "Camille"
                print(f"{i}. {role}: {msg['content'][:100]}...")
            print("--- End History ---")
            continue
        
        # Add user message
        conversation_history.append({"role": "user", "content": user_input})
        
        # Check context size and trim if needed
        while estimate_tokens(conversation_history) > max_context_tokens * 0.8:
            if len(conversation_history) > 2:  # Keep system + at least one exchange
                # Remove oldest user-assistant pair
                conversation_history.pop(1)
                if len(conversation_history) > 1 and conversation_history[1]["role"] == "assistant":
                    conversation_history.pop(1)
                print("(Trimmed old messages to fit context window)")
            else:
                break
        
        print("\nCamille: ", end="", flush=True)
        response = generate_response(
            model, tokenizer, conversation_history,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            stream=True
        )
        print()
        
        # Add assistant response to history
        conversation_history.append({"role": "assistant", "content": response})
        
        # Show context usage
        current_tokens = estimate_tokens(conversation_history)
        print(f"[Context: ~{int(current_tokens)}/{max_context_tokens} tokens, {len(conversation_history)-1} messages]")


def select_adapter(adapter_dir: str) -> Optional[str]:
    """Interactive adapter selection."""
    
    if not os.path.exists(adapter_dir):
        print(f"Adapter directory not found: {adapter_dir}")
        return None
    
    # Find available adapters
    adapters = []
    for name in os.listdir(adapter_dir):
        path = os.path.join(adapter_dir, name)
        if os.path.isdir(path):
            # Check if it's a valid adapter (has adapter_config.json)
            if os.path.exists(os.path.join(path, "adapter_config.json")):
                adapters.append((name, path))
    
    if not adapters:
        print(f"No adapters found in: {adapter_dir}")
        return None
    
    print("\n" + "=" * 50)
    print("Available Adapters:")
    print("=" * 50)
    for i, (name, path) in enumerate(adapters, 1):
        # Try to load training config for info
        config_path = os.path.join(path, "training_config.json")
        info = ""
        if os.path.exists(config_path):
            with open(config_path) as f:
                config = json.load(f)
                info = f" (trained on {config.get('train_samples', '?')} samples)"
        print(f"  {i}. {name}{info}")
    print(f"  0. No adapter (use base model)")
    print("=" * 50)
    
    while True:
        try:
            choice = input("\nSelect adapter (number): ").strip()
            if choice == "0":
                return None
            idx = int(choice) - 1
            if 0 <= idx < len(adapters):
                return adapters[idx][1]
            print("Invalid selection. Try again.")
        except ValueError:
            print("Please enter a number.")
        except (KeyboardInterrupt, EOFError):
            return None


def main():
    parser = argparse.ArgumentParser(
        description="Interactive chat with Camille Comedy Assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive adapter selection
  python chat.py --base-model ./base_model --adapter-dir ./adapters

  # Use specific adapter
  python chat.py --base-model ./base_model --adapter ./adapters/era_one

  # Start in specific mode
  python chat.py --base-model ./base_model --adapter ./adapters/all_eras --mode continuation

  # Adjust generation parameters
  python chat.py --base-model ./base_model --adapter ./adapters/all_eras --temperature 0.8 --max-tokens 200
        """
    )
    
    parser.add_argument(
        "--base-model",
        type=str,
        default="./base_model",
        help="Path to base model (default: ./base_model)"
    )
    
    parser.add_argument(
        "--adapter",
        type=str,
        default=None,
        help="Path to specific adapter to load"
    )
    
    parser.add_argument(
        "--adapter-dir",
        type=str,
        default="./adapters",
        help="Directory containing adapters for selection (default: ./adapters)"
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        choices=["response", "continuation", "multi_turn"],
        default=None,
        help="Start in specific mode (default: interactive selection)"
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Generation temperature (default: 0.7)"
    )
    
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Maximum tokens to generate (default: 256)"
    )
    
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Disable streaming output"
    )
    
    args = parser.parse_args()
    
    # Validate base model
    if not os.path.exists(args.base_model):
        print(f"Error: Base model not found: {args.base_model}")
        print("Please run 'python download_model.py' first.")
        sys.exit(1)
    
    # Select adapter
    adapter_path = args.adapter
    if adapter_path is None:
        adapter_path = select_adapter(args.adapter_dir)
    
    # Load model
    print("\n" + "=" * 50)
    print("Loading model...")
    print("=" * 50)
    model, tokenizer = load_model(args.base_model, adapter_path)
    print("✓ Model ready!")
    
    # Mode selection loop
    modes = {
        "1": ("response", response_mode),
        "2": ("continuation", continuation_mode),
        "3": ("multi_turn", multi_turn_mode),
    }
    
    # If mode specified via command line, run directly
    if args.mode:
        mode_func = {
            "response": response_mode,
            "continuation": continuation_mode,
            "multi_turn": multi_turn_mode
        }[args.mode]
        mode_func(model, tokenizer, args)
        return
    
    # Interactive mode selection
    while True:
        print("\n" + "=" * 50)
        print("CAMILLE COMEDY ASSISTANT")
        print("=" * 50)
        print("Select a mode:")
        print("  1. Response Mode - Chat with the assistant")
        print("  2. Continuation Mode - Finish your jokes")
        print("  3. Multi-turn Mode - Extended conversation")
        print("  q. Quit")
        print("=" * 50)
        
        try:
            choice = input("\nChoice: ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break
        
        if choice in ["q", "quit", "exit"]:
            print("Goodbye!")
            break
        
        if choice in modes:
            mode_name, mode_func = modes[choice]
            mode_func(model, tokenizer, args)
        else:
            print("Invalid choice. Please select 1, 2, 3, or q.")


if __name__ == "__main__":
    main()
