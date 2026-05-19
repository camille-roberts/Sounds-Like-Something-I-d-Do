#!/usr/bin/env python3
"""
Improved Robot Voice Configuration
Cleaner, crisper robotic sound without the breathy quality
"""

import pyttsx3

def configure_clean_robot_voice(engine):
    """
    Configure TTS for clean, crisp robot voice
    - Mechanical but clear
    - Not breathy or whispery
    - Good articulation
    """
    voices = engine.getProperty('voices')
    
    # STRATEGY: Find the clearest, most articulate voice
    # Then slow it down for robotic effect
    
    best_voice_index = 0
    
    # Try to find a female voice (often clearer/less breathy than male)
    # Or a voice with "clear" or "sharp" characteristics
    for i, voice in enumerate(voices):
        voice_name = voice.name.lower()
        
        # Female voices tend to be less breathy
        if 'female' in voice_name or 'zira' in voice_name or 'hazel' in voice_name:
            best_voice_index = i
            break
        # Look for voices known to be clear
        elif 'david' not in voice_name and i > 0:  # David can be breathy
            best_voice_index = i
    
    engine.setProperty('voice', voices[best_voice_index].id)
    
    # KEY SETTINGS FOR CLEAN ROBOT SOUND:
    
    # Rate: Slower = more robotic, but not TOO slow (causes breathiness)
    # Sweet spot: 140-160 for clean robotic sound
    engine.setProperty('rate', 155)
    
    # Volume: Higher volume = less breathy, clearer
    # Set to max for crispness
    engine.setProperty('volume', 1.0)
    
    print(f"🤖 Clean Robot Voice: {voices[best_voice_index].name}")
    print(f"   Rate: {engine.getProperty('rate')} (mechanical but clear)")
    print(f"   Volume: {engine.getProperty('volume')} (maximum clarity)")
    
    return voices[best_voice_index].name


def test_robot_voice_configs():
    """Test different configurations to find the cleanest robot sound"""
    
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    
    test_phrase = "I am a robot. My voice is mechanical but clear."
    
    print("=" * 70)
    print("🤖 CLEAN ROBOT VOICE TESTER")
    print("=" * 70)
    print("\nTesting different configurations for clean robotic sound...\n")
    
    configs = [
        {
            "name": "Clean Robot (Recommended)",
            "rate": 155,
            "volume": 1.0,
            "voice_index": 0,
            "description": "Clear, crisp, mechanical"
        },
        {
            "name": "Sharp Robot",
            "rate": 165,
            "volume": 1.0,
            "voice_index": 0,
            "description": "Faster, very articulate"
        },
        {
            "name": "Deep Robot",
            "rate": 145,
            "volume": 1.0,
            "voice_index": 0,
            "description": "Slower, deeper, less breathy"
        }
    ]
    
    # If multiple voices available, try female voice
    if len(voices) > 1:
        configs.append({
            "name": "Female Robot (Often Clearest)",
            "rate": 155,
            "volume": 1.0,
            "voice_index": 1,
            "description": "Female voices are less breathy"
        })
    
    for i, config in enumerate(configs):
        print(f"\n[{i+1}] {config['name']}")
        print(f"    {config['description']}")
        print(f"    Voice: {voices[config['voice_index']].name}")
        print(f"    Rate: {config['rate']}, Volume: {config['volume']}")
        
        choice = input("    Press Enter to hear this, 's' to skip: ").strip().lower()
        if choice != 's':
            engine.setProperty('voice', voices[config['voice_index']].id)
            engine.setProperty('rate', config['rate'])
            engine.setProperty('volume', config['volume'])
            
            print("    🔊 Speaking...")
            engine.say(test_phrase)
            engine.runAndWait()
    
    print("\n" + "=" * 70)
    print("CUSTOM TUNING")
    print("=" * 70)
    
    while True:
        print("\nWant to try custom settings?")
        print("Current voices available:")
        for i, voice in enumerate(voices):
            print(f"  [{i}] {voice.name}")
        
        try:
            voice_idx = input("\nVoice index (or 'q' to quit): ").strip()
            if voice_idx.lower() == 'q':
                break
            
            voice_idx = int(voice_idx)
            if voice_idx < 0 or voice_idx >= len(voices):
                print(f"Invalid. Use 0-{len(voices)-1}")
                continue
            
            rate = int(input("Rate (try 140-170 for clean robot): ").strip())
            volume = float(input("Volume (0.8-1.0 for less breathy): ").strip())
            
            engine.setProperty('voice', voices[voice_idx].id)
            engine.setProperty('rate', rate)
            engine.setProperty('volume', volume)
            
            print(f"\n🔊 Testing: {voices[voice_idx].name} @ rate={rate}, vol={volume}")
            engine.say(test_phrase)
            engine.runAndWait()
            
            save = input("\nSave these settings? (y/n): ").strip().lower()
            if save == 'y':
                print("\n✓ Use these settings in your robot controller:")
                print(f"  voice_index = {voice_idx}")
                print(f"  rate = {rate}")
                print(f"  volume = {volume}")
                
        except ValueError:
            print("Invalid input. Try again.")
        except KeyboardInterrupt:
            break
    
    print("\n👋 Done testing!")


if __name__ == "__main__":
    test_robot_voice_configs()