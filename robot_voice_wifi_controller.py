#!/usr/bin/env python3
"""
Robot Voice Controller with WiFi Face + Motor Control
- Type text, it speaks in robot voice from computer
- Sends "TALK" emotion while speaking (animated mouth)
- After speaking, transitions to "SMILE" (HAPPY emotion)
- Waves arms sequentially (left then right)
"""

import pyttsx3
import threading
import queue
import time
import requests
from typing import Optional

class RobotVoiceController:
    def __init__(self, esp32_ip: str = "192.168.1.xxx"):
        """
        Initialize the robot voice controller with WiFi control
        
        Args:
            esp32_ip: IP address of ESP32 robot face (or use 'robotface.local')
        """
        # Text-to-speech engine
        self.engine = pyttsx3.init()
        self._configure_robot_voice()
        
        # ESP32 connection
        self.esp32_ip = esp32_ip
        self.esp32_base_url = f"http://{esp32_ip}"
        
        # Test connection
        self._test_connection()
    
    def _configure_robot_voice(self):
        """Configure TTS for clean robot-like voice (Zarvox-style)"""
        voices = self.engine.getProperty('voices')
        
        # PRIORITY ORDER: Zarvox first, then other robot voices
        priority_order = [
            'zarvox',      # TOP PRIORITY - Classic robot voice
            'trinoids',    # Robot chorus
            'cellos',      # Deep robot
            'bad news',    # Robotic newsreader
            'good news',   # Robotic
            'bells',       # Mechanical
            'boing',       # Bouncy robot
            'bubbles',     # Quirky robot
        ]
        
        best_voice_index = 0
        voice_found = False
        
        # Search in priority order - stops at FIRST match
        for keyword in priority_order:
            for i, voice in enumerate(voices):
                voice_name = voice.name.lower()
                voice_id = voice.id.lower()
                
                if keyword in voice_name or keyword in voice_id:
                    best_voice_index = i
                    voice_found = True
                    print(f"   🎯 Found robot voice: {voice.name}")
                    break
            
            if voice_found:
                break  # Stop searching once we find a match
        
        # If no robot voice found, try to find a female voice (clearer)
        if not voice_found and len(voices) > 1:
            for i, voice in enumerate(voices):
                voice_name = voice.name.lower()
                if ('female' in voice_name or 'zira' in voice_name or 
                    'samantha' in voice_name):
                    best_voice_index = i
                    break
        
        if len(voices) > 0:
            self.engine.setProperty('voice', voices[best_voice_index].id)
            print(f"🤖 Robot voice: {voices[best_voice_index].name}")
        
        # For Zarvox-style voices, use moderate settings
        # These voices already sound robotic, so don't slow them down too much
        self.engine.setProperty('rate', 150)      # Natural pace for robot voices
        self.engine.setProperty('volume', 1.0)    # Max clarity
        
        print(f"   Rate: {self.engine.getProperty('rate')}")
        print(f"   Volume: {self.engine.getProperty('volume')}")
    
    def _test_connection(self):
        """Test connection to ESP32"""
        print(f"\n📡 Testing connection to ESP32 at {self.esp32_ip}...")
        try:
            response = requests.get(f"{self.esp32_base_url}/", timeout=3)
            if response.status_code == 200:
                print("✓ ESP32 connected!")
            else:
                print(f"⚠ ESP32 responded but with code {response.status_code}")
        except Exception as e:
            print(f"✗ Could not connect to ESP32: {e}")
            print("  Make sure ESP32 is powered on and connected to WiFi")
            print("  Update the IP address if needed")
    
    def speak(self, text: str):
        """
        Speak text with robot voice and animations
        This will:
        1. Set face to TALK mode (animated mouth)
        2. Speak the text (switch to happy slightly before it finishes)
        3. Wave arms
        """
        print(f"\n🔊 Speaking: \"{text}\"")
        
        # Start talking animation
        self.set_emotion("TALK")
        
        # Estimate speech duration based on text length and speech rate
        # Rate is words per minute, roughly 150 WPM for our robot voice
        words = len(text.split())
        speech_rate = self.engine.getProperty('rate')  # ~150
        estimated_duration = (words / (speech_rate / 60.0)) 
        
        # Switch to happy 0.5 seconds before speech ends
        early_switch_time = max(0, estimated_duration - .5)
        
        # Start speaking in a non-blocking way
        self.engine.say(text)
        
        # Start a timer to switch emotion early
        def switch_to_happy():
            time.sleep(early_switch_time)
            self.set_emotion("HAPPY")
        
        # Run the timer in background
        timer_thread = threading.Thread(target=switch_to_happy, daemon=True)
        timer_thread.start()
        
        # Wait for speech to complete
        self.engine.runAndWait()
        
        # Force stop any remaining audio buffer
        try:
            self.engine.stop()
        except:
            pass
        
        # Wait a moment for emotion to settle
        time.sleep(0.5)
        
        # Wave arms sequentially
        self.wave_arms()
    
    def set_emotion(self, emotion: str):
        """
        Set robot face emotion
        
        Args:
            emotion: NEUTRAL, HAPPY, SAD, ANGRY, SURPRISE, SLEEPY, CONFUSED, LOVE, TALK
        """
        try:
            url = f"{self.esp32_base_url}/emotion?e={emotion}"
            response = requests.get(url, timeout=2)
            if response.status_code == 200:
                print(f"😊 Emotion: {emotion}")
            else:
                print(f"⚠ Emotion command failed: {response.status_code}")
        except Exception as e:
            print(f"✗ Error setting emotion: {e}")
    
    def wave_arms(self):
        """Wave left arm then right arm sequentially"""
        print("👋 Waving arms...")
        try:
            url = f"{self.esp32_base_url}/wave?arms=BOTH"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                print("✓ Arms waved!")
            else:
                print(f"⚠ Wave command failed: {response.status_code}")
        except Exception as e:
            print(f"✗ Error waving arms: {e}")
    
    def send_motor_command(self, command: str, speed: int = 150):
        """
        Send movement command to motors via ESP32
        
        Args:
            command: FORWARD, BACK, LEFT, RIGHT, STOP
            speed: Motor speed (0-255)
        """
        try:
            url = f"{self.esp32_base_url}/motor?cmd={command}&speed={speed}"
            response = requests.get(url, timeout=2)
            if response.status_code == 200:
                print(f"🚗 Motor: {command} @ {speed}")
            else:
                print(f"⚠ Motor command failed: {response.status_code}")
        except Exception as e:
            print(f"✗ Error sending motor command: {e}")
    
    def execute_choreography(self, move_name: str):
        """
        Execute pre-programmed movement choreography
        
        Args:
            move_name: HAPPY_WIGGLE, SAD_BACKUP, SURPRISE_FORWARD, etc.
        """
        try:
            url = f"{self.esp32_base_url}/move?cmd={move_name}"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                print(f"💃 Choreography: {move_name}")
            else:
                print(f"⚠ Choreography command failed: {response.status_code}")
        except Exception as e:
            print(f"✗ Error executing choreography: {e}")
    
    def speak_with_emotion(self, text: str, emotion: str = "TALK"):
        """
        Speak with a specific emotion (no auto-smile or wave after)
        """
        print(f"\n🔊 Speaking with {emotion}: \"{text}\"")
        self.set_emotion(emotion)
        self.engine.say(text)
        self.engine.runAndWait()
    
    def list_available_voices(self):
        """Print all available TTS voices"""
        voices = self.engine.getProperty('voices')
        print("\n📢 Available voices:")
        for i, voice in enumerate(voices):
            print(f"  [{i}] {voice.name}")
            print(f"      ID: {voice.id}")
            print(f"      Languages: {voice.languages}")
    
    def change_voice(self, voice_index: int):
        """Change to different voice by index"""
        voices = self.engine.getProperty('voices')
        if 0 <= voice_index < len(voices):
            self.engine.setProperty('voice', voices[voice_index].id)
            print(f"✓ Changed to voice: {voices[voice_index].name}")
        else:
            print(f"✗ Invalid voice index. Available: 0-{len(voices)-1}")
    
    def adjust_speed(self, rate: int):
        """Adjust speech rate (50-300, lower = more robotic)"""
        self.engine.setProperty('rate', rate)
        print(f"✓ Speech rate: {rate}")
    
    def cleanup(self):
        """Clean up resources"""
        # Return to neutral
        self.set_emotion("NEUTRAL")
        print("\n👋 Controller shutdown complete")


def interactive_mode():
    """Run interactive command-line interface"""
    print("=" * 70)
    print("🤖 ROBOT VOICE CONTROLLER - WiFi Edition")
    print("=" * 70)
    
    # Get ESP32 IP
    print("\nEnter ESP32 IP address")
    esp32_ip = input("(or press Enter for 'robotface.local'): ").strip()
    if not esp32_ip:
        esp32_ip = "robotface.local"
    
    controller = RobotVoiceController(esp32_ip=esp32_ip)
    
    print("\n" + "=" * 70)
    print("COMMANDS:")
    print("  Just type text to speak it (auto: talk → smile → wave)")
    print("  /emotion <name>     - Set face emotion only")
    print("  /motor <cmd> <spd>  - Motor control (FORWARD, BACK, LEFT, RIGHT, STOP)")
    print("  /move <choreography>- Execute choreography (HAPPY_WIGGLE, etc.)")
    print("  /speed <rate>       - Adjust speech rate (50-300)")
    print("  /voices             - List available voices")
    print("  /voice <index>      - Change voice")
    print("  /test               - Test all emotions")
    print("  /quit               - Exit")
    print("=" * 70)
    
    try:
        while True:
            user_input = input("\n> ").strip()
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input.lower() == '/quit':
                break
                
            elif user_input.lower() == '/test':
                print("Testing all emotions...")
                emotions = ["NEUTRAL", "HAPPY", "SAD", "ANGRY", "SURPRISE", 
                           "SLEEPY", "CONFUSED", "LOVE"]
                for emo in emotions:
                    controller.set_emotion(emo)
                    time.sleep(2)
                controller.set_emotion("NEUTRAL")
                
            elif user_input.lower() == '/voices':
                controller.list_available_voices()
                
            elif user_input.startswith('/voice '):
                try:
                    index = int(user_input.split()[1])
                    controller.change_voice(index)
                except (ValueError, IndexError):
                    print("Usage: /voice <index>")
                    
            elif user_input.startswith('/speed '):
                try:
                    rate = int(user_input.split()[1])
                    controller.adjust_speed(rate)
                except (ValueError, IndexError):
                    print("Usage: /speed <rate>")
                    
            elif user_input.startswith('/emotion '):
                emotion = user_input.split(maxsplit=1)[1].upper()
                controller.set_emotion(emotion)
                
            elif user_input.startswith('/motor '):
                parts = user_input.split()
                if len(parts) >= 2:
                    cmd = parts[1].upper()
                    speed = int(parts[2]) if len(parts) >= 3 else 150
                    controller.send_motor_command(cmd, speed)
                else:
                    print("Usage: /motor <command> [speed]")
                    
            elif user_input.startswith('/move '):
                move = user_input.split(maxsplit=1)[1].upper()
                controller.execute_choreography(move)
                
            else:
                # Regular text - speak it with full animation sequence
                controller.speak(user_input)
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    finally:
        controller.cleanup()


if __name__ == "__main__":
    interactive_mode()
