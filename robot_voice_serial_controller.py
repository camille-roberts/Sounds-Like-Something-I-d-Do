#!/usr/bin/env python3
"""
Robot Voice Controller - SERIAL VERSION
Direct serial connection to ESP32 for instant response
- Type text, it speaks in robot voice from computer
- Sends commands directly over USB serial (no WiFi delay)
- Instant emotion changes and arm control
"""

import pyttsx3
import serial
import time
import threading

class RobotVoiceControllerSerial:
    def __init__(self, esp32_port: str = "/dev/tty.usbserial-0001"):
        """
        Initialize the robot voice controller with Serial control
        
        Args:
            esp32_port: Serial port for ESP32 (e.g., '/dev/tty.usbserial-0001' on Mac,
                       'COM3' on Windows, '/dev/ttyUSB0' on Linux)
        """
        # Text-to-speech engine
        self.engine = pyttsx3.init()
        self._configure_robot_voice()
        
        # Serial connection to ESP32
        self.serial_port = None
        try:
            self.serial_port = serial.Serial(esp32_port, 115200, timeout=1)
            time.sleep(2)  # Wait for connection to establish
            print(f"✓ Connected to ESP32 on {esp32_port}")
        except Exception as e:
            print(f"✗ Could not connect to ESP32: {e}")
            print("  Make sure ESP32 is plugged in via USB")
            print("  Check the port name with: ls /dev/tty.* (Mac/Linux) or Device Manager (Windows)")
    
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
        self.engine.setProperty('rate', 150)
        self.engine.setProperty('volume', 1.0)
        
        print(f"   Rate: {self.engine.getProperty('rate')}")
        print(f"   Volume: {self.engine.getProperty('volume')}")
    
    def send_serial_command(self, command: str):
        """Send command to ESP32 over serial"""
        if self.serial_port and self.serial_port.is_open:
            try:
                self.serial_port.write(f"{command}\n".encode())
                self.serial_port.flush()  # Ensure immediate send
                # Read response
                time.sleep(0.05)
                if self.serial_port.in_waiting:
                    response = self.serial_port.readline().decode().strip()
                    return response
            except Exception as e:
                print(f"✗ Serial error: {e}")
        return None
    
    def set_emotion(self, emotion: str):
        """Set robot face emotion via serial"""
        print(f"😊 Emotion: {emotion}")
        self.send_serial_command(f"EMOTION:{emotion}")
    
    def wave_arms(self):
        """Wave both arms sequentially via serial"""
        print("👋 Waving arms...")
        self.send_serial_command("WAVE:BOTH")
    
    def wave_left_arm(self):
        """Wave left arm via serial"""
        self.send_serial_command("WAVE:LEFT")
    
    def wave_right_arm(self):
        """Wave right arm via serial"""
        self.send_serial_command("WAVE:RIGHT")
    
    def send_motor_command(self, command: str, speed: int = 150):
        """Send motor command via serial"""
        print(f"🚗 Motor: {command} @ {speed}")
        self.send_serial_command(f"MOTOR:{command},{speed}")
    
    def speak(self, text: str):
        """
        Speak text with robot voice and animations
        Serial version for instant response
        """
        print(f"\n🔊 Speaking: \"{text}\"")
        
        # IMMEDIATELY set to TALK (no network delay)
        self.set_emotion("TALK")
        
        # Small delay to let emotion settle before speaking
        time.sleep(0.1)
        
        # WORKAROUND: Reinitialize engine for each speech (fixes consecutive speech bug)
        try:
            del self.engine
        except:
            pass
        
        self.engine = pyttsx3.init()
        self._configure_robot_voice()
        
        # Start speaking
        self.engine.say(text)
        self.engine.runAndWait()
        
        # Add small delay after speech to ensure completion
        time.sleep(0.1)
        
        # IMMEDIATELY change to happy after speech ends
        self.set_emotion("HAPPY")
        time.sleep(0.5)
        
        # Wave arms
        self.wave_arms()
    
    def list_available_voices(self):
        """Print all available TTS voices"""
        voices = self.engine.getProperty('voices')
        print("\n📢 Available voices:")
        for i, voice in enumerate(voices):
            print(f"  [{i}] {voice.name}")
    
    def change_voice(self, voice_index: int):
        """Change to different voice by index"""
        voices = self.engine.getProperty('voices')
        if 0 <= voice_index < len(voices):
            self.engine.setProperty('voice', voices[voice_index].id)
            print(f"✓ Changed to voice: {voices[voice_index].name}")
        else:
            print(f"✗ Invalid voice index. Available: 0-{len(voices)-1}")
    
    def adjust_speed(self, rate: int):
        """Adjust speech rate"""
        self.engine.setProperty('rate', rate)
        print(f"✓ Speech rate: {rate}")
    
    def cleanup(self):
        """Clean up resources"""
        self.set_emotion("NEUTRAL")
        if self.serial_port and self.serial_port.is_open:
            self.serial_port.close()
        print("\n👋 Controller shutdown complete")


def interactive_mode():
    """Run interactive command-line interface"""
    print("=" * 70)
    print("🤖 ROBOT VOICE CONTROLLER - SERIAL Edition")
    print("=" * 70)
    
    # Get ESP32 serial port
    print("\nCommon serial ports:")
    print("  Mac: /dev/tty.usbserial-* or /dev/tty.SLAB_USBtoUART")
    print("  Linux: /dev/ttyUSB0 or /dev/ttyACM0")
    print("  Windows: COM3, COM4, etc.")
    
    esp32_port = input("\nEnter ESP32 serial port: ").strip()
    if not esp32_port:
        print("No port specified. Exiting.")
        return
    
    controller = RobotVoiceControllerSerial(esp32_port=esp32_port)
    
    if not controller.serial_port:
        print("Failed to connect. Exiting.")
        return
    
    print("\n" + "=" * 70)
    print("COMMANDS:")
    print("  Just type text to speak it (auto: talk → smile → wave)")
    print("  /emotion <name>  - Set face emotion")
    print("  /motor <cmd> <speed> - Motor control")
    print("  /wave            - Wave both arms")
    print("  /speed <rate>    - Adjust speech rate")
    print("  /voices          - List available voices")
    print("  /voice <index>   - Change voice")
    print("  /quit            - Exit")
    print("=" * 70)
    
    try:
        while True:
            user_input = input("\n> ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() == '/quit':
                break
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
            elif user_input.lower() == '/wave':
                controller.wave_arms()
            else:
                # Regular text - speak it
                controller.speak(user_input)
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    finally:
        controller.cleanup()


if __name__ == "__main__":
    interactive_mode()