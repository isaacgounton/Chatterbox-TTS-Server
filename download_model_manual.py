#!/usr/bin/env python3
"""
Manual model download script for Chatterbox TTS.
Run this to pre-download the model and avoid startup delays.
"""

import os
import sys
import signal
import time
from pathlib import Path

def timeout_handler(signum, frame):
    print("\n❌ Model download timed out after 15 minutes")
    print("This could be due to:")
    print("1. Slow internet connection")
    print("2. Hugging Face servers being overloaded")
    print("3. Network connectivity issues")
    sys.exit(1)

def check_disk_space():
    """Check if we have enough disk space for the model."""
    cache_dir = os.environ.get('HF_HOME', '/app/hf_cache')
    try:
        statvfs = os.statvfs(cache_dir)
        free_space_gb = (statvfs.f_frsize * statvfs.f_bavail) / (1024**3)
        print(f"💾 Available disk space: {free_space_gb:.1f} GB")
        
        if free_space_gb < 5:
            print(f"❌ Insufficient disk space! Need at least 5GB, have {free_space_gb:.1f}GB")
            return False
        return True
    except Exception as e:
        print(f"⚠️  Could not check disk space: {e}")
        return True  # Continue anyway

def main():
    print("🎤 Chatterbox TTS Model Downloader")
    print("=" * 50)
    
    # Check disk space
    if not check_disk_space():
        sys.exit(1)
    
    # Set up timeout (15 minutes)
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(900)  # 15 minutes
    
    try:
        print("📡 Starting model download...")
        print("📊 Model size: ~2-4GB")
        print("⏱️  Timeout: 15 minutes")
        print("🔄 This may take a while on first run...")
        print()
        
        start_time = time.time()
        
        # Import and download
        from chatterbox.tts import ChatterboxTTS
        
        print("🚀 Initializing ChatterboxTTS.from_pretrained()...")
        model = ChatterboxTTS.from_pretrained(device='cpu')
        
        # Cancel timeout
        signal.alarm(0)
        
        elapsed_time = time.time() - start_time
        print(f"\n✅ Model download completed successfully in {elapsed_time:.1f} seconds!")
        print(f"🎯 Model is cached and ready for use")
        
        # Verify model works
        print("🔍 Testing model functionality...")
        test_audio = model.generate("Hello, this is a test.")
        print(f"✅ Model test passed! Generated audio shape: {test_audio.shape}")
        
    except KeyboardInterrupt:
        signal.alarm(0)
        print("\n⚠️  Download interrupted by user")
        sys.exit(1)
        
    except Exception as e:
        signal.alarm(0)
        print(f"\n❌ Model download failed: {e}")
        print("\nTroubleshooting:")
        print("1. Check your internet connection")
        print("2. Verify Hugging Face access")
        print("3. Ensure sufficient disk space (5GB+)")
        print("4. Try running again later")
        sys.exit(1)

if __name__ == "__main__":
    main()
