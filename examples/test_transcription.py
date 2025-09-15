"""
Simple test script to demonstrate audio transcription with the PyTorch ADTOF model.
"""

from example_usage import transcribe_audio_file
import os

def main():
    print("ADTOF Audio Transcription Test")
    print("=" * 35)
    
    audio_path = "test.wav"
    
    if not os.path.exists(audio_path):
        print(f"Audio file not found: {audio_path}")
        return
    
    try:
        # Transcribe the audio file using ADTOF optimized thresholds
        drum_events = transcribe_audio_file(audio_path)  # Uses default ADTOF thresholds
        
        print(f"\n✓ Transcription completed!")
        print(f"Detected drum events:")
        
        total_events = 0
        for class_name, events in drum_events.items():
            print(f"\n{class_name}:")
            if events:
                print(f"  {len(events)} events detected")
                # Show first few events
                for i, (time, confidence) in enumerate(events[:5]):
                    print(f"    {time:6.2f}s (conf: {confidence:.3f})")
                if len(events) > 5:
                    print(f"    ... and {len(events) - 5} more")
                total_events += len(events)
            else:
                print(f"  No events detected")
        
        print(f"\nTotal events: {total_events}")
        
    except Exception as e:
        print(f"Transcription failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
