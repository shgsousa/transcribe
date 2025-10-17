import whisperx
from pyannote.audio import Pipeline
import gc
import argparse
import os
import subprocess
import time
import datetime
from pathlib import Path
import tqdm

# --- Environment Setup for Accelerated Downloads ---
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

def assign_speakers_to_segments(diarization_result, transcription_result):
    """
    Assign speakers to transcription segments based on pyannote.audio diarization results.
    """
    segments = transcription_result["segments"]
    
    for segment in segments:
        segment_start = segment["start"]
        segment_end = segment["end"]
        
        # Find the speaker that has the most overlap with the segment
        assigned_speaker = "SPEAKER_UNKNOWN"
        max_overlap = 0
        
        for turn, _, speaker in diarization_result.itertracks(yield_label=True):
            # Check overlap between segment and speaker turn
            overlap_start = max(segment_start, turn.start)
            overlap_end = min(segment_end, turn.end)
            
            if overlap_start < overlap_end:
                overlap_duration = overlap_end - overlap_start
                
                if overlap_duration > max_overlap:
                    max_overlap = overlap_duration
                    assigned_speaker = f"SPEAKER_{speaker}"
        
        segment["speaker"] = assigned_speaker
    
    return transcription_result
def format_time(seconds):
    """Converts seconds to HH:MM:SS,ms or HH:MM:SS.ms format."""
    delta = datetime.timedelta(seconds=seconds)
    hours, remainder = divmod(delta.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = int(delta.microseconds / 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

def format_time_vtt(seconds):
    """Converts seconds to HH:MM:SS.ms VTT format."""
    return format_time(seconds).replace(',', '.')

def generate_output_files(result, base_filename):
    """Generates SRT, VTT, and Markdown files from the transcription result."""
    
    # --- Generate SRT File ---
    srt_path = f"{base_filename}.srt"
    with open(srt_path, "w", encoding="utf-8") as srt_file:
        for i, segment in enumerate(result["segments"]):
            start_time = format_time(segment["start"])
            end_time = format_time(segment["end"])
            speaker = segment.get("speaker", "SPEAKER_??")
            text = segment["text"].strip()
            
            srt_file.write(f"{i + 1}\n")
            srt_file.write(f"{start_time} --> {end_time}\n")
            srt_file.write(f"[{speaker}] {text}\n\n")
    print(f"Generated SRT file: {srt_path}")

    # --- Generate VTT File ---
    vtt_path = f"{base_filename}.vtt"
    with open(vtt_path, "w", encoding="utf-8") as vtt_file:
        vtt_file.write("WEBVTT\n\n")
        for segment in result["segments"]:
            start_time = format_time_vtt(segment["start"])
            end_time = format_time_vtt(segment["end"])
            speaker = segment.get("speaker", "SPEAKER_??")
            text = segment["text"].strip()

            vtt_file.write(f"{start_time} --> {end_time}\n")
            vtt_file.write(f"<v {speaker}>{text}</v>\n\n")
    print(f"Generated VTT file: {vtt_path}")

    # --- Generate Markdown File with Paragraph Breaks ---
    md_path = f"{base_filename}.md"
    with open(md_path, "w", encoding="utf-8") as md_file:
        md_file.write(f"# Transcription of {os.path.basename(base_filename)}\n\n")
        
        last_speaker = None
        last_segment_end = 0.0
        sentence_count = 0
        
        # Paragraph breaking parameters
        pause_threshold = 2.0  # seconds
        sentence_limit = 5
        sentence_enders = ".!?"

        for segment in result["segments"]:
            speaker = segment.get("speaker", "UNKNOWN")
            text = segment['text'].strip()
            if not text:
                continue

            # --- Paragraph breaking logic ---
            # Condition 1: Speaker changes
            if speaker != last_speaker:
                if last_speaker is not None:
                    md_file.write("\n\n")  # End previous speaker's text block
                
                start_time_str = time.strftime('%H:%M:%S', time.gmtime(segment['start']))
                md_file.write(f"## Speaker: {speaker} ({start_time_str})\n")
                
                md_file.write(text)
                sentence_count = sum(text.count(p) for p in sentence_enders)
            
            # Condition 2: Same speaker
            else:
                pause = segment['start'] - last_segment_end
                
                # Break paragraph on long pause or after enough sentences
                break_paragraph = (pause > pause_threshold) or \
                                  (sentence_count >= sentence_limit and any(text.endswith(p) for p in sentence_enders))

                if break_paragraph:
                    md_file.write("\n\n" + text)
                    sentence_count = 0
                else:
                    md_file.write(" " + text)
            
            # Update state for next iteration
            sentence_count += sum(text.count(p) for p in sentence_enders)
            last_speaker = speaker
            last_segment_end = segment['end']
            
    print(f"Generated Markdown file: {md_path}")


def main():
    parser = argparse.ArgumentParser(description="Transcribe a video file using WhisperX.")
    parser.add_argument("--input", required=True, help="Path to the video or audio file.")
    parser.add_argument("--model", default="large-v2", help="Whisper model size (e.g., tiny, base, small, medium, large-v2).")
    parser.add_argument("--language", default=None, help="Language code (e.g., 'en', 'pt'). Leave blank for auto-detect.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for transcription.")
    parser.add_argument("--compute_type", default="int8", help="Compute type for the model (e.g., int8, float32).")
    parser.add_argument("--align_model", default=None, help="Optional path or Hugging Face repo id for the alignment model.")
    parser.add_argument("--align_model_dir", default=None, help="Directory to cache or locate alignment model weights.")
    
    args = parser.parse_args()

    # --- 1. Setup ---
    input_path = args.input
    base_filename, _ = os.path.splitext(input_path)
    audio_path = f"{base_filename}.wav"

    device = "cpu"
    print("Running on CPU.")
    # On CPU, float16 is not supported, switch to a compatible type if user provides it
    if args.compute_type == "float16":
        args.compute_type = "int8"
        print("Switched compute_type from 'float16' to 'int8' for CPU compatibility.")


    # --- 2. Audio Processing ---
    # Use ffmpeg to convert any input file to a standardized 16kHz mono WAV file.
    # The '-vn' flag strips video from video files, and is ignored for audio files.
    print(f"Processing and standardizing audio from '{input_path}'...")
    try:
        subprocess.run([
            "ffmpeg", "-i", input_path, "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", audio_path, "-y"
        ], check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print("Error during ffmpeg audio processing:")
        print(e.stderr)
        return

    # --- 3. Transcription and Diarization ---
    try:
        # Load Whisper model
        print(f"Loading Whisper model '{args.model}'...")
        model = whisperx.load_model(args.model, device, compute_type=args.compute_type, language=args.language)
        
        # Load audio
        audio = whisperx.load_audio(audio_path)
        
        # Transcribe
        print("Transcribing audio (this may take a while)...")
        # Use the underlying faster-whisper model to get a generator and progress info
        segments_generator, info = model.model.transcribe(audio)

        # Use tqdm to display a progress bar based on audio duration
        segments_list = []
        total_duration = round(info.duration, 2)
        last_progress = 0
        with tqdm.tqdm(total=total_duration, desc="Transcribing segments") as pbar:
            for segment in segments_generator:
                segments_list.append(segment)
                # Update progress bar to the end time of the current segment
                pbar.update(round(segment.end - last_progress, 2))
                last_progress = segment.end
            # Ensure the progress bar is full upon completion
            if last_progress < total_duration:
                pbar.update(round(total_duration - last_progress, 2))

        # Reconstruct the result object that the rest of the script expects
        # This is the corrected line: convert named tuples to dictionaries
        result = {"segments": [s._asdict() for s in segments_list], "language": info.language}

        # Align timestamps
        print("Aligning timestamps...")
        align_model_name = args.align_model
        if align_model_name:
            candidate_path = Path(align_model_name)
            if candidate_path.exists():
                align_model_name = str(candidate_path.resolve())
        align_cache_dir = args.align_model_dir
        if align_cache_dir:
            align_cache_dir = str(Path(align_cache_dir).resolve())
        model_a, metadata = whisperx.load_align_model(
            language_code=result["language"],
            device=device,
            model_name=align_model_name,
            model_dir=align_cache_dir,
        )
        result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
        
        # Diarize (identify speakers)
        print("Identifying speakers...")
        hf_token = os.environ.get("HF_TOKEN")
        if hf_token is None:
            print("\n--- HUGGING FACE TOKEN NOT FOUND ---")
            print("This script requires a Hugging Face access token for speaker diarization.")
            print("1. Go to https://huggingface.co/settings/tokens to create a token.")
            print("2. Run the script again after setting the environment variable:")
            print("   In PowerShell: $env:HF_TOKEN = \"your_token_here\"")
            print("-------------------------------------\n")
            return # Exit the function early

        print("Using Hugging Face token to download speaker diarization models.")
        # Use pyannote.audio speaker-diarization-3.1 directly instead of WhisperX wrapper
        diarize_model = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token
        )
        
        diarize_segments = diarize_model(audio_path)
        result = assign_speakers_to_segments(diarize_segments, result)
        
        print("Transcription complete.")
        
    except Exception as e:
        print(f"An error occurred during transcription: {e}")
        return
    finally:
        # Clean up memory
        gc.collect()

    # --- 4. Output Generation ---
    print("Generating output files...")
    generate_output_files(result, base_filename)

    # --- 5. Cleanup ---
    print("Cleaning up temporary audio file...")
    os.remove(audio_path)
    
    print("Process finished successfully.")


if __name__ == "__main__":
    main()

