#!/usr/bin/env python3
import argparse
import os
import json
import tempfile
from pathlib import Path
import time
import shutil
import requests
import wave
import numpy as np
import asyncio

try:
    from gradio_client import Client, file as gradio_file
    from huggingface_hub import hf_hub_download
except ImportError:
    print("Please install core dependencies: pip install gradio_client huggingface_hub requests")
    exit(1)

try:
    import edge_tts
except ImportError:
    print("For Edge TTS, please install: pip install edge-tts")

try:
    from bs4 import BeautifulSoup
except ImportError:
    print("For HTML/EPUB processing, please install: pip install beautifulsoup4")

try:
    import markdown
    from markdown.extensions.wikilinks import WikiLinkExtension
except ImportError:
    print("For Markdown processing, please install: pip install markdown")

try:
    import pypdfium2 as pdfium
except ImportError:
    print("For PDF processing, please install: pip install pypdfium2")

try:
    from ebooklib import epub
except ImportError:
    print("For EPUB processing, please install: pip install EbookLib")

try:
    import soundfile as sf # Keep for potential future use or if a library implicitly needs it
    from pydub import AudioSegment
    from pydub.playback import play as pydub_play
    import sounddevice as sd
except ImportError:
    print("For audio playback and conversion, please install: pip install soundfile pydub sounddevice")

LLAMA_CPP_AVAILABLE = False
try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    print("INFO: For local GGUF inference, llama-cpp-python is not installed. Run: pip install llama-cpp-python")
    print("      For M1/M2/M3 Mac Metal GPU support: CMAKE_ARGS='-DLLAMA_METAL=on' pip install -U llama-cpp-python --no-cache-dir")
    Llama = None

PIPER_TTS_AVAILABLE = False
try:
    from piper.voice import PiperVoice
    PIPER_TTS_AVAILABLE = True
except ImportError:
    print("INFO: For Piper TTS, the 'piper-tts' package is not installed or its dependencies are missing.")
    print("      Installation steps:")
    print("      1. Ensure 'espeak-ng' (or 'espeak') is installed via your system package manager.")
    print("         (e.g., 'sudo apt-get install espeak-ng' or 'brew install espeak')")
    print("      2. Install the phonemizer backend: 'pip install piper-phonemize-cross'")
    print("      3. Install piper-tts itself: 'pip install piper-tts'")
    PiperVoice = None

# Torch is needed by OuteTTS
TORCH_AVAILABLE = False
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    print("INFO: PyTorch not installed. Some local models (like OuteTTS) may depend on it. Run: pip install torch")
    torch = None


OUTETTS_AVAILABLE = False
try:
    import outetts
    from outetts import (
        Interface as OuteTTSInterface,
        ModelConfig as OuteTTSModelConfig,
        Models as OuteTTSModels,
        Backend as OuteTTSBackend,
        LlamaCppQuantization as OuteTTSLlamaCppQuantization,
        GenerationConfig as OuteTTSGenerationConfig,
        GenerationType as OuteTTSGenerationType,
        SamplerConfig as OuteTTSSamplerConfig
    )
    OUTETTS_AVAILABLE = True
except ImportError:
    print("INFO: For OuteTTS, the 'outetts' library is not installed. Run: pip install outetts")
    print("      For specific backends (e.g., llama.cpp with GPU), refer to OuteTTS documentation for installation.")


# --- Orpheus/SauerkrautTTS Specific Settings ---
ORPHEUS_SAMPLE_RATE = 24000
ORPHEUS_AVAILABLE_VOICES_BASE = ["tara", "leah", "jess", "leo", "dan", "mia", "zac", "zoe"]
ORPHEUS_GERMAN_VOICES = ["jana", "thomas", "max"]
ORPHEUS_DEFAULT_VOICE = "jana"
LM_STUDIO_API_URL_DEFAULT = "http://127.0.0.1:1234/v1/completions"
LM_STUDIO_HEADERS = {"Content-Type": "application/json"}
OLLAMA_API_URL_DEFAULT = "http://localhost:11434/api/generate"

# --- Model Configurations (Cleaned) ---
GERMAN_TTS_MODELS = {
    "edge": {
        "handler_function": "synthesize_with_edge_tts",
        "default_voice_id": "de-DE-KatjaNeural",
        "available_german_voices": ["de-DE-KatjaNeural", "de-DE-ConradNeural", "de-AT-IngridNeural", "de-AT-JonasNeural", "de-CH-LeniNeural", "de-CH-JanNeural"],
        "is_gradio_client": False, "requires_hf_token": False,
        "notes": "Uses edge-tts library. Very reliable for German. Output: MP3."
    },
    "piper_local": {
        "handler_function": "synthesize_with_piper_local",
        "piper_voice_repo_id": "rhasspy/piper-voices",
        "default_voice_id": {
            "model": "de/de_DE/thorsten/medium/de_DE-thorsten-medium.onnx",
            "config": "de/de_DE/thorsten/medium/de_DE-thorsten-medium.onnx.json"
        },
        "available_voices_info": "Many German voices available from rhasspy/piper-voices. Provide model path part for --german-voice-id.",
        "is_gradio_client": False, "requires_hf_token": True, # For model download
        "notes": "Local Piper TTS. Downloads model/voice. Output: WAV."
    },
     "orpheus_german_ft_gguf": {
        "model_repo_id": "lex-au/Orpheus-3b-German-FT-Q8_0.gguf",
        "model_filename": "Orpheus-3b-German-FT-Q8_0.gguf",
        "handler_function": "synthesize_with_orpheus_gguf_local",
        "is_gradio_client": False, "requires_hf_token": True, # For model download
        "default_voice_id": "jana",
        "available_voices": ORPHEUS_GERMAN_VOICES,
        "notes": "Local German Orpheus GGUF (lex-au). Uses llama-cpp-python. Requires user's decoder.py. Output WAV."
    },
    "orpheus_lm_studio": {
        "api_url": LM_STUDIO_API_URL_DEFAULT,
        "gguf_model_name_in_api": "SauerkrautTTS-Preview-0.1", # User specific
        "handler_function": "synthesize_with_orpheus_lm_studio",
        "is_gradio_client": False, "requires_hf_token": False,
        "default_voice_id": "tom", # User specific from logs
        "available_voices": ORPHEUS_GERMAN_VOICES + ORPHEUS_AVAILABLE_VOICES_BASE,
        "notes": "Uses LM Studio API for Orpheus GGUF. Requires decoder.py. Output: WAV. Ensure LM Studio is running and the model name (--gguf-model-name-in-api) matches your LM Studio setup."
    },
    "orpheus_ollama": {
        "ollama_api_url": OLLAMA_API_URL_DEFAULT,
        "ollama_model_name": "orpheus-german-tts:latest", # USER MUST SET THIS or via CLI
        "handler_function": "synthesize_with_orpheus_ollama",
        "is_gradio_client": False, "requires_hf_token": False,
        "default_voice_id": "jana",
        "available_voices": ORPHEUS_GERMAN_VOICES + ORPHEUS_AVAILABLE_VOICES_BASE,
        "notes": "Uses Ollama API for Orpheus GGUF. Requires decoder.py. Output: WAV. Ensure Ollama service is running and model (e.g., 'orpheus-german-tts:latest' via 'ollama pull orpheus-german-tts') is available and correctly named via --ollama-model-name."
    },
    "oute": {
        "handler_function": "synthesize_with_outetts_local",
        "default_voice_id": "path/to/your/german_reference_voice.wav", # USER MUST REPLACE or provide ./german.wav
        "is_gradio_client": False, "requires_hf_token": False,
        "notes": "Local OuteTTS library. Requires a German .wav speaker reference (max 20s) via --german-voice-id or ./german.wav. Uses llama.cpp backend by default. Output: WAV."
    }
}
# --- Placeholder for User's Orpheus Decoder ---
def orpheus_decoder_convert_to_audio_placeholder(multiframe, count):
    print("WARN: Using PLACEHOLDER orpheus_decoder_convert_to_audio. NO ACTUAL AUDIO WILL BE GENERATED for Orpheus GGUF.")
    return None
try:
    from decoder import convert_to_audio as user_orpheus_decoder
    print("INFO: Successfully imported 'convert_to_audio' from user's decoder.py")
except ImportError:
    print("WARN: 'decoder.py' not found or 'convert_to_audio' not in it. Orpheus GGUF audio generation will use a placeholder.")
    user_orpheus_decoder = orpheus_decoder_convert_to_audio_placeholder
except Exception as e:
    print(f"ERROR: Importing from decoder.py failed: {e}. Using placeholder.")
    user_orpheus_decoder = orpheus_decoder_convert_to_audio_placeholder

# --- Orpheus GGUF Specific Helper Functions ---
def orpheus_format_prompt(prompt_text, voice_name, available_voices_list):
    if voice_name not in available_voices_list:
        print(f"WARN: Orpheus voice '{voice_name}' not in known list {available_voices_list}. Using default '{ORPHEUS_DEFAULT_VOICE}'.")
        voice_name = ORPHEUS_DEFAULT_VOICE
    special_start = "<|audio|>"
    special_end = "<|eot_id|>"
    formatted_prompt = f"{voice_name}: {prompt_text}"
    return f"{special_start}{formatted_prompt}{special_end}"

def orpheus_turn_token_into_id(token_string, index):
    token_string = token_string.strip()
    CUSTOM_TOKEN_PREFIX = "<custom_token_"
    if token_string.startswith(CUSTOM_TOKEN_PREFIX) and token_string.endswith(">"):
        try:
            number_str = token_string[len(CUSTOM_TOKEN_PREFIX):-1]
            token_id = int(number_str) - 10 - ((index % 7) * 4096)
            return token_id
        except ValueError:
            return None
    return None

# --- Common Orpheus GGUF Token Stream Processing and Decoding Pipeline (for local llama.cpp & Ollama) ---
def _orpheus_master_token_processor_and_decoder(raw_token_text_generator, output_file_wav=None):
    print("DEBUG: Orpheus Master Processor - Starting token processing.")
    all_audio_data = bytearray()
    wav_file = None
    if output_file_wav:
        output_p = Path(output_file_wav)
        output_p.parent.mkdir(parents=True, exist_ok=True)
        try:
            wav_file = wave.open(str(output_p), "wb")
            wav_file.setnchannels(1); wav_file.setsampwidth(2); wav_file.setframerate(ORPHEUS_SAMPLE_RATE)
        except Exception as e: print(f"ERROR: Orpheus Master Processor - Could not open WAV file {output_file_wav}: {e}"); wav_file = None

    token_buffer = []; token_count_for_decoder = 0; audio_segment_count = 0; id_conversion_idx = 0

    for text_chunk in raw_token_text_generator:
        # Removed VERY_VERBOSE_DEBUG print of raw LLM chunk here
        current_pos = 0
        while True:
            start_custom = text_chunk.find("<custom_token_", current_pos)
            if start_custom == -1: break
            end_custom = text_chunk.find(">", start_custom)
            if end_custom == -1: break

            custom_token_str = text_chunk[start_custom : end_custom+1]
            token_id = orpheus_turn_token_into_id(custom_token_str, id_conversion_idx)

            if token_id is not None and token_id > 0:
                token_buffer.append(token_id)
                token_count_for_decoder += 1
                id_conversion_idx += 1
                if token_count_for_decoder % 7 == 0 and token_count_for_decoder > 27:
                    buffer_to_process = token_buffer[-28:]
                    audio_chunk_bytes = user_orpheus_decoder(buffer_to_process, token_count_for_decoder)
                    if audio_chunk_bytes and isinstance(audio_chunk_bytes, bytes) and len(audio_chunk_bytes) > 0:
                        all_audio_data.extend(audio_chunk_bytes)
                        audio_segment_count +=1
                        if wav_file:
                            try: wav_file.writeframes(audio_chunk_bytes)
                            except Exception as e: print(f"ERROR: Orpheus Master Processor - Failed to write frames to WAV: {e}")
            current_pos = end_custom + 1

    if wav_file:
        try: wav_file.close(); print(f"INFO: Orpheus Master Processor - Audio saved to {output_file_wav}")
        except Exception as e: print(f"ERROR: Orpheus Master Processor - Failed to close WAV file: {e}")

    if not all_audio_data: print("WARN: Orpheus Master Processor - No audio data was generated by the decoder. Check verbose logs for LLM output. Ensure 'decoder.py' is correct and the GGUF model is producing <custom_token_...> sequences.")
    duration_frames = len(all_audio_data) // 2 # 2 bytes per frame (16-bit mono)
    duration_seconds = duration_frames / ORPHEUS_SAMPLE_RATE
    print(f"INFO: Orpheus Master Processor - Processed {audio_segment_count} audio segments. Total duration: {duration_seconds:.2f}s.")
    return bytes(all_audio_data)

# --- Text Extraction Functions ---
def extract_text_from_txt(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f: return f.read()
    except Exception as e: print(f"ERROR: reading TXT file {filepath}: {e}"); return None
def extract_text_from_md(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f: md_text = f.read()
        html = markdown.markdown(md_text, extensions=[WikiLinkExtension()])
        soup = BeautifulSoup(html, "html.parser"); return soup.get_text()
    except Exception as e: print(f"ERROR: reading Markdown file {filepath}: {e}"); return None
def extract_text_from_html(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f: soup = BeautifulSoup(f, "html.parser")
        for script_or_style in soup(["script", "style"]): script_or_style.decompose()
        return soup.get_text(separator=' ', strip=True)
    except Exception as e: print(f"ERROR: reading HTML file {filepath}: {e}"); return None
def extract_text_from_pdf(filepath):
    try:
        text = ""
        doc = pdfium.PdfDocument(filepath)
        for i in range(len(doc)):
            page = doc.get_page(i); textpage = page.get_textpage()
            text += textpage.get_text_range() + "\n"; textpage.close(); page.close()
        doc.close(); return text
    except Exception as e: print(f"ERROR: reading PDF file {filepath}: {e}"); return None
def extract_text_from_epub(filepath):
    try:
        book = epub.read_epub(filepath); text_parts = []
        for item in book.get_items_of_type(epub.ITEM_DOCUMENT):
            soup = BeautifulSoup(item.get_content(), "html.parser")
            for script_or_style in soup(["script", "style"]): script_or_style.decompose()
            text_parts.append(soup.get_text(separator=' ', strip=True))
        return "\n".join(text_parts)
    except Exception as e: print(f"ERROR: reading EPUB file {filepath}: {e}"); return None
def get_text_from_input(input_text, input_file):
    if input_text: return input_text
    if input_file:
        ext = Path(input_file).suffix.lower()
        if ext == '.txt': return extract_text_from_txt(input_file)
        elif ext == '.md': return extract_text_from_md(input_file)
        elif ext in ['.html', '.htm']: return extract_text_from_html(input_file)
        elif ext == '.pdf': return extract_text_from_pdf(input_file)
        elif ext == '.epub': return extract_text_from_epub(input_file)
        else: print(f"ERROR: Unsupported file type: {ext}"); return None
    return None

# --- Audio Handling ---
def save_audio(audio_data_or_path, output_filepath, source_is_path=False, input_format=None):
    try:
        output_filepath = Path(output_filepath); output_filepath.parent.mkdir(parents=True, exist_ok=True)
        target_format = output_filepath.suffix[1:].lower() if output_filepath.suffix else "mp3"
        if source_is_path:
            source_path = Path(audio_data_or_path)
            if not source_path.exists(): print(f"ERROR: Source audio path does not exist: {source_path}"); return
            if source_path.suffix.lower() == f".{target_format}": shutil.copyfile(source_path, output_filepath)
            else: AudioSegment.from_file(source_path).export(output_filepath, format=target_format)
        else: # bytes
            from io import BytesIO
            fmt = input_format if input_format else "wav"
            if fmt == "wav_bytes": fmt = "wav"
            if fmt == "pcm_s16le":
                # For raw PCM, we assume it's already in the correct format to be written to a WAV
                # if ORPHEUS_SAMPLE_RATE is used. Pydub might struggle without more info.
                # It's safer to create a WAV BytesIO object first for pydub if format is truly raw.
                # However, since Orpheus processing yields WAV-compatible bytes:
                audio_segment = AudioSegment(data=audio_data_or_path, sample_width=2, frame_rate=ORPHEUS_SAMPLE_RATE, channels=1)
            else:
                audio_segment = AudioSegment.from_file(BytesIO(audio_data_or_path), format=fmt)
            audio_segment.export(output_filepath, format=target_format)
        print(f"INFO: Audio saved to {output_filepath}")
    except Exception as e: print(f"ERROR: saving audio to {output_filepath}: {e}")

def play_audio(audio_path_or_data, is_path=True, input_format=None, sample_rate=None):
    try:
        if is_path:
            if not Path(audio_path_or_data).exists(): print(f"ERROR: Audio file for playback does not exist: {audio_path_or_data}"); return
            pydub_play(AudioSegment.from_file(audio_path_or_data))
        else: # bytes
            from io import BytesIO
            fmt = input_format if input_format else "wav"
            if fmt == "wav_bytes": fmt = "wav"

            if fmt == "pcm_s16le" and sample_rate:
                if 'sd' not in globals(): print("ERROR: sounddevice not available for pcm playback."); return
                audio_np_array = np.frombuffer(audio_path_or_data, dtype=np.int16)
                sd.play(audio_np_array, samplerate=sample_rate, blocking=True)
            else:
                if 'AudioSegment' not in globals() or 'pydub_play' not in globals():
                    print("ERROR: pydub not available for playback."); return
                pydub_play(AudioSegment.from_file(BytesIO(audio_path_or_data), format=fmt))
        print(f"INFO: Playback finished.")
    except Exception as e: print(f"ERROR: playing audio: {e}")

# --- TTS Handler Functions ---

async def synthesize_with_edge_tts_async(text, voice_id, output_file_path):
    communicate = None
    try:
        communicate = edge_tts.Communicate(text, voice_id)
        await communicate.save(output_file_path)
        return output_file_path
    except Exception as e:
        print(f"ERROR: during EdgeTTS synthesis: {e}")
        return None

def synthesize_with_edge_tts(model_config, text, german_voice_id_override, model_params_override, output_file, play_direct):
    try: import asyncio
    except ImportError: print("ERROR: EdgeTTS requires asyncio."); return
    voice_id = german_voice_id_override if german_voice_id_override else model_config.get("default_voice_id", "de-DE-KatjaNeural")
    temp_mp3_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmpfile: temp_mp3_path = tmpfile.name
        
        # Get existing event loop or create a new one
        try:
            loop = asyncio.get_event_loop_policy().get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        except RuntimeError: # No current event loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        success_path = loop.run_until_complete(synthesize_with_edge_tts_async(text, voice_id, temp_mp3_path))
        # Do not close the loop if it was pre-existing and running. Only close if newly created for this.
        # For simplicity, let's assume we always create one if it's not running (Python 3.7+ default behavior for run_until_complete)
        # or let the user manage their loop if they run this within a larger async context.

        if success_path:
            target_output_file = Path(output_file).with_suffix(".mp3") if output_file else None
            if target_output_file: save_audio(success_path, target_output_file, source_is_path=True)
            if play_direct: play_audio(success_path, is_path=True)
        else: print("ERROR: EdgeTTS synthesis failed.")
    except Exception as e: print(f"ERROR: initializing EdgeTTS or running asyncio loop: {e}")
    finally:
        if temp_mp3_path and os.path.exists(temp_mp3_path):
            try: os.remove(temp_mp3_path)
            except Exception as e_del: print(f"WARN: Could not delete temp file {temp_mp3_path}: {e_del}")

def synthesize_with_orpheus_gguf_local(model_config, text, german_voice_id_override, model_params_override, output_file, play_direct):
    if not LLAMA_CPP_AVAILABLE :
        print("ERROR: llama-cpp-python not available. Skipping Orpheus GGUF local synthesis.")
        return
    print(f"DEBUG: Orpheus GGUF Local - Text: '{text[:50]}...', Voice: {german_voice_id_override or model_config['default_voice_id']}")
    model_repo_id = model_config["model_repo_id"]; model_filename = model_config["model_filename"]
    model_cache_dir = Path.home() / ".cache" / "german_tts_script_models"; model_cache_dir.mkdir(parents=True, exist_ok=True)
    local_model_path = model_cache_dir / model_filename
    if not local_model_path.exists():
        print(f"INFO: Orpheus GGUF - Downloading {model_filename} from {model_repo_id}...")
        try: hf_hub_download(repo_id=model_repo_id, filename=model_filename, local_dir=model_cache_dir, token=os.getenv("HF_TOKEN"), repo_type="model")
        except Exception as e: print(f"ERROR: Orpheus GGUF - Download failed for {model_repo_id}/{model_filename}: {e}"); return
    else: print(f"INFO: Orpheus GGUF - Model found: {local_model_path}")

    voice = german_voice_id_override or model_config.get('default_voice_id', ORPHEUS_DEFAULT_VOICE)
    available_voices = model_config.get('available_voices', ORPHEUS_GERMAN_VOICES + ORPHEUS_AVAILABLE_VOICES_BASE)
    formatted_prompt = orpheus_format_prompt(text, voice, available_voices)

    cli_params = json.loads(model_params_override) if model_params_override else {}
    temperature = cli_params.get("temperature", 0.5); top_p = cli_params.get("top_p", 0.9)
    max_tokens_gen = cli_params.get("max_tokens", 3072); repetition_penalty = cli_params.get("repetition_penalty", 1.1)
    n_gpu_layers = cli_params.get("n_gpu_layers", -1 if ("arm64" in os.uname().machine.lower() or "aarch64" in os.uname().machine.lower()) else 0)

    llm = None
    try:
        llm = Llama(model_path=str(local_model_path), verbose=False, n_gpu_layers=n_gpu_layers, n_ctx=2048, logits_all=True)
        print(f"INFO: Orpheus GGUF - Model loaded: {local_model_path}")
    except Exception as e: print(f"ERROR: Orpheus GGUF - Load failed for {local_model_path}: {e}"); return

    def _llama_cpp_text_stream_generator():
        print(f"DEBUG: Orpheus GGUF Local - llama.cpp prompt: {formatted_prompt}")
        stream = llm.create_completion(prompt=formatted_prompt, max_tokens=max_tokens_gen, temperature=temperature, top_p=top_p, repeat_penalty=repetition_penalty, stream=True)
        full_raw_output_for_debug = ""
        for output_chunk in stream:
            token_text = output_chunk['choices'][0]['text']
            full_raw_output_for_debug += token_text
            yield token_text
        # Removed VERY_VERBOSE_DEBUG of full raw output here
        print(f"DEBUG: Orpheus GGUF Local - Full raw output (first 200 chars if long): '{full_raw_output_for_debug[:200]}{'...' if len(full_raw_output_for_debug)>200 else ''}'")
        print("DEBUG: Orpheus GGUF Local - llama-cpp-python stream finished.")


    effective_output_file_wav = Path(output_file).with_suffix(".wav") if output_file else None
    audio_bytes = _orpheus_master_token_processor_and_decoder(_llama_cpp_text_stream_generator(),
                                                              output_file_wav=str(effective_output_file_wav) if effective_output_file_wav else None)
    if audio_bytes:
        if play_direct:
            if effective_output_file_wav and effective_output_file_wav.exists(): play_audio(str(effective_output_file_wav), is_path=True)
            else: play_audio(audio_bytes, is_path=False, input_format="pcm_s16le", sample_rate=ORPHEUS_SAMPLE_RATE)

def synthesize_with_orpheus_lm_studio(model_config, text, german_voice_id_override, model_params_override, output_file, play_direct):
    print(f"DEBUG: Orpheus LM Studio - Text: '{text[:50]}...', Voice: {german_voice_id_override or model_config['default_voice_id']}")

    voice = german_voice_id_override or model_config.get('default_voice_id', ORPHEUS_DEFAULT_VOICE)
    available_voices = model_config.get('available_voices', ORPHEUS_GERMAN_VOICES + ORPHEUS_AVAILABLE_VOICES_BASE)
    api_url = model_config.get("api_url", LM_STUDIO_API_URL_DEFAULT)

    cli_params = json.loads(model_params_override) if model_params_override else {}
    temperature = cli_params.get("temperature", 0.5)
    top_p = cli_params.get("top_p", 0.9)
    max_tokens_api = cli_params.get("max_tokens", 1200) # LM Studio default or user override
    repetition_penalty = cli_params.get("repetition_penalty", 1.1)
    model_name_in_api = cli_params.get("gguf_model_name_in_api", model_config.get("gguf_model_name_in_api", "SauerkrautTTS-Preview-0.1"))

    def _lm_studio_format_prompt_internal(prompt_text, voice_name):
        if voice_name not in available_voices:
            print(f"WARN: Orpheus LM Studio - Voice '{voice_name}' not in known list {available_voices}. Using default '{ORPHEUS_DEFAULT_VOICE}'.")
            voice_name = ORPHEUS_DEFAULT_VOICE
        special_start = "<|audio|>"
        special_end = "<|eot_id|>"
        return f"{special_start}{voice_name}: {prompt_text}{special_end}"

    formatted_prompt = _lm_studio_format_prompt_internal(text, voice)
    print(f"DEBUG: Orpheus LM Studio - Formatted prompt for API: {formatted_prompt}")

    def _lm_studio_generate_raw_token_text_stream():
        print(f"DEBUG: Orpheus LM Studio - Requesting tokens from API: {api_url} for model '{model_name_in_api}'")
        payload = {
            "model": model_name_in_api, "prompt": formatted_prompt,
            "max_tokens": max_tokens_api, "temperature": temperature, "top_p": top_p,
            "repeat_penalty": repetition_penalty, "stream": True
        }
        print(f"DEBUG: Orpheus LM Studio - API Payload: {json.dumps(payload)}")
        try:
            response = requests.post(api_url, headers=LM_STUDIO_HEADERS, json=payload, stream=True)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"ERROR: Orpheus LM Studio - API connection failed: {e}")
            return
        except Exception as e_gen:
            print(f"ERROR: Orpheus LM Studio - API request failed unexpectedly: {e_gen}")
            return

        full_raw_output_for_debug = ""
        for line in response.iter_lines():
            if line:
                line_str = line.decode('utf-8')
                if line_str.startswith('data: '):
                    data_str = line_str[6:]
                    if data_str.strip() == '[DONE]': break
                    try:
                        data = json.loads(data_str)
                        if 'choices' in data and data['choices'] and 'text' in data['choices'][0]:
                            token_text_chunk = data['choices'][0].get('text', '')
                            full_raw_output_for_debug += token_text_chunk
                            if token_text_chunk: yield token_text_chunk
                    except json.JSONDecodeError: continue
        # Removed VERY_VERBOSE_DEBUG of full raw output here
        print(f"DEBUG: Orpheus LM Studio - Full raw output (first 200 chars if long): '{full_raw_output_for_debug[:200]}{'...' if len(full_raw_output_for_debug)>200 else ''}'")
        print("DEBUG: Orpheus LM Studio - API stream finished.")

    effective_output_file_wav = Path(output_file).with_suffix(".wav") if output_file else None
    audio_bytes_result = _orpheus_master_token_processor_and_decoder(
        _lm_studio_generate_raw_token_text_stream(),
        output_file_wav=str(effective_output_file_wav) if effective_output_file_wav else None
    )

    if audio_bytes_result:
        if play_direct:
            if effective_output_file_wav and effective_output_file_wav.exists(): play_audio(str(effective_output_file_wav), is_path=True)
            else: play_audio(audio_bytes_result, is_path=False, input_format="pcm_s16le", sample_rate=ORPHEUS_SAMPLE_RATE)

def synthesize_with_orpheus_ollama(model_config, text, german_voice_id_override, model_params_override, output_file, play_direct):
    print(f"DEBUG: Orpheus Ollama - Text: '{text[:50]}...', Voice: {german_voice_id_override or model_config['default_voice_id']}")

    voice = german_voice_id_override or model_config.get('default_voice_id', ORPHEUS_DEFAULT_VOICE)
    available_voices = model_config.get('available_voices', ORPHEUS_GERMAN_VOICES + ORPHEUS_AVAILABLE_VOICES_BASE)
    formatted_prompt = orpheus_format_prompt(text, voice, available_voices)

    cli_params = json.loads(model_params_override) if model_params_override else {}
    temperature = cli_params.get("temperature", 0.5); top_p = cli_params.get("top_p", 0.9)
    repetition_penalty = cli_params.get("repeat_penalty", 1.1)

    ollama_model_name = cli_params.get("ollama_model_name", model_config.get("ollama_model_name", "my-orpheus-tts:latest"))
    ollama_api_url = cli_params.get("ollama_api_url", model_config.get("ollama_api_url", OLLAMA_API_URL_DEFAULT))

    def _ollama_text_stream_generator():
        print(f"DEBUG: Orpheus Ollama - Requesting from API: {ollama_api_url} for model: {ollama_model_name}")
        payload = {
            "model": ollama_model_name, "prompt": formatted_prompt, "stream": True,
            "options": { "temperature": temperature, "top_p": top_p, "repeat_penalty": repetition_penalty }
        }
        print(f"DEBUG: Orpheus Ollama - API Payload: {json.dumps(payload, indent=2)}")
        try:
            response = requests.post(ollama_api_url, json=payload, stream=True)
            response.raise_for_status()
        except requests.exceptions.RequestException as e: print(f"ERROR: Orpheus Ollama - API connection failed: {e}"); return
        except Exception as e_gen: print(f"ERROR: Orpheus Ollama - API request failed: {e_gen} - Response: {response.text if 'response' in locals() else 'No response object'}"); return

        full_raw_output_for_debug = ""
        for line in response.iter_lines():
            if line:
                try:
                    line_str = line.decode('utf-8'); data = json.loads(line_str)
                    if data.get("error"): print(f"ERROR: Orpheus Ollama - API error: {data['error']}"); break
                    token_text_chunk = data.get('response', '');
                    full_raw_output_for_debug += token_text_chunk
                    if token_text_chunk: yield token_text_chunk
                    if data.get("done", False): print("DEBUG: Orpheus Ollama - Ollama API stream 'done'."); break
                except json.JSONDecodeError: continue
        # Removed VERY_VERBOSE_DEBUG of full raw output here
        print(f"DEBUG: Orpheus Ollama - Full raw output (first 200 chars if long): '{full_raw_output_for_debug[:200]}{'...' if len(full_raw_output_for_debug)>200 else ''}'")
        print("DEBUG: Orpheus Ollama - API stream processing finished.")

    effective_output_file_wav = Path(output_file).with_suffix(".wav") if output_file else None
    audio_bytes = _orpheus_master_token_processor_and_decoder(_ollama_text_stream_generator(),
                                                              output_file_wav=str(effective_output_file_wav) if effective_output_file_wav else None)
    if audio_bytes:
        if play_direct:
            if effective_output_file_wav and effective_output_file_wav.exists(): play_audio(str(effective_output_file_wav), is_path=True)
            else: play_audio(audio_bytes, is_path=False, input_format="pcm_s16le", sample_rate=ORPHEUS_SAMPLE_RATE)
    else: print("WARN: Orpheus Ollama - No audio bytes generated.")

def synthesize_with_piper_local(model_config, text, german_voice_id_override, model_params_override, output_file, play_direct):
    if not PIPER_TTS_AVAILABLE :
        print("ERROR: piper-tts library not installed or import failed. Skipping Piper local.")
        print("       1. Ensure 'espeak-ng' (or 'espeak') is installed: e.g., 'brew install espeak' or 'sudo apt-get install espeak-ng'.")
        print("       2. Install phonemizer: 'pip install piper-phonemize-cross'")
        print("       3. Install piper-tts itself: 'pip install piper-tts'")
        return
    print(f"DEBUG: Piper Local - Text: '{text[:50]}...'")

    voice_details_config = model_config.get("default_voice_id", {})
    final_voice_model_path_in_repo = voice_details_config.get("model")
    final_voice_config_path_in_repo = voice_details_config.get("config")

    if german_voice_id_override:
        if isinstance(german_voice_id_override, str) and german_voice_id_override.startswith('{'):
            try:
                override_details = json.loads(german_voice_id_override)
                final_voice_model_path_in_repo = override_details.get("model", final_voice_model_path_in_repo)
                final_voice_config_path_in_repo = override_details.get("config", final_voice_config_path_in_repo)
                print(f"DEBUG: Piper - Using JSON voice override: model='{final_voice_model_path_in_repo}', config='{final_voice_config_path_in_repo}'")
            except json.JSONDecodeError:
                print(f"WARN: Piper - Failed to parse --german-voice-id as JSON. Assuming it's a model path part: {german_voice_id_override}")
                final_voice_model_path_in_repo = german_voice_id_override
                final_voice_config_path_in_repo = german_voice_id_override + ".json" if not german_voice_id_override.endswith(".onnx.json") else german_voice_id_override.replace(".onnx", ".onnx.json")
        elif isinstance(german_voice_id_override, str):
            final_voice_model_path_in_repo = german_voice_id_override
            if german_voice_id_override.endswith(".onnx.json"): # if user provided config path
                base_name = german_voice_id_override.replace(".onnx.json","")
                final_voice_model_path_in_repo = base_name + ".onnx"
                final_voice_config_path_in_repo = german_voice_id_override
            else: # assume user provided model path part
                 final_voice_config_path_in_repo = german_voice_id_override + ".json"
            print(f"DEBUG: Piper - Using string voice override: model='{final_voice_model_path_in_repo}', config='{final_voice_config_path_in_repo}'")
        elif isinstance(german_voice_id_override, dict): # Should not happen via CLI but good for internal calls
            final_voice_model_path_in_repo = german_voice_id_override.get("model", final_voice_model_path_in_repo)
            final_voice_config_path_in_repo = german_voice_id_override.get("config", final_voice_config_path_in_repo)
            print(f"DEBUG: Piper - Using dict voice override: model='{final_voice_model_path_in_repo}', config='{final_voice_config_path_in_repo}'")


    piper_voice_repo_id = model_config.get("piper_voice_repo_id", "rhasspy/piper-voices")
    if not final_voice_model_path_in_repo or not final_voice_config_path_in_repo:
        print(f"ERROR: Piper - Missing model or config path for {piper_voice_repo_id}")
        return

    model_cache_dir = Path.home() / ".cache" / "german_tts_script_piper_models"; model_cache_dir.mkdir(parents=True, exist_ok=True)
    local_piper_model_path = model_cache_dir / final_voice_model_path_in_repo
    local_piper_config_path = model_cache_dir / final_voice_config_path_in_repo

    local_piper_model_path.parent.mkdir(parents=True, exist_ok=True)
    local_piper_config_path.parent.mkdir(parents=True, exist_ok=True)

    if not local_piper_model_path.exists():
        print(f"INFO: Piper - Downloading model {final_voice_model_path_in_repo} from {piper_voice_repo_id}...")
        try:
            hf_hub_download(repo_id=piper_voice_repo_id, filename=final_voice_model_path_in_repo,
                            local_dir=model_cache_dir,
                            local_dir_use_symlinks=False, # To place it directly under specified path
                            token=os.getenv("HF_TOKEN"), repo_type="model")
        except Exception as e: print(f"ERROR: Piper - Download failed for model {final_voice_model_path_in_repo}: {e}"); return
    else: print(f"INFO: Piper - Model found locally: {local_piper_model_path}")

    if not local_piper_config_path.exists():
        print(f"INFO: Piper - Downloading config {final_voice_config_path_in_repo} from {piper_voice_repo_id}...")
        try:
            hf_hub_download(repo_id=piper_voice_repo_id, filename=final_voice_config_path_in_repo,
                            local_dir=model_cache_dir,
                            local_dir_use_symlinks=False,
                            token=os.getenv("HF_TOKEN"), repo_type="model")
        except Exception as e: print(f"ERROR: Piper - Download failed for config {final_voice_config_path_in_repo}: {e}"); return
    else: print(f"INFO: Piper - Config found locally: {local_piper_config_path}")

    try:
        print(f"INFO: Piper - Loading voice: Model='{local_piper_model_path}', Config='{local_piper_config_path}'")
        voice_obj = PiperVoice.load(str(local_piper_model_path), config_path=str(local_piper_config_path))
        print("INFO: Piper - Voice loaded.")

        # Use SpooledTemporaryFile for efficient in-memory handling of WAV data
        audio_bytes_io = tempfile.SpooledTemporaryFile()
        with wave.open(audio_bytes_io, 'wb') as wf:
            voice_obj.synthesize(text, wf) # Piper synthesizes directly to WAV file object

        audio_bytes_io.seek(0) # Rewind to the beginning of the "file"
        audio_data = audio_bytes_io.read() # Read all bytes
        audio_bytes_io.close() # Close (and delete if it was spooled to disk)

        if audio_data:
            print(f"INFO: Piper - Synthesis successful, {len(audio_data)} bytes generated.")
            effective_output_file_wav = Path(output_file).with_suffix(".wav") if output_file else None
            if effective_output_file_wav: save_audio(audio_data, effective_output_file_wav, source_is_path=False, input_format="wav")
            if play_direct: play_audio(audio_data, is_path=False, input_format="wav", sample_rate=voice_obj.config.sample_rate if hasattr(voice_obj, 'config') and hasattr(voice_obj.config, 'sample_rate') else 22050) # Fallback sample rate
        else: print("WARN: Piper - No audio data generated.")
    except Exception as e: print(f"ERROR: Piper - Synthesis failed: {e}")


def synthesize_with_outetts_local(model_config, text, german_voice_id_override, model_params_override, output_file, play_direct):
    if not OUTETTS_AVAILABLE:
        print("ERROR: OuteTTS library not available. Skipping OuteTTS synthesis.")
        return

    print(f"DEBUG: OuteTTS Local - Text: '{text[:50]}...'")

    speaker_ref_path_str = german_voice_id_override or model_config.get("default_voice_id")
    speaker_ref_path = None

    if isinstance(speaker_ref_path_str, str):
        if "path/to/your" in speaker_ref_path_str: # Default placeholder
            german_wav_fallback = Path("./german.wav")
            if german_wav_fallback.exists() and german_wav_fallback.is_file():
                print(f"INFO: OuteTTS - Found ./german.wav, using it as reference.")
                speaker_ref_path = german_wav_fallback
            else:
                print(f"ERROR: OuteTTS - Default speaker placeholder '{speaker_ref_path_str}' is set. "
                      "Please provide a speaker WAV via --german-voice-id or place 'german.wav' in the current directory.")
                return
        else:
            speaker_ref_path = Path(speaker_ref_path_str)
            if not speaker_ref_path.exists() or not speaker_ref_path.is_file() or speaker_ref_path.suffix.lower() != '.wav':
                print(f"ERROR: OuteTTS - Speaker reference path '{speaker_ref_path}' is not a valid .wav file.")
                return
    else:
        print(f"ERROR: OuteTTS - Invalid speaker reference path provided: {speaker_ref_path_str}")
        return

    print(f"INFO: OuteTTS - Using speaker reference: {speaker_ref_path}")

    try:
        interface_config = OuteTTSModelConfig.auto_config(
            model=OuteTTSModels.VERSION_1_0_SIZE_1B,
            backend=OuteTTSBackend.LLAMACPP,
            quantization=OuteTTSLlamaCppQuantization.FP16 # Default to FP16 for quality, can be changed
        )
        print("INFO: OuteTTS - Initializing interface...")
        interface = OuteTTSInterface(config=interface_config)

        print("INFO: OuteTTS - Creating speaker profile...")
        speaker = interface.create_speaker(str(speaker_ref_path))

        sampler_config = OuteTTSSamplerConfig(temperature=0.4, repetition_penalty=1.1, top_k=40, top_p=0.9, min_p=0.05)
        max_len = 8192
        if model_params_override:
            try:
                cli_params = json.loads(model_params_override)
                sampler_config.temperature = cli_params.get("temperature", sampler_config.temperature)
                sampler_config.repetition_penalty = cli_params.get("repetition_penalty", sampler_config.repetition_penalty)
                sampler_config.top_k = cli_params.get("top_k", sampler_config.top_k)
                sampler_config.top_p = cli_params.get("top_p", sampler_config.top_p)
                sampler_config.min_p = cli_params.get("min_p", sampler_config.min_p)
                max_len = cli_params.get("max_length", max_len)
            except json.JSONDecodeError:
                print(f"WARN: OuteTTS - Could not parse --model-params JSON: {model_params_override}")

        generation_config = OuteTTSGenerationConfig(
            text=text, generation_type=OuteTTSGenerationType.CHUNKED,
            speaker=speaker, sampler_config=sampler_config, max_length=max_len
        )

        print("INFO: OuteTTS - Generating speech...")
        start_time = time.time()
        output_audio = interface.generate(config=generation_config)
        end_time = time.time()
        print(f"INFO: OuteTTS - Speech generated in {end_time - start_time:.2f} seconds.")

        effective_output_file_wav = Path(output_file).with_suffix(".wav") if output_file else None
        if effective_output_file_wav:
            print(f"INFO: OuteTTS - Saving audio to {effective_output_file_wav}...")
            output_audio.save(str(effective_output_file_wav))
        
        if play_direct:
            print("INFO: OuteTTS - Playing audio...")
            output_audio.play()

    except Exception as e:
        print(f"ERROR: OuteTTS - Synthesis failed: {e}")


# --- Listing and Info Functions ---
def list_available_models():
    print("\nAvailable TTS Models for German (Cleaned List):")
    print("-------------------------------------------------")
    if not GERMAN_TTS_MODELS: print("No models configured."); return
    for model_id, config in GERMAN_TTS_MODELS.items():
        print(f"- {model_id}:")
        print(f"  Notes: {config.get('notes', 'N/A')}")
        if config.get('handler_function'): print(f"  Type: Local Handler ({config.get('handler_function')})")
        if "orpheus_gguf" in model_id: print(f"  Local GGUF Repo: {config.get('model_repo_id')}, File: {config.get('model_filename')}")
        elif "orpheus_lm_studio" in model_id: print(f"  Type: LM Studio API. Model in LM Studio: {config.get('gguf_model_name_in_api')}")
        elif "orpheus_ollama" in model_id: print(f"  Type: Ollama API. Model in Ollama: {config.get('ollama_model_name')}")
        elif "piper_local" in model_id: print(f"  Piper Voice Repo: {config.get('piper_voice_repo_id')}, Default model path part: {config.get('default_voice_id',{}).get('model')}")
    print("-------------------------------------------------")

def get_voice_info(model_id):
    print(f"\nVoice Information for Model: {model_id}")
    print("-------------------------------------")
    if model_id not in GERMAN_TTS_MODELS: print(f"Model ID '{model_id}' not found."); return
    config = GERMAN_TTS_MODELS[model_id]
    default_voice = config.get('default_voice_id', 'Not specified')

    if model_id == "piper_local":
        print(f"  Type: Piper TTS (Local)")
        print(f"  Default Voice Model Path (in repo {config.get('piper_voice_repo_id')}): {default_voice.get('model') if isinstance(default_voice, dict) else default_voice}")
        print(f"  Default Voice Config Path (in repo): {default_voice.get('config') if isinstance(default_voice, dict) else str(default_voice) + '.json'}")
        print(f"  To use a different voice with --german-voice-id, provide the relative path to the .onnx model file within the piper-voices repo (e.g., 'de/de_DE/eva_k/medium/de_DE-eva_k-medium.onnx') or a JSON string '{{\"model\": \"path/to/model.onnx\", \"config\": \"path/to/config.json\"}}'.")
        print(f"  Available voices: {config.get('available_voices_info', 'See rhasspy/piper-voices on Hugging Face.')}")
    elif model_id == "oute":
        print(f"  Type: OuteTTS (Local)")
        print(f"  Voice Selection: Requires a path to a German reference .wav file (max 20s) for '--german-voice-id'.")
        if "path/to/your" in str(default_voice): print(f"  NOTE: Default voice ('{default_voice}') is a placeholder. Provide via CLI or place ./german.wav.")
    else:
        print(f"  Configured Default German Voice/ID/Prompt: {default_voice}")
        if "available_voices" in config:
            print(f"  Known available voices for this model type (use with --german-voice-id):")
            for voice in config["available_voices"]:
                marker = " ★ (default)" if voice == default_voice else ""
                print(f"    - {voice}{marker}")
        elif "orpheus" in model_id:
            print(f"  Type: Orpheus/SauerkrautTTS variant")
            print(f"  Use '--german-voice-id' with one of the listed voices.")
            print(f"  Emotion tags like <laugh>, <sigh> can be added to the input text.")
    print("-------------------------------------")


# --- Test All Models ---
def test_all_models(text, base_output_dir_str="tts_test_outputs", args_cli=None):
    print(f"\n--- Starting Test for All Models (Cleaned Set) ---"); print(f"Input text: \"{text[:100]}...\"")
    base_output_dir = Path(base_output_dir_str); base_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Outputs will be saved to: {base_output_dir.resolve()}"); print("------------------------------------")
    if not GERMAN_TTS_MODELS: print("No models configured."); return

    for model_id, config in GERMAN_TTS_MODELS.items():
        print(f"\n>>> Testing Model: {model_id} <<<")
        output_suffix = ".wav" if "orpheus" in model_id or "piper" in model_id or "oute" in model_id else ".mp3"
        output_filename = base_output_dir / f"test_output_{model_id.replace('/', '_').replace(':','_')}{output_suffix}"
        voice_to_use = config.get("default_voice_id")

        if model_id == "oute": # OuteTTS requires a local .wav file
            german_wav_path = Path("./german.wav")
            if "path/to/your" in str(voice_to_use) and german_wav_path.exists():
                print(f"INFO: Model '{model_id}' - Found ./german.wav, using it as reference for test.")
                voice_to_use = str(german_wav_path)
            elif not (isinstance(voice_to_use, str) and Path(voice_to_use).exists() and Path(voice_to_use).is_file() and Path(voice_to_use).suffix.lower() == '.wav'):
                print(f"WARN: Model '{model_id}' requires a valid .wav file path for 'default_voice_id' or a local './german.wav'. Current: '{voice_to_use}'. Skipping."); print("------------------------------------"); continue

        temp_model_config = config.copy()
        if model_id == "orpheus_lm_studio" and args_cli:
            temp_model_config["api_url"] = args_cli.lm_studio_api_url
            if args_cli.gguf_model_name_in_api: temp_model_config["gguf_model_name_in_api"] = args_cli.gguf_model_name_in_api
        elif model_id == "orpheus_ollama" and args_cli:
            temp_model_config["ollama_api_url"] = args_cli.ollama_api_url
            if args_cli.ollama_model_name: temp_model_config["ollama_model_name"] = args_cli.ollama_model_name
            elif not temp_model_config.get("ollama_model_name"):
                print(f"ERROR: Testing {model_id}: 'ollama_model_name' not set in config and not provided via CLI. Skipping."); continue
        elif model_id == "piper_local" and args_cli:
                if args_cli.piper_voice_repo: temp_model_config["piper_voice_repo_id"] = args_cli.piper_voice_repo

        try:
            handler_name = temp_model_config.get("handler_function")
            if handler_name:
                handler_func = globals().get(handler_name)
                if handler_func: handler_func(temp_model_config, text, voice_to_use, None, str(output_filename), False)
                else: print(f"WARN: Handler '{handler_name}' for {model_id} not found. Skipping.")
            else: print(f"WARN: No handler for {model_id}. Skipping.")

            if output_filename.exists() and output_filename.stat().st_size > 100: # Basic check
                print(f"SUCCESS: Output for {model_id} saved to {output_filename}")
            else: print(f"NOTE: Synthesis for {model_id} ran. Check logs. Output file '{output_filename}' not created or is empty.")
        except Exception as e: print(f"ERROR: Testing model {model_id} failed: {e}")
        print("------------------------------------")
    print("\n--- Test for All Models Finished ---")

# --- Main Execution ---
def main():
    parser = argparse.ArgumentParser(description="German Text-to-Speech Synthesizer (Cleaned)", formatter_class=argparse.RawTextHelpFormatter)
    action_group = parser.add_argument_group(title="Actions"); input_group = parser.add_mutually_exclusive_group()
    action_group.add_argument("--list-models", action="store_true", help="List configured (working) models.")
    action_group.add_argument("--voice-info", type=str, metavar="MODEL_ID", help="Show voice info for a model.")
    action_group.add_argument("--test-all", action="store_true", help="Test all models with --input-text. Outputs to --output-dir.")

    synth_group = parser.add_argument_group(title="Synthesis Options")
    input_group.add_argument("--input-text", type=str, help="Text to synthesize.")
    input_group.add_argument("--input-file", type=str, help="Path to file (txt, md, html, pdf, epub).")
    synth_group.add_argument("--model-id", type=str, choices=GERMAN_TTS_MODELS.keys(), help="TTS model ID.")
    synth_group.add_argument("--output-file", type=str, help="Save audio (e.g., output.mp3 or output.wav). Suffix determined by model.")
    synth_group.add_argument("--output-dir", type=str, default="tts_test_outputs", help="Output dir for --test-all.")
    synth_group.add_argument("--play-direct", action="store_true", help="Play audio directly (not with --test-all).")
    synth_group.add_argument("--german-voice-id", type=str, help="Voice ID/path/prompt/JSON (overrides default). For Piper, specific formats apply.")
    synth_group.add_argument("--model-params", type=str, help="JSON string of additional model parameters (e.g., temperature).")

    api_group = parser.add_argument_group(title="API Backend Options")
    api_group.add_argument("--lm-studio-api-url", type=str, default=LM_STUDIO_API_URL_DEFAULT, help=f"LM Studio API URL (default: {LM_STUDIO_API_URL_DEFAULT}).")
    api_group.add_argument("--gguf-model-name-in-api", type=str, help="Model name in LM Studio (if different from config for orpheus_lm_studio).")
    api_group.add_argument("--ollama-api-url", type=str, default=OLLAMA_API_URL_DEFAULT, help=f"Ollama API URL (default: {OLLAMA_API_URL_DEFAULT}).")
    api_group.add_argument("--ollama-model-name", type=str, help="Model name/tag in Ollama (e.g., 'orpheus-german-tts:latest'). Required for orpheus_ollama if not set in config.")

    local_model_group = parser.add_argument_group(title="Local Model Options")
    local_model_group.add_argument("--piper-voice-repo", type=str, help="HF repo for Piper voices (e.g., rhasspy/piper-voices). Overrides config.")
    # --bark-model-repo removed as bark_local_transformers is removed for now

    args = parser.parse_args()

    if args.list_models: list_available_models(); return
    if args.voice_info: get_voice_info(args.voice_info); return

    text_to_synthesize = get_text_from_input(args.input_text, args.input_file)
    if not text_to_synthesize:
        if args.test_all and not args.input_text: parser.error("--test-all requires --input-text.")
        elif not args.test_all: parser.error("Need --input-text or --input-file for synthesis.")
        return
    # Limit text length for sanity, especially for API calls or long processing
    text_to_synthesize = text_to_synthesize[:3000] # Increased slightly, but still a good limit

    if args.test_all:
        if not args.input_text: print("ERROR: --test-all requires --input-text."); return
        test_all_models(args.input_text, args.output_dir, args_cli=args); return

    if not args.model_id: parser.error("--model-id required for single synthesis."); return
    model_config = GERMAN_TTS_MODELS.get(args.model_id)
    if not model_config: print(f"ERROR: Invalid model ID: {args.model_id}."); return

    print(f"\nSynthesizing with: {args.model_id}"); print(f"Input (start): '{text_to_synthesize[:70]}...'")

    effective_voice_id = args.german_voice_id
    if not effective_voice_id and args.model_id == "oute": # Oute needs a wav
        default_model_voice = model_config.get("default_voice_id", "")
        if isinstance(default_model_voice, str) and "path/to/your" in default_model_voice:
            german_wav_path = Path("./german.wav")
            if german_wav_path.exists() and german_wav_path.is_file():
                print(f"INFO: Model '{args.model_id}' - Found ./german.wav, using it as reference for single synthesis.")
                effective_voice_id = str(german_wav_path)
            else:
                print(f"ERROR: Model '{args.model_id}' requires a speaker .wav via --german-voice-id or ./german.wav since default is placeholder.")
                return


    current_model_config = model_config.copy()
    if args.model_id == "orpheus_lm_studio":
        current_model_config["api_url"] = args.lm_studio_api_url
        if args.gguf_model_name_in_api: current_model_config["gguf_model_name_in_api"] = args.gguf_model_name_in_api
    elif args.model_id == "orpheus_ollama":
        current_model_config["ollama_api_url"] = args.ollama_api_url
        if args.ollama_model_name: current_model_config["ollama_model_name"] = args.ollama_model_name
        elif not current_model_config.get("ollama_model_name"):
            print(f"ERROR: For --model-id {args.model_id}, --ollama-model-name is required or 'ollama_model_name' must be set in GERMAN_TTS_MODELS config.")
            return
    elif args.model_id == "piper_local":
        if args.piper_voice_repo: current_model_config["piper_voice_repo_id"] = args.piper_voice_repo


    handler_to_call = None
    handler_name = current_model_config.get("handler_function")
    if handler_name: handler_to_call = globals().get(handler_name)
    # Gradio client based models were removed, so no need for that specific check here.

    if handler_to_call:
        handler_to_call(current_model_config, text_to_synthesize, effective_voice_id, args.model_params, args.output_file, args.play_direct)
    else: print(f"ERROR: No synthesis method for model ID: {args.model_id}")

if __name__ == "__main__":
    if any(m.get("requires_hf_token") for m in GERMAN_TTS_MODELS.values()) and not os.getenv("HF_TOKEN"):
        print("INFO: Some models might need HF_TOKEN env var (for downloads from Hugging Face).")
    main()
