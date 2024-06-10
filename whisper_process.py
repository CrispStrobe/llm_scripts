import subprocess
import argparse
import re
import os

def convert_audio_to_wav(input_file, output_file):
    command = ['ffmpeg', '-i', input_file, '-ar', '16000', '-ac', '1', '-c:a', 'pcm_s16le', output_file]
    subprocess.run(command, check=True)

def run_whisper_and_capture_output(command, output_file, cleaned_file):
    with open(output_file, 'w', encoding='utf-8') as file:
        # Run the command as a subprocess
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)

        # Read the output in real-time
        for line in process.stdout:
            # Write the output to the console
            print(line, end='')
            # Write the output to the file
            file.write(line)

    # Wait for the process to complete
    process.wait()

    # Process the output file to remove timecodes
    with open(output_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    with open(cleaned_file, 'w', encoding='utf-8') as file:
        for line in lines:
            # Match lines with timecodes
            match = re.match(r'\[\d{2}:\d{2}:\d{2}\.\d{3} --> \d{2}:\d{2}:\d{2}\.\d{3}\]', line)
            if match:
                # Remove the timecodes but preserve the rest of the line
                cleaned_line = re.sub(r'\[\d{2}:\d{2}:\d{2}\.\d{3} --> \d{2}:\d{2}:\d{2}\.\d{3}\]\s+', '', line).lstrip()
                file.write(cleaned_line)

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run whisper.cpp with audio file input and process the output.')
    parser.add_argument('input_file', type=str, help='Input audio file (any format that ffmpeg supports)')
    parser.add_argument('--model', type=str, default='models/ggml-medium-q5_0.bin', help='Path to the model file')
    parser.add_argument('--output', type=str, default='whisper_output.txt', help='File to save whisper.cpp output')
    parser.add_argument('--cleaned', type=str, default='cleaned_output.txt', help='File to save cleaned output')
    parser.add_argument('--lang', type=str, default='de', help='Language code for whisper.cpp')
    parser.add_argument('--print-colors', type=str, choices=['true', 'false', 'no'], default='true', help='Enable or disable color printing in whisper.cpp output')
    parser.add_argument('whisper_args', nargs=argparse.REMAINDER, help='Additional arguments for whisper.cpp')

    args = parser.parse_args()

    # Check if the input file is not WAV and needs conversion
    if not args.input_file.endswith('.wav'):
        converted_file = 'output.wav'
        convert_audio_to_wav(args.input_file, converted_file)
    else:
        converted_file = args.input_file

    # Construct whisper.cpp command
    command = ['./main', '-m', args.model, '-f', converted_file, '-l', args.lang]
    
    if args.print_colors == 'true':
        command.append('--print-colors')

    command += args.whisper_args

    # Run whisper.cpp and capture output
    run_whisper_and_capture_output(command, args.output, args.cleaned)

if __name__ == "__main__":
    main()
