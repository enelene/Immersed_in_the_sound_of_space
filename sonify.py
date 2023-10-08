import cv2
import numpy as np
import soundfile as sf
import time
import math
from matplotlib import pyplot as plt
from moviepy.editor import VideoFileClip, AudioFileClip
import sys


def main(video_file_path):
    audio_output_path = f"sonified_video_audio_{time.time()}.wav"
    audio_data = []
    sample_rate = 30000
    cap = cv2.VideoCapture(video_file_path)
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    i = 0
    while True:
        ret, frame = cap.read()
        i+=1
        if i%30!=0:
            continue

        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

        saturation = np.sum(frame[:,:,1])
        value = np.sum(frame[:,:,2])

        max = 1
        for dim in frame[:,:,1].shape:
            max *= dim
        max *= 255
        
        saturation_pitch = map_color_sum_to_pitch(saturation, max, [i - 12 for i in [67,68,69,72,74,75,71,77]])
        value_pitch = map_color_sum_to_pitch(value, max, [67,68,69,72,74,75,71,77])

        saturation_sine = generate_sine_wave_with_offset(saturation_pitch, 1, sample_rate)
        value_sine = generate_sine_wave_with_offset(value_pitch, 1, sample_rate)

        audio_segment = ((saturation_sine + 0.5 * value_sine) * 16383).astype(np.int16)

        audio_data.extend(audio_segment)

    audio_data = np.array(audio_data, dtype=np.int16)
    sf.write(audio_output_path, audio_data, sample_rate, format='WAV')
    cap.release()
    print("Sonified video audio generated and saved to", audio_output_path)
    input_audio_path = audio_output_path
    output_video_path = 'output_video.mp4'
    video_clip = VideoFileClip(video_file_path)
    audio_clip = AudioFileClip(input_audio_path)
    video_clip = video_clip.set_audio(audio_clip)
    video_clip.write_videofile(output_video_path, codec='libx264', audio_codec='aac')

    print(f"New video with input audio saved to {output_video_path}")




def generate_sine_wave_with_offset(pitch, duration, sample_rate=44100):
    frequency = 440 * (2 ** ((pitch - 69) / 12))
    num_samples = int(sample_rate * duration)
    t = np.linspace(0, duration, num_samples, endpoint=False)
    tfreq = np.round(frequency * t)
    sine_wave = np.sin(2 * np.pi * frequency * t)
    return sine_wave

def map_color_sum_to_pitch(color_sum, max, pitches):
    min_color_sum = 0
    max_color_sum = max + 1
    min_pitch = 0
    max_pitch = len(pitches)
    pitch = np.interp(color_sum, [min_color_sum, max_color_sum], [min_pitch, max_pitch])
    pitch = math.floor(pitch)
    return pitches[pitch]

if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise(Exception("Usage: python sonify.py {video path}"))
    main(sys.argv[1])