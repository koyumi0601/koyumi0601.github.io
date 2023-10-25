# pip install moviepy openai-whisper
import moviepy.editor as mp
import whisper

###########################################################################################
# mp4 파일 경로
mp4_file_path = r'D:\Shared_window2ubuntu\Lecture\Machine_Learning_Legend13_Image_Papers'
mp4_file_name = 'Pop1'
mp4_file = mp4_file_path + '\\' + mp4_file_name
# .wav 파일 경로
wav_file = mp4_file + '.wav'  # 실제 .wav 파일 경로로 변경해야 합니다.
# mp4 파일을 오디오로 추출 및 WAV로 변환
# audio_clip = mp.AudioFileClip(mp4_file + '.mp4')
# audio_clip.write_audiofile(mp4_file + '.wav')
# print(f"MP4 파일을 WAV 파일로 변환했습니다.")
###########################################################################################







##########################################################################################



# pip install numpy matplotlib librosa
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display

# .wav 파일 로드
audio, sr = librosa.load(wav_file, sr=None)

# 음성 스펙트럼 계산
D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)

# 음성 스펙트럼 플롯
plt.figure(figsize=(10, 6))
librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
plt.colorbar(format='%+2.0f dB')
plt.title('Voice Spectrum')
plt.show()


###############################################################
# # Whisper 모델 로드
# model = whisper.load_model("base")

# # 오디오 파일을 로드
# # audio = whisper.load_audio(r'D:\Shared_window2ubuntu\Lecture\Machine_Learning_Legend13_Image_Papers\Day1_Audio.wav')
# audio = whisper.load_audio('D:/Shared_window2ubuntu/Lecture/Machine_Learning_Legend13_Image_Papers/Day1_Audio.wav')
# # 오디오를 로그-Mel 스펙트로그램으로 변환하고 모델의 장치로 이동
# mel = whisper.log_mel_spectrogram(audio).to(model.device)

# # 텍스트로 디코딩을 위한 옵션 설정 (한국어로)
# options = whisper.DecodingOptions(language="ko-KR")

# # 텍스트로 디코딩
# result = whisper.decode(model, mel, options)

# # 추출된 텍스트 출력
# extracted_text = result.text
# print(extracted_text)

# # 추출된 텍스트를 파일로 저장 (원하는 파일 경로 및 이름 지정)
# output_text_path = mp4_file_name + '.txt'
# with open(output_text_path, 'w', encoding='utf-8') as file:
#     file.write(extracted_text)

# print(f"추출된 텍스트를 {output_text_path} 파일에 저장했습니다.")