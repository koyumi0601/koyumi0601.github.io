import os

def rename_files_with_spaces_and_remove_track(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".mp3"):
                old_file_path = os.path.join(root, file)

                # "Track"을 제거하고 언더바를 스페이스로 교체
                new_file_name = file.replace('Track', '').replace('_', ' ')

                # 맨 앞과 끝에 있는 스페이스 제거
                new_file_name = new_file_name.strip()

                # 새 파일 이름으로 변경
                new_file_path = os.path.join(root, new_file_name)

                # 새 파일 이름으로 변경
                os.rename(old_file_path, new_file_path)

# 사용 예시
directory_path = r"D:\Shared_window2ubuntu\Music_CD_artist"  # 최상위 폴더 경로를 지정하세요

# 언더바를 스페이스로 변경하고 "Track" 제거 및 맨 앞과 끝의 스페이스 제거
rename_files_with_spaces_and_remove_track(directory_path)