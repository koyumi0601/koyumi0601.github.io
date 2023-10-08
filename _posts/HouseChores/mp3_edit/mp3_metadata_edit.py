from mutagen.easyid3 import EasyID3
import os

def extract_artist_and_album(folder_name):
    # 'Pops_' 다음의 첫 번째 '_'의 위치 찾기
    artist_start = folder_name.find('Pops_') + len('Pops_')
    
    # 다음 '_'의 위치 찾기
    artist_end = folder_name.find('_', artist_start)
    
    if artist_start == -1 or artist_end == -1:
        raise ValueError("Invalid folder structure. Expected 'Pops_Artist_Album'.")

    new_artist = folder_name[artist_start:artist_end]
    new_album = folder_name[artist_end + 1:]

    return new_artist, new_album

def change_metadata(mp3_file, new_artist, new_album):
    try:
        tags = EasyID3(mp3_file)
        tags['artist'] = new_artist
        tags['album'] = new_album
        tags.save()
        print(f"Metadata for {mp3_file} updated.")
    except Exception as e:
        print(f"Error updating metadata for {mp3_file}: {str(e)}")

def process_folder(root_folder):
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.endswith(".mp3"):
                mp3_file = os.path.join(root, file)
                folder_name = os.path.basename(root)
                new_artist, new_album = extract_artist_and_album(folder_name)
                change_metadata(mp3_file, new_artist, new_album)

# 사용 예시: 상위 폴더를 지정
root_folder = r'D:\Shared_window2ubuntu\Music_CD_artist\가요'  # 상위 폴더 경로를 지정하세요

# 모든 하위 폴더에서 메타데이터 변경
process_folder(root_folder)