import os
import subprocess
import sys

def check_ffmpeg():
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def install_yt_dlp():
    subprocess.check_call([sys.executable, "-m", "pip", "install", "yt-dlp"])

def check_available_formats(url, target_height=1080):
    try:
        import yt_dlp
        
        ydl_opts = {
            'quiet': True,
            'cookiesfrombrowser': ('firefox',),
            'no_warnings': True,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            video_title = info.get('title', 'Unknown')
            
            print(f"Анализ видео: {video_title}")
            
            video_formats = []
            h264_formats = []
            vp9_formats = []
            av1_formats = []
            
            for f in info.get('formats', []):
                if f.get('vcodec') != 'none': 
                    height = f.get('height', 0)
                    width = f.get('width', 0)
                    vcodec = f.get('vcodec', 'unknown')
                    ext = f.get('ext', '?')
                    fps = f.get('fps', '?')
                    bitrate = f.get('tbr', f.get('vbr', 0))
                    
                    format_info = {
                        'height': height,
                        'width': width,
                        'codec': vcodec,
                        'ext': ext,
                        'fps': fps,
                        'bitrate': bitrate,
                        'format_id': f.get('format_id')
                    }
                    
                    video_formats.append(format_info)
                    
                    if 'avc1' in vcodec or 'h264' in vcodec.lower():
                        h264_formats.append(format_info)
                    elif 'vp9' in vcodec.lower():
                        vp9_formats.append(format_info)
                    elif 'av1' in vcodec.lower():
                        av1_formats.append(format_info)
            
            # Находим лучшее качество в каждом кодеке
            best_h264 = max(h264_formats, key=lambda x: x['height']) if h264_formats else None
            best_vp9 = max(vp9_formats, key=lambda x: x['height']) if vp9_formats else None
            best_av1 = max(av1_formats, key=lambda x: x['height']) if av1_formats else None
            
            # Находим лучшее качество вообще
            all_formats_by_height = sorted(video_formats, key=lambda x: x['height'], reverse=True)
            best_overall = all_formats_by_height[0] if all_formats_by_height else None
            
            print(f"\n Доступные форматы:")
            print(f"{'Кодек':<10} {'Макс. качество':<15} {'Доступные разрешения'}")

            if best_h264:
                h264_resolutions = sorted(set([f['height'] for f in h264_formats if f['height'] > 0]), reverse=True)
                print(f"{'H.264':<10} {best_h264['height']}p{' '*(13-len(str(best_h264['height'])+'p'))} {h264_resolutions}")
            else:
                print(f"{'H.264':<10} {'Нет':<15}")
            
            if best_vp9:
                vp9_resolutions = sorted(set([f['height'] for f in vp9_formats if f['height'] > 0]), reverse=True)
                print(f"{'VP9':<10} {best_vp9['height']}p{' '*(13-len(str(best_vp9['height'])+'p'))} {vp9_resolutions}")
            else:
                print(f"{'VP9':<10} {'Нет':<15}")
            
            if best_av1:
                av1_resolutions = sorted(set([f['height'] for f in av1_formats if f['height'] > 0]), reverse=True)
                print(f"{'AV1':<10} {best_av1['height']}p{' '*(13-len(str(best_av1['height'])+'p'))} {av1_resolutions}")
            else:
                print(f"{'AV1':<10} {'Нет':<15}")
            
            result = {
                'title': video_title,
                'best_overall': best_overall['height'] if best_overall else 0,
                'best_h264': best_h264['height'] if best_h264 else 0,
                'best_vp9': best_vp9['height'] if best_vp9 else 0,
                'best_av1': best_av1['height'] if best_av1 else 0,
                'can_download_target': (best_h264 and best_h264['height'] >= target_height) or 
                                       (best_vp9 and best_vp9['height'] >= target_height) or
                                       (best_av1 and best_av1['height'] >= target_height),
                'needs_conversion': best_h264 and best_h264['height'] < target_height and 
                                   (best_vp9 and best_vp9['height'] >= target_height) or
                                   (best_av1 and best_av1['height'] >= target_height)
            }
            
            return result
            
    except Exception as e:
        print(f"Ошибка при анализе форматов: {e}")
        return None

def download_guaranteed_quality(url, output_path=".", target_quality="1080"):

    quality_map = {
        "1080": 1080,
        "1440": 1440,
        "2160": 2160,
        "4320": 4320
    }
    target_height = quality_map.get(target_quality, 1080)
    
    if not check_ffmpeg():
        print("FFmpeg не найден")
        return False

    try:
        import yt_dlp
    except ImportError:
        print("yt-dlp не найден, начинается установка")
        install_yt_dlp()
        import yt_dlp
    
    os.makedirs(output_path, exist_ok=True)
    
    print(f"Анализ доступных форматов для {url}")
    formats_info = check_available_formats(url, target_height)
    
    if not formats_info:
        print("Не удалось получить информацию о форматах")
        return False
    
    print(f"\n Целевое качество: {target_quality}p")
    print(f"Лучшее доступное качество: {formats_info['best_overall']}p")
    print(f"Лучшее H.264 качество: {formats_info['best_h264']}p")

    if not formats_info['can_download_target']:
        print(f"\n Видео недоступно в качестве {target_quality}p или выше")
        print(f"Максимальное доступное качество: {formats_info['best_overall']}p")
        return False

    if formats_info['best_h264'] >= target_height:
        #1: Скачиваем напрямую в H.264
        print(f"\n Доступно H.264 в качестве {formats_info['best_h264']}p")
        
        format_selector = f"bestvideo[height<={formats_info['best_h264']}][vcodec^=avc1]+bestaudio[ext=m4a]/best[height<={formats_info['best_h264']}][vcodec^=avc1]"
        
        ydl_opts = {
            'format': format_selector,
            'merge_output_format': 'mp4',
            'outtmpl': os.path.join(output_path, '%(title)s.%(ext)s'),
            'noplaylist': True,
            'quiet': False,
            'cookiesfrombrowser': ('firefox',),
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        
        print(f"\n Видео скачано в H.264 {formats_info['best_h264']}p")
        return True
        
    else:
        print(f"\n H.264 доступен только в {formats_info['best_h264']}p")

        temp_dir = os.path.join(output_path, "temp_conversion")
        os.makedirs(temp_dir, exist_ok=True)
        
        try:
            ydl_opts = {
                'format': f'bestvideo[height<={formats_info["best_overall"]}]+bestaudio/best[height<={formats_info["best_overall"]}]',
                'outtmpl': os.path.join(temp_dir, '%(title)s.%(ext)s'),
                'noplaylist': True,
                'quiet': False,
                'cookiesfrombrowser': ('firefox',),
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                video_title = info.get('title', 'Unknown')

                downloaded_file = None
                for file in os.listdir(temp_dir):
                    if file.endswith(('.mp4', '.webm', '.mkv')) and video_title in file:
                        downloaded_file = os.path.join(temp_dir, file)
                        break
                
                if not downloaded_file:
                    raise Exception("Не найден скачанный файл")

                output_file = os.path.join(output_path, f"{video_title}.mp4")
                
                cmd = [
                    'ffmpeg', '-i', downloaded_file,
                    '-c:v', 'libx264',
                    '-preset', 'medium',
                    '-crf', '23',
                    '-c:a', 'aac',
                    '-b:a', '192k',
                    '-movflags', '+faststart',
                    '-y',
                    output_file
                ]
                
                subprocess.run(cmd, capture_output=True, check=True)

                import shutil
                shutil.rmtree(temp_dir)
                
                print(f"\n Видео сконвертировано в H.264 {formats_info['best_overall']}p")
                return True
                
        except Exception as e:
            print(f"\n Ошибка при конвертации: {e}")
            import shutil
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            return False

def download_multiple_videos_guaranteed(urls, output_path=".", target_quality="1080"):
    print(f"Всего видео: {len(urls)}")
    print(f"Папка для сохранения: {output_path}")
    
    os.makedirs(output_path, exist_ok=True)
    
    statistics = {
        'success': [],
        'failed': [],
        'quality_report': []
    }
    
    for i, url in enumerate(urls, 1):
        print(f"Видео{i} из {len(urls)}")  
        success = download_guaranteed_quality(url, output_path, target_quality)
        
        if success:
            statistics['success'].append(url)
        else:
            statistics['failed'].append(url)
        
        print(f"Прогресс: {i}/{len(urls)}, успешно: {len(statistics['success'])}, ошибок: {len(statistics['failed'])}")

    print(f"Скачано: {len(statistics['success'])}")
    print(f"Не удалось скачать: {len(statistics['failed'])}")
    
    if statistics['failed']:
        print("Список видео, которые не удалось скачать в нужном качестве:")
        for i, url in enumerate(statistics['failed'], 1):
            print(f"  {i}. {url}")
    
    print(f"Все видео сохранены в: {os.path.abspath(output_path)}")
    
    return statistics

if __name__ == "__main__":
    
    # Пример 1: Скачивание одного видео с проверкой качества
    video_url = "https://www.youtube.com/watch?v=hq5t4zNUqFU"

    formats = check_available_formats(video_url, 1080)
    
    if formats and formats['can_download_target']:
        print("\n Видео можно скачать в 1080p+")
        download_guaranteed_quality(video_url, "./downloads4", "1080")
    else:
        print("\n Видео недоступно в 1080p+")
    
    """
    #Пример 2: Массовое скачивание нескольких видео
    video_urls = [
        "https://www.youtube.com/watch?v=jlEUgPqtlC4",
        "https://www.youtube.com/watch?v=another_video"]
    
    download_multiple_videos_guaranteed(
        urls=video_urls,
        output_path="./downloads_high_quality",
        target_quality="1080")  # Можно еще"1440" или "2160"
    
    #Пример 3: Скачивание в 4K
    download_guaranteed_quality(
         url="https://www.youtube.com/watch?v=some_4k_video",
         output_path="./4k_downloads",
         target_quality="2160")
    """