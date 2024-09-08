import yt_dlp

def stream_video(url): # Streams a YouTube Video given as url
    ydl_opts = {'format': 'best'}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(url, download=False)
        video_url = info_dict['url']

    return video_url