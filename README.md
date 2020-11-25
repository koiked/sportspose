# sportspose
This is project for acquireing pose from the sports videos.
The code and trained data is obtained from the 
Coral edge tpu projects {project-posenet} and {project-bodypix}
https://github.com/google-coral

You need Edge tpu from google coral to run the project. 
And you need opencv-python library to run the code.

You also need vidoes to analyze, you better to use your own 
mp4 files. What kind of the video format you can use is depending 
on your build option for ffmpeg libraries. 
You can download from <youtube_URL> using youtube-dl.
> youtube-dl -F <youtubu_URL>
give you what kind of video is affordable, then find <id> you want use
then 
>youtube-dl -f <id> <youtube_URL> 
you can download the file.
If you need to trim the movie, 
find keyframe using ffprobe, then you trim videos from the frame 
using ffmpeg. 

enjoy

