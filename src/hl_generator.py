import os
import sys


if len(sys.argv) < 4:
	video_names = [5, 8, 10, 12, 20, 24, 26, 37]
	for clip_nb in video_names:
		back_path = '../Video/{}/back/back_camera.mp4' .format(clip_nb)
		front_path = '../Video/{}/front/front_camera.mp4' .format(clip_nb)
		hl_path = '../Video/{}/highlight' .format(clip_nb)

		os.system("python3 main.py {} {} {} -1 -1".format(back_path, front_path, hl_path))
else:
	back_videos = sys.argv[1]
	front_videos = sys.argv[2]
	hl_videos = sys.argv[3]


	for file in os.listdir(back_videos):
		if file[0] != '.' and file.split('.')[-1] == 'mp4':
			back_path = os.path.join(back_videos, file)
			front_path = os.path.join(front_videos, file)

			os.system("python3 main.py {} {} {} -1 -1 True".format(back_path, front_path, hl_videos))