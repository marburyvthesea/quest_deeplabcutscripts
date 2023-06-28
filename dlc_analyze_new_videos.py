import os
os.environ["DLClight"]="True"
import sys
sys.path.append('/home/jma819/DeepLabCut/')
import deeplabcut


#path_config_file = '/projects/p30771/dlc_analysis/GRIN_rotarod_rearview-JJM-2020-04-02/config.yaml' 
path_config_file = sys.argv[1]


behav_cam_directory = sys.argv[2]

avi_list = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(behav_cam_directory)) for f in fn if f.endswith('.avi')]

mp4_list = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(behav_cam_directory)) for f in fn if f.endswith('.mp4')]

if len(avi_list)>0:
	deeplabcut.analyze_videos(path_config_file, avi_list, videotype='.avi', save_as_csv=True)
	print('finished retraining!, creating labellef videos...')
	#Create labeled video
	deeplabcut.create_labeled_video(path_config_file, avi_list)
	
if len(mp4_list)>0:
	deeplabcut.analyze_videos(path_config_file, mp4_list, videotype='.mp4', save_as_csv=True)
	print('finished retraining!, creating labellef videos...')
	#Create labeled video
	deeplabcut.create_labeled_video(path_config_file, mp4_list)

print('finished creating labeled videos!')
