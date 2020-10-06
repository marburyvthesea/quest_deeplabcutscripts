import os
os.environ["DLClight"]="True"
import sys
sys.path.append('/home/jma819/DeepLabCut/')
import deeplabcut


#path_config_file = '/projects/p30771/dlc_analysis/GRIN_rotarod_rearview-JJM-2020-04-02/config.yaml' 
path_config_file = sys.argv[1]


behav_cam_directory = sys.argv[2]

avi_list = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(behav_cam_directory)) for f in fn if f.endswith('.avi')]

deeplabcut.analyze_videos(path_config_file, avi_list, videotype='.avi')

#Following the refinement of labels and appending them to the original dataset, 
#this creates a new iteration of training dataset. This is automatically set in the config.yaml file, 
#so let's get training!

print('finished video analysis!, beginning retraining...')

deeplabcut.create_training_dataset(path_config_file)

print('finished retraining!, creating labellef videos...')
#Create labeled video

deeplabcut.create_labeled_video(path_config_file, avi_list)

print('finished creating labeled videos!')
