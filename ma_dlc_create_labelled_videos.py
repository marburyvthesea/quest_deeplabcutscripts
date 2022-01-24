import os
os.environ["DLClight"]="True"
import sys
sys.path.append('/home/jma819/DeepLabCut/')
import deeplabcut


#path_config_file = '/projects/p30771/dlc_analysis/GRIN_rotarod_rearview-JJM-2020-04-02/config.yaml' 
path_config_file = sys.argv[1]


behav_cam_directory = sys.argv[2]

avi_list = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(behav_cam_directory)) for f in fn if f.endswith('.avi')]

scorername = 'DLC_resnet50_rotarod_side_view_multi_animalAug25shuffle1_200000'

deeplabcut.create_video_with_all_detections(path_config_file, avi_list, DLCscorername=scorername)

print('finished creating labeled videos!')
