import os
os.environ["DLClight"]="True"
import sys
sys.path.append('/home/jma819/DeepLabCut/')
import deeplabcut


#path_config_file = '/projects/p30771/dlc_analysis/Open_Field_v2-JJM-2020-01-27/config.yaml' 
path_config_file = sys.argv[1]
#num_gpu = sys.argv[2]

# create training dataset
#deeplabcut.create_training_dataset(path_config_file)

# train network 
#deeplabcut.train_network(path_config_file, maxiters=100000)

#print('finished training!')
#
deeplabcut.evaluate_network(path_config_file, plotting=True)

print('finished evaluating')
#
#videofile_path = ['videos/video3.avi','videos/video4.avi'] #Enter a folder OR a list of videos to analyze.

#
#deeplabcut.analyze_videos(path_config_file,videofile_path, videotype='.avi')

#Following the refinement of labels and appending them to the original dataset, 
#this creates a new iteration of training dataset. This is automatically set in the config.yaml file, 
#so let's get training!

#deeplabcut.create_training_dataset(path_config_file)

#Create labeled video

#deeplabcut.create_labeled_video(path_config_file,videofile_path)
