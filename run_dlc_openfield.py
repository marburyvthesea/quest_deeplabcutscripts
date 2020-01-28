import sys
sys.path.append('/home/jma819/DeepLabCut/')



path_config_file = '/projects/p30771/dlc_analysis/Open_Field-JJM-2020-01-26/config.yaml'

# create training dataset
deeplabcut.create_training_dataset(path_config_file)

# train network 
deeplabcut.train_network(path_config_file)

#
deeplabcut.evaluate_network(path_config_file, plotting=True)

#
videofile_path = ['videos/video3.avi','videos/video4.avi'] #Enter a folder OR a list of videos to analyze.

#
deeplabcut.analyze_videos(path_config_file,videofile_path, videotype='.avi')

#Following the refinement of labels and appending them to the original dataset, 
#this creates a new iteration of training dataset. This is automatically set in the config.yaml file, 
#so let's get training!

deeplabcut.create_training_dataset(path_config_file)

#Create labeled video

deeplabcut.create_labeled_video(path_config_file,videofile_path)