import os
os.environ["DLClight"]="True"
import sys
sys.path.append('/home/jma819/DeepLabCut/')
import deeplabcut


#path_config_file = '/projects/p30771/dlc_analysis/GRIN_rotarod_rearview-JJM-2020-04-02/config.yaml' 
path_config_file = sys.argv[1]

#

deeplabcut.convertannotationdata_fromwindows2unixstyle(path_config_file, False)

print('paths converted')

#



