# /*
#  * Project: deep-adaptive-optics
#  * Created on: Wednesday, 6th November 2019 9:47:12 am
#  * --------
#  * @author: hmcgovern
#  */

# -e
# -o

########################## THIS WILL BE MASTER LOOP ##############################
# 1. start the SLM GUI (Wiebke's code)
# 2. call abberior.py to acquire image, pass it through model and get predicted coefficients, pass coefficients through a function in the GUI code?
# need to think about how that refreshes

# string file path to a .pth file 
# there has to be a more eloquent way of doing this
# can be found under 
MODEL_NAME="08.01.20_corrected_pattern_calc_w_val_5_epochs_Adam_lr_0.001_batchsize_64_custom_loss.pth"

########################### DO NOT MODIFY BELOW THIS LINE #########################
USER=$(whoami)
OUTPUT_DIR="/c/Users/${USER}/Seafile/My Library"
MODEL_STORE_PATH="${OUTPUT_DIR}/Models/${MODEL_NAME}"
PARAMETER_PATH=

python aberrior.py "${MODEL_STORE_PATH}"
# python Patterns_Python_MAIN.py &

# python ../hello.py &

# IMAGE_DIR="Z:/Measurements/Abberior/Data/20191104_PythonMachineAlignment_HopeWiebke/20191104_0102_AuNP_AllZernike00_8bit.tif"
# IMAGE_DIR="Z:/Measurements/Abberior/Alignment_WJ_PV_NAD/20191104_Routine_PV/20191104_0102_100xSil_AuSTED_before.msr"
# # IMAGE_DIR="Z:/Measurements/Abberior/Alignment_WJ_PV_NAD/20191104_Routine_PV/20191104_0103_100xSil_AuSTED_after.msr"
# MODEL_STORE_PATH="/c/Users/hmcgover/Seafile/My Library/Models/fixed_zern_old_10_epochs_SGD_lr_0.1_w_val.pth"


# python aberrior.py "$IMAGE_DIR" "${MODEL_STORE_PATH}" 



# eof