# This bash preamble figures out which server we are on, and accordingly
# sets paths to some directories of interest.

# In general, we care about the following things:
#   * Where is our data stored?
#   * Where are we storing and loading checkpoints?
#   * Miscellaneous things like pretrained model weights etc.
source ~/venv/imagination/bin/activate
if [[ "${HOME}" == *"vrama91"* ]]; then
	echo "On VT servers."
	GLOBAL_WHICH_SERVER="VT"
	GLOBAL_RUNS_PATH="${HOME}/runs"
	GLOBAL_DATA_PATH="/ssd_local/rama/datasets"
elif [[ "${HOME}" == *"rvedantam3"* ]]; then
	echo "On GT servers."
	GLOBAL_WHICH_SERVER="GT"
	GLOBAL_RUNS_PATH="/coc/scratch/rvedantam3/runs/"
	GLOBAL_DATA_PATH="/srv/share/datasets"
elif [[ "${HOME}" == *"cfarhomes"* ]]; then
	echo "On UMD servers."
	GLOBAL_WHICH_SERVER="UMD"
	GLOBAL_RUNS_PATH="/cfarhomes/vrama91/runs/"
	GLOBAL_DATA_PATH="/cfarhomes/vrama91/data/"
	module add cuda/8.0.44 cudnn/v5.1
fi
