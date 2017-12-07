# This bash preamble figures out which server we are on, and accordingly
# sets paths to some directories of interest.

# In general, we care about the following things:
#   * Where is our data stored?
#   * Where are we storing and loading checkpoints?
#   * Miscellaneous things like pretrained model weights etc.
source ~/venv/imagination/bin/activate
GLOBAL_RUNS_PATH="${PWD}/runs"
GLOBAL_DATA_PATH="${PWD}/data"
