# Install dependencies for the imagination project.
#
# Run from the root directory of the project.

# Create a virtualenvironment.
if [ ! -e ${HOME}/venv ]; then
	mkdir ${HOME}/venv
fi

virtualenv --python=/usr/bin/python2.7 --system-site-packages ${HOME}/venv/imagination
source ${HOME}/venv/imagination/bin/activate

# Install tensorflow > 1.2 needed for sonnet.
pip install --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.2.1-cp27-none-linux_x86_64.whl

# Install sonnet.
pip install dm-sonnet

# Install some other goodies.
pip install yapf

# Create a few directories that we will use.
if [ ! -e "${PWD}/data/" ]; then
  mkdir ${PWD}/data
fi

if [ ! -e "${PWD}/runs/" ]; then
	mkdir ${PWD}/runs
fi

# Download the mnista dataset.
if [ ! -e "data/mnist_with_attributes" ]; then
  wget "https://filebox.ece.vt.edu/~vrama91/imagination/mnist_with_attributes.tar.gz"
  tar -xf mnist_with_attributes.tar.gz -C data/
  rm mnist_with_attributes.tar.gz
fi

# Download the mnista classifier checkpoint.
if [ ! -e "mnista_classifier_checkpoint" ]; then
  wget "https://filebox.ece.vt.edu/~vrama91/imagination/mnista_classifier_checkpoint.tar.gz"
  tar -xf mnista_classifier_checkpoint.tar.gz
  rm mnista_classifier_checkpoint.tar.gz
fi
