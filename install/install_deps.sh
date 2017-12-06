# Install dependencies for the imagination project.

# Create a virtualenvironment.
if [ ! -e ${HOME}/venv ]; then
	mkdir ${HOME}/venv
fi

cd ${HOME}/venv

virtualenv --python=/usr/bin/python2.7 --system-site-packages imagination 
source ${HOME}/venv/imagination/bin/activate

# Install tensorflow > 1.2 needed for sonnet.
pip install --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.2.1-cp27-none-linux_x86_64.whl

# Install sonnet.
pip install dm-sonnet

# Install some other goodies.
pip install yapf
