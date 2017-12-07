# Joint VAE Model API
# Variational autoencoder for images and text

---
Describes the code used to create results for the following paper:

Vedantam, Ramakrishna, Ian Fischer, Jonathan Huang, and Kevin Murphy. 2017.
*Generative Models of Visually Grounded Imagination.*
arXiv [cs.LG]. arXiv. http://arxiv.org/abs/1705.10762.

## Usage:
NOTE: All scripts should be run from the root directory of the project.

### Basic Setup

#### Install basic dependencies.
`cd` to the root directory and run the following command.

```
source install/install_deps.sh
```
This sets up the python virtual environment for the project, and downloads
necessary files for MNIST-A experiments.

### Additional Data Setup
To create your own MNISTA dataset see `scripts/create_affine_mnist.sh`.
Set appropriate paths in `datasets/mnist_attributes/affine_mnist_dataset_iid.py` and `datasets/mnist_attributes/affine_mnist_dataset_comp.py` respectively.

To download and process CELEBA dataset run `scripts/process_and_write_celeba.sh`.
And set appropriate paths in `datasets/celeba/celeba_dataset.py`.

### Experiments
See scripts/ for example uses of different models/ experiments reported in the
paper.
  1) `iclr_comp_mnista_fresh.sh`: Experiments on compositional split of MNISTA.
	2) `iclr_mnista_fresh.sh`: Experiments on iid split of MNISTA.
	3) `celeba_training.sh`: Experiments on CelebA dataset.

## Disclaimer:

This project is not an official Google project. It is not supported by Google
and Google specifically disclaims all warranties as to its quality,
merchantability, or fitness for a particular purpose.


## Contributing:

See how to [contribute](./CONTRIBUTING.md).


## License:

[Apache 2.0](./LICENSE).
