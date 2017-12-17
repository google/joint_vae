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

Running above scripts uses [slurm](https://slurm.schedmd.com/) to run/launch jobs on clusters, so only
run the above if you have access to a cluster with slurm installed on it. If you dont have access to slurm simply run the corresponding
python commands in the above scripts with the appropriate parameters.

---
Each experiment is associated with three jobs: `train`, `eval` and `imeval` (imagination evaluation).

Example commands to run experiments with slurm:
  * `source scripts/iclr_mnista_fresh.sh ''` (to run all 3 jobs for every experiment)
  * `source scripts/iclr_mnista_fresh.sh train` (to launch only training jobs)
  * `source scripts/iclr_mnista_fresh.sh eval` (to launch only eval jobs)
	* `source scripts/iclr_mnista_fresh.sh imeval` (to launch only imagination eval jobs)

If you dont have access to slurm, you can see what command line arguments are used
for running experiments (in the above scripts) and run those commands in bash.

### Quantiative Results
See the ipython notebook `experiments/iclr_results_aggregate.ipynb` on how to view
the imagination results after running imeval (imagination evaluation) jobs, post
training.

## Contributors
* Ramakrishna Vedantam
* Hernan Moraldo
* Ian Fischer

## Disclaimer:

This project is not an official Google project. It is not supported by Google
and Google specifically disclaims all warranties as to its quality,
merchantability, or fitness for a particular purpose.


## Contributing:

See how to [contribute](./CONTRIBUTING.md).


## License:

[Apache 2.0](./LICENSE).
