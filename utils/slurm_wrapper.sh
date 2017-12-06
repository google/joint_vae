#! /bin/sh
sbatch<<EOT
#! /bin/sh
#SBATCH -p short
#SBATCH --job-name=${3}
#SBATCH --output=${4}/${3}.log
#SBATCH --nodes=1
#SBATCH --gres=gpu:1

echo "Starting job "${2}
echo "Running command" ${1}
PYTHONPATH="."
$1
EOT
