
This repository contains the code to reproduce the experiments of the paper.


### Instructions:

1. Install the required packages: `pip install -r requirements.txt`.

    We recommend creating and activating a virtual python environment first: `python3 -m venv venv`, `. venv/bin/activate`.
2. Download the TEgO dataset (https://iamlabumd.github.io/tego/) and extract it somewhere
3. set the `TEGO_ARGS` environment variable to `--tego /path/to/tego` or any other valid combination of arguments and run `run_all_experiments.sh`
    
    example: `TEGO_ARGS="--tego ./tego_ds --augment" ./run_all_experiments.sh`

The results will be written in the `results` directory.

Running all of the experiments takes about a day on a modern machine with a GPU, so we also provide our results here: https://drive.google.com/drive/folders/1vT9QI9cIPSvbumkogi82upH7eHrBsotk?usp=sharing

