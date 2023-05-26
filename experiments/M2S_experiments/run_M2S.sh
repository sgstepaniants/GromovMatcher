#!/bin/bash

# Loading the required module
source /etc/profile

module load anaconda/2022b

# Run the script
matlab -nodisplay -nosplash -r "M2S_script $1 $2"
