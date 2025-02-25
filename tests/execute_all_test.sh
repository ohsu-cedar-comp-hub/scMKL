# This script will test the most recent package version in scMKL/dist/ and run 
# all of the tests in scMKL/tests

# It does this by:
#   1) creating a new conda env
#   2) installing the local scMKL package into the env
#   3) running all of the test scripts
#   4) deleting the new conda env

# Creating conda env with python=3.12
conda create -n scmkl_test_env python=3.12 --yes

# Activating new env
conda init
conda activate scmkl_test_env

# Capturing newest package name
DIST=$(ls -1 ../dist | sort -n | tail -1)
pip install ../dist/${DIST}

# Running all of the test files
for TEST in $(ls -1 *.py); \
    do echo; \
    echo; \
    echo $TEST; \
    echo; \
    python $TEST; \
    done

# Deactivating env
conda init
conda deactivate

# Deleting env
conda remove -n scmkl_test_env --all --yes

echo
echo "Conda env removed, all tests completed"