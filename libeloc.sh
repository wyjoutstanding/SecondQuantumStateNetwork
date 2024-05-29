# usage: source libeloc.sh

# config original local energy libaray, ensure that has been complied
SRC_LIBELOC=../github_tmp/nnqs-eloc/NeuralNetworkQuantumState/local_energy/
TARGET_LIBELOC=libeloc # anything you want

# copy so and interface
mkdir -p ${TARGET_LIBELOC}
cp ${SRC_LIBELOC}interface/python/eloc.py ${TARGET_LIBELOC}/interface/python
cp ${SRC_LIBELOC}*.so ${TARGET_LIBELOC}

# export environment
cur_path=`pwd`
export PYTHONPATH=$cur_path/${TARGET_LIBELOC}/:$PYTHONPATH
export LD_LIBRARY_PATH=$cur_path/${TARGET_LIBELOC}/:$LD_LIBRARY_PATH
