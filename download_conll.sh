#!bin/bash

conda create --name py27 python=2.7 -y

conda activate py27
conda init

source /home/$USER/.bashrc
source ~/anaconda3/etc/profile.d/conda.sh
conda activate py27

pip install gdown
mkdir -p data/
cd data

gdown https://drive.google.com/u/0/uc?id=1jRm5Md-VCjxJNalCm3wErDJFUz0-0GhF
tar -xvzf LDC2013T19.tgz
rm LDC2013T19.tgz

gdown https://github.com/ontonotes/conll-formatted-ontonotes-5.0/archive/refs/tags/v12.tar.gz
tar -xvzf v12.tar.gz
rm v12.tar.gz
mv conll-formatted-ontonotes-5.0-12/ conll-2012/
cd conll-2012/
mv conll-formatted-ontonotes-5.0/ v4/
cd ../

gdown  http://conll.cemantix.org/2012/download/conll-2012-scripts.v3.tar.gz
tar -xvzf conll-2012-scripts.v3.tar.gz
rm conll-2012-scripts.v3.tar.gz

# data
# |---conll-formatted-ontonotes-5.0-12->conll-2012
# |   |---v3
# |   |---conll-formatted-ontonotes-5.0->v4
# |---ontonotes-release-5.0
#     |---data
#     |---docs
#     |---tools

# !!!! Optional !!!!
# go to /home/jylab_intern001/seal_ex/tmp/conll-2012/v3/scripts/skeleton2conll.sh 
# line 178 and remove arabic and chinese (because we do not use them)
#  e.g) for language in arabic english chinese; do -> for language in english; do

# If ExceptionError occurs, check conda py27 env activated 
bash ./conll-2012/v3/scripts/skeleton2conll.sh -D ./ontonotes-release-5.0/data/files/data ./conll-2012

cd conll-2012
mv v4/ v12/
