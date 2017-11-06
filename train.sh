export PYTHONPATH=/home/mcj/work/project/My_ReID/caffe/python:$PYTHONPATH
export PATH="/home/mcj/work/local/anaconda2/bin:$PATH"
echo $PATH
which python
source activate tensorflow
export CUDA_VISIBLE_DEVICES=${1}
python train.py ${2}
# cd caffe-tensorflow-master
# python convert.py --caffemodel ./VGG16.v2.caffemodel --data-output-path ./1.npy --code-output-path ./2  ./VGG_2014_16.prototxt 