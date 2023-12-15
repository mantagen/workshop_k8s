# ./
pip install torchserve torch-model-archiver
torch-model-archiver \
    --model-name mnist \
    --version 1.0 \
    --model-file ./mnist/net.py \
    --serialized-file ./mnist/mnist.pt \
    --handler ./mnist/handler.py \
    --export-path ./store \
    --force
torchserve \
    --start \
    --model-store ./store \
    --models mnist=mnist.mar
torchserve --stop
bash ./mnist/test.sh