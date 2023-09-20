## PyTorch Backend Examples
[End to End Guide](https://medium.com/towards-data-science/deploying-pytorch-models-with-nvidia-triton-inference-server-bb139066a387)

### Container Startup

```
docker run --gpus all --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 
-v /home/ec2-user/SageMaker:/models nvcr.io/nvidia/tritonserver:23.08-py3 
tritonserver --model-repository=/models --exit-on-error=false --log-verbose=1
```
