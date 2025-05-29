LOCAL_IMAGE=/home/twubt/containers/vllm.sqsh
srun --account lsdisttrain --nodes 1 --gpus-per-node 1 --no-container-mount-home --container-remap-root --container-mounts=/home/twubt/.cache/huggingface:/root/.cache/huggingface,/home/twubt/workspace:/workspace --container-workdir=/workspace --container-writable --container-image $LOCAL_IMAGE --pty bash
