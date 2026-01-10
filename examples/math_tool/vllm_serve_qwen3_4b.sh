#!/bin/bash

# set -x  # debug mode, prints each command before executing it

# Getting the nodes names - extract hostnames from allocated nodes
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 --overlap -w "$head_node" hostname --ip-address)

SLURM_GPUS_PER_TASK=8  # sets 8 GPUs per task (Ray worker)

USER_CPUS_PER_TASK=24  # sets 24 CPUs per task

# if we detect a space character in the head node IP, we'll convert it to an ipv4 address.
if [[ "$head_node_ip" == *" "* ]]; then
IFS=' ' read -ra ADDR <<<"$head_node_ip"
# splits IP string into arrays
# selects shorter (IPv4) or longer (IPv6) address based on length
if [[ ${#ADDR[0]} -gt 16 ]]; then
  head_node_ip=${ADDR[1]}
else
  head_node_ip=${ADDR[0]}
fi
echo "IPV6 address detected. We split the IPV4 address as $head_node_ip"
fi

# port selection from all ports between 49152 and 65535 that are not in use
port=$(comm -23 <(seq 49152 65535) <(ss -tan | awk '{print $4}' | cut -d':' -f2 | grep "[0-9]\{1,5\}" | sort | uniq) | shuf | head -n 1)
ip_head=$head_node_ip:$port
export ip_head  # export the IP address to be used by the Ray workers
echo "IP Head: $ip_head"

echo "Starting HEAD at $head_node"
# start the Ray head, runs in background
srun --nodes=1 --ntasks=1 -w "$head_node" \
    ray start --head --node-ip-address="$head_node_ip" --port=$port \
    --num-cpus "${USER_CPUS_PER_TASK}" --num-gpus ${SLURM_GPUS_PER_TASK} --block &


# number of worker nodes other than the head node
worker_num=$((SLURM_JOB_NUM_NODES - 1))

for ((i = 1; i <= worker_num; i++)); do
    node_i=${nodes_array[$i]}
    echo "Starting WORKER $i at $node_i"
    # start the worker node while connecting to the head node, runs in background
    srun --nodes=1 --ntasks=1 --cpus-per-task="${USER_CPUS_PER_TASK}" -w "$node_i" \
        ray start --address "$ip_head" \
        --num-cpus "${USER_CPUS_PER_TASK}" --num-gpus "${SLURM_GPUS_PER_TASK}" --block &
    sleep 5
done

sleep 5  # ensure all works connect


echo "Starting vllm serve process"
srun --nodes=1 --ntasks=1 --overlap -w "$head_node" vllm serve /fsx/zyhang/Qwen/Qwen3-4B \
  --trust-remote-code \
  --seed=1 \
  --port 8001 \
  --served-model-name "qwen3_4b_serve" \
  --tensor-parallel-size=$SLURM_GPUS_PER_TASK \
  --gpu-memory-utilization 0.95 \
  --enforce-eager