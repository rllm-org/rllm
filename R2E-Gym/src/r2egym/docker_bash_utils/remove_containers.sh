VLLM_CONTAINER=$(docker ps -q --filter ancestor=vllm/vllm-openai:latest)
SGLANG_CONTAINER=$(docker ps -q --filter ancestor=lmsysorg/sglang:latest)

# VLLM_CONTAINER=""
# SGLANG_CONTAINER=""

# Combine container IDs to exclude (if they exist)
EXCLUDE_CONTAINERS=""
if [ -n "$VLLM_CONTAINER" ]; then
    EXCLUDE_CONTAINERS="$VLLM_CONTAINER"
fi

if [ -n "$SGLANG_CONTAINER" ]; then
    if [ -n "$EXCLUDE_CONTAINERS" ]; then
        EXCLUDE_CONTAINERS="$EXCLUDE_CONTAINERS|$SGLANG_CONTAINER"
    else
        EXCLUDE_CONTAINERS="$SGLANG_CONTAINER"
    fi
fi

if [ -z "$EXCLUDE_CONTAINERS" ]; then
    # No vLLM or SGLang containers found, but preserve their images
    echo "No vLLM or SGLang containers found. Stopping and removing all containers."
    docker stop $(docker ps -q)
    docker rm $(docker ps -a -q)
else
    echo "Preserving vLLM and SGLang containers. Stopping and removing all other containers."
    # Stop and remove all containers except the ones we want to preserve
    CONTAINERS_TO_STOP=$(docker ps -q | grep -v -E "$EXCLUDE_CONTAINERS")
    CONTAINERS_TO_REMOVE=$(docker ps -a -q | grep -v -E "$EXCLUDE_CONTAINERS")
    
    if [ -n "$CONTAINERS_TO_STOP" ]; then
        docker stop $CONTAINERS_TO_STOP
    fi
    
    if [ -n "$CONTAINERS_TO_REMOVE" ]; then
        docker rm $CONTAINERS_TO_REMOVE
    fi
fi
