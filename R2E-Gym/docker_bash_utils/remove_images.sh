images=$(docker images -a -q)
if [ -z "$images" ]; then
    echo "No images to remove"
    exit 0
fi
docker rmi -f $(docker images -a -q)