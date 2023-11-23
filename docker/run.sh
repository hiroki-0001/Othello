#/bin/bash 

docker_image="ubuntu20"

if [ "$#" -lt 1 ]; then 
	echo "Usage ${0} <input_file>"
	exit
fi

echo ${1}

xhost +
docker run -it --rm -e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix:rw -e QT_X11_NO_MITSHM=1 --name osero -v ${1}:/osero:rw ${docker_image} /bin/bash
