#!/usr/bin/env bash

if [ $# -eq 2 ]; then
	input_video_path=$1
	output_video_path=$2

	if [ -f $input_video_path ]; then
		if [ ! -d .venv ]; then
			python3 -m venv .venv
			source .venv/bin/activate
			python -m pip install -U pip
			pip install -r requirements.txt
		else
			source .venv/bin/activate
		fi

		python script.py $input_video_path $output_video_path
	else 
		echo "Video not found"
	fi
else
	echo "Usage: ./run.sh [INPUT VIDEO PATH] [OUTPUT VIDEO PATH]"
fi
