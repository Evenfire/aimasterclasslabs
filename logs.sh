#!/bin/sh

FILE=$1
#new=".new_log_$FILE"
new=".new_log"
old=".old_log"

function clear_n_lines {
# N=$1
# let i=0
# while [[ $i -lt $N ]]
# do
	echo "\033c"
	# echo "\r\033[1A"
# let i=$i+1
# done
}


let lines=0
while [ 1 ]
do
	/Users/pkirsch/Library/Python/2.7/bin/floyd logs $1 > $new
	TAIL=$(cat $new | tail -40)
	if [[ -a $old ]]; then
		RESULT=$(diff --speed-large-files -s $old $new)
		if [[ $RESULT != *"identical"* ]]; then
			# lines=$(cat $new | wc -l | tr -d '[:space:]')
			clear_n_lines #$lines
			echo "floyd logs $FILE"
			cat $new
			cat $new > $old
			echo "\c" > $new
		fi
	else
		cat $new
		cat $new > $old
		# lines=$(cat $new | wc -l | tr -d '[:space:]')
		echo "\c" > $new
	fi
	if [[ $TAIL == *"Job exited with status code:"* ]]; then
		rm -rf $old
		rm -rf $new
		exit 1
	fi
	sleep 2
done
