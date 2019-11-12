#!/bin/bash

case "${1%%.*}" in
 overview)
	convert -density 400 overview.pdf -resize 25% -crop 55x30+190+180% overview.png
	;;
 loop_closure)
	convert -density 400 loop_closure.pdf -resize 25% -crop 55x18+180+150% loop_closure.png
	;;
 *)
 	echo "
	Usage: ${0##*/} <filename.pdf>

	"
esac

