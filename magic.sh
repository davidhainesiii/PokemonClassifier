#!/usr/bin/env zsh

for f in *
do
  echo "resizing file $f"
  convert $f -resize 224x224! "resize_$f"
  mv $f originals/

done
