#!/usr/bin/env zsh

for f in *
do
  echo "resizing file $f"
  convert $f -resize 224x224! "resize_$f"
  mv $f originals/

done

# Cog Ser API key 8fa52214-9ec1-456e-a5e9-4b4af4239f2b
#pre-trained model (coco, resnet50)
#model checkpoints
#
#
