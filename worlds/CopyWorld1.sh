#!/bin/bash

echo "hello world"

rm -r ToCopy
mkdir ToCopy

if [[ -z $@ ]]; then
   num=5
else
   num=$@
fi

echo $num

for ((i = 0 ; i < $num ; i++ ))
do
   #echo "Welcome $i times"
   cp -r World_Train ToCopy 
   mv ToCopy/World_Train ToCopy/$i

done

echo "Complete!"

