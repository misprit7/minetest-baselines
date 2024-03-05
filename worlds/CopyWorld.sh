#!/bin/bash

echo "hello world"

rm -r worlds/ToCopy
mkdir worlds/ToCopy

if [[ -z $@ ]]; then
   num=5
else
   num=$@
fi

echo $num

for ((i = 0 ; i < $num ; i++ ))
do
   #echo "Welcome $i times"
   cp -r worlds/World_Train worlds/ToCopy 
   mv worlds/ToCopy/World_Train worlds/ToCopy/$i

done

echo "Complete!"

