#!/bin/bash

echo "hello world"

<<<<<<< HEAD
rm -r ToCopy/WorldTrain
mkdir ToCopy/WorldTrain

cp -r ToCopy/World1/. ToCopy/WorldTrain
=======
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
>>>>>>> origin/customWorld

echo "Complete!"

