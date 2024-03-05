#!/bin/bash

echo "hello world"

rm -r ToCopy/WorldTrain
mkdir ToCopy/WorldTrain

cp -r ToCopy/World1/. ToCopy/WorldTrain

echo "Complete!"

