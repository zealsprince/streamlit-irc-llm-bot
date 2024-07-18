#!/bin/bash

rm -fr dist/
cxfreeze -c run.py --target-dir $1

cp app.py $1/

cp -r system/resources/.* $1/
