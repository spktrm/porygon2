#!/bin/bash

cd data
ts-node src/main.ts

cd ../
python embeddings/main.py 