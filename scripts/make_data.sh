#!/bin/bash

cd data
ts-node src/main.ts
ts-node src/randoms.ts

cd ../
python embeddings/main.py 