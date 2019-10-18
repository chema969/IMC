#!/bin/bash
./practica2 -t $1 -T $2 -l $4 -h $5 -v 0.0 -d 1 -s -f 1 >$3.txt
./practica2 -t $1 -T $2 -l $4 -h $5 -v 0.15 -d 1 -s -f 1 >>$3.txt
./practica2 -t $1 -T $2 -l $4 -h $5 -v 0.25 -d 1 -s -f 1 >>$3.txt
./practica2 -t $1 -T $2 -l $4 -h $5 -v 0.0 -d 2 -s -f 1 >>$3.txt
./practica2 -t $1 -T $2 -l $4 -h $5 -v 0.15 -d 2 -s -f 1 >>$3.txt
./practica2 -t $1 -T $2 -l $4 -h $5 -v 0.25 -d 2 -s -f 1 >>$3.txt
