#!/bin/bash
./practica2 -t $1 -T $2 -l $4 -h $5 -s -f 0 -v 0.0 -d 1 >$3.txt
./practica2 -t $1 -T $2 -l $4 -h $5 -s -f 1 -v 0.0 -d 1  >>$3.txt
./practica2 -t $1 -T $2 -l $4 -h $5 -f 0 -v 0.0 -d 1  >>$3.txt
