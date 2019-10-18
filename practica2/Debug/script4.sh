#!/bin/bash

./script3.sh ../basesDatosPr2IMC/dat/train_xor.dat  ../basesDatosPr2IMC/dat/test_xor.dat xor_d_v 2 100
./script3.sh ../basesDatosPr2IMC/dat/train_vote.dat  ../basesDatosPr2IMC/dat/test_vote.dat vote_d_v 1 16
./script3.sh ../basesDatosPr2IMC/dat/train_nomnist.dat ../basesDatosPr2IMC/dat/test_nomnist.dat nomnist_d_v 1 64


