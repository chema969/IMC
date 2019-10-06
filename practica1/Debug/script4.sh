#!/bin/bash

./script3.sh /home/chema969/IMC/practica1/bases_de_datos/train_xor.dat /home/chema969/IMC/practica1/bases_de_datos/test_xor.dat xor_d_v 2 100
./script3.sh /home/chema969/IMC/practica1/bases_de_datos/train_sin.dat /home/chema969/IMC/practica1/bases_de_datos/test_sin.dat sin_d_v 2 64
./script3.sh /home/chema969/IMC/practica1/bases_de_datos/train_quake.dat /home/chema969/IMC/practica1/bases_de_datos/test_quake.dat quake_d_v 1 100
./script3.sh /home/chema969/IMC/practica1/bases_de_datos/train_parkinsons.dat /home/chema969/IMC/practica1/bases_de_datos/test_parkinsons.dat parkinsons_d_v 2 100
