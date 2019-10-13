#include <string>
#include <cstdlib>


int main(){
  int capas,neuronas,decremento;  
  double eta, validacion;
  for(capas=3;capas<4;capas++){
    for(neuronas=6;neuronas<26;neuronas++){
	  for(eta=0.1;eta<0.8;eta+=0.1){
	    std::string aux="../practica1 -t train.dat -T test.dat -i 5000 -l "+std::to_string(capas)+" -h "+std::to_string(neuronas)+" -e "+std::to_string(eta)+" -v "+std::to_string(validacion)+">>salida2.txt";
	    system(aux.c_str());
          }  
    }
  }
  return 0;
}
