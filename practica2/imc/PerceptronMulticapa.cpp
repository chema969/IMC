/*********************************************************************
 * File  : PerceptronMulticapa.cpp
 * Date  : 2018
 *********************************************************************/

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cstdlib>  // Para establecer la semilla srand() y generar números aleatorios rand()
#include <limits>
#include <math.h>
#include <algorithm>
#include "PerceptronMulticapa.h"
#include "util.h"

using namespace imc;
using namespace std;
using namespace util;


// ------------------------------
// CONSTRUCTOR: Dar valor por defecto a todos los parámetros (dEta, dMu, dValidacion y dDecremento)
PerceptronMulticapa::PerceptronMulticapa(){
	pCapas = NULL;
	dDecremento = 1;
	nNumCapas=2;
	dEta=0.1;
	dMu=0.9;
	dDecremento=1;
	dValidacion=0;
	bOnline=false;
	nNumPatronesTrain=1;
}

// ------------------------------
// Reservar memoria para las estructuras de datos
// nl tiene el numero de capas y npl es un vector que contiene el número de neuronas por cada una de las capas
// tipo contiene el tipo de cada capa (0 => sigmoide, 1 => softmax)
// Rellenar vector Capa* pCapas
int PerceptronMulticapa::inicializar(int nl, int npl[], int tipo[]) {

	if(nl>2){
		nNumCapas=nl;
		pCapas=new Capa[nl];
		for(int i=0;i<nl;i++){
			pCapas[i].nNumNeuronas=npl[i];
			pCapas[i].tipo=tipo[i];
			pCapas[i].pNeuronas=new Neurona[npl[i]];
			for(int j=0;j<npl[i];j++){
				Neurona *nuevaNeurona= new Neurona;
				pCapas[i].pNeuronas[j]=*nuevaNeurona;
				if(i>0){
					pCapas[i].pNeuronas[j].w = new double[pCapas[i-1].nNumNeuronas+1];
					pCapas[i].pNeuronas[j].wCopia = new double[pCapas[i-1].nNumNeuronas+1];
					pCapas[i].pNeuronas[j].deltaW = new double[pCapas[i-1].nNumNeuronas+1];
					pCapas[i].pNeuronas[j].ultimoDeltaW = new double[pCapas[i-1].nNumNeuronas+1];

				}
			}
		}
	}
	return 1;
}


// ------------------------------
// DESTRUCTOR: liberar memoria
PerceptronMulticapa::~PerceptronMulticapa() {
	liberarMemoria();
}


// ------------------------------
// Liberar memoria para las estructuras de datos
void PerceptronMulticapa::liberarMemoria() {
	for(int i=0;i<nNumCapas;i++){
		for(int j=0;j<pCapas[i].nNumNeuronas;j++){
			if(i!=0){
				delete pCapas[i].pNeuronas[j].deltaW;
				delete pCapas[i].pNeuronas[j].ultimoDeltaW;
				delete pCapas[i].pNeuronas[j].w;
				delete pCapas[i].pNeuronas[j].wCopia;
			}
		}
		delete[] pCapas[i].pNeuronas;

	}
	delete[] pCapas;
}

// ------------------------------
// Rellenar todos los pesos (w) aleatoriamente entre -1 y 1
void PerceptronMulticapa::pesosAleatorios() {
	for(int i=1;i<nNumCapas;i++){
		for(int j=0;j<pCapas[i].nNumNeuronas;j++){
			for(int k=0;k<pCapas[i-1].nNumNeuronas+1;k++){
				pCapas[i].pNeuronas[j].w[k]=pow(-1,rand())*static_cast <float> (rand()) / (static_cast <float> (RAND_MAX));
			}
		}
	}
}

// ------------------------------
// Alimentar las neuronas de entrada de la red con un patrón pasado como argumento
void PerceptronMulticapa::alimentarEntradas(double* input) {
	for(int i=0;i<pCapas[0].nNumNeuronas;i++){
		pCapas[0].pNeuronas[i].x=input[i];
	}

}

// ------------------------------
// Recoger los valores predichos por la red (out de la capa de salida) y almacenarlos en el vector pasado como argumento
void PerceptronMulticapa::recogerSalidas(double* output){
	for(int i=0; i < pCapas[nNumCapas-1].nNumNeuronas; i++){
		output[i] = pCapas[nNumCapas-1].pNeuronas[i].x;
	}
}

// ------------------------------
// Hacer una copia de todos los pesos (copiar w en copiaW)
void PerceptronMulticapa::copiarPesos() {
	for(int i=1;i<nNumCapas;i++){
		for(int j=0;j<pCapas[i].nNumNeuronas;j++){
			for(int k=0;k<pCapas[i-1].nNumNeuronas+1;k++){
				pCapas[i].pNeuronas[j].wCopia[k]=pCapas[i].pNeuronas[j].w[k];
			}
		}
	}
}

// ------------------------------
// Restaurar una copia de todos los pesos (copiar copiaW en w)
void PerceptronMulticapa::restaurarPesos() {
	for(int i=1;i<nNumCapas;i++){
		for(int j=0;j<pCapas[i].nNumNeuronas;j++){
			for(int k=0;k<pCapas[i-1].nNumNeuronas+1;k++){
				pCapas[i].pNeuronas[j].w[k]=pCapas[i].pNeuronas[j].wCopia[k];
			}
		}
	}
}

// ------------------------------
// Calcular y propagar las salidas de las neuronas, desde la primera capa hasta la última
void PerceptronMulticapa::propagarEntradas() {
	for(int i=1;i<nNumCapas;i++){
		if(pCapas[i].tipo==0){
			for(int j=0;j<pCapas[i].nNumNeuronas;j++){
				double sigmoidSum=pCapas[i].pNeuronas[j].w[0];
				for(int k=0;k<pCapas[i-1].nNumNeuronas+1;k++){
				   sigmoidSum+=pCapas[i].pNeuronas[j].w[k+1]*pCapas[i-1].pNeuronas[k].x;
				}
				pCapas[i].pNeuronas[j].x=(double)1/(1+ exp(-1*sigmoidSum));
			}
		}
		else{
			double sumSoftMax=0;
			for(int j=0;j<pCapas[i].nNumNeuronas;j++){
				double sigmoidSum=pCapas[i].pNeuronas[j].w[0];
				for(int k=0;k<pCapas[i-1].nNumNeuronas+1;k++){
				   sigmoidSum+=pCapas[i].pNeuronas[j].w[k+1]*pCapas[i-1].pNeuronas[k].x;
				}
				pCapas[i].pNeuronas[j].x=exp(sigmoidSum);
				sumSoftMax+=pCapas[i].pNeuronas[j].x;
			}
			for(int j=0;j<pCapas[i].nNumNeuronas;j++){
				pCapas[i].pNeuronas[j].x=pCapas[i].pNeuronas[j].x/sumSoftMax;
			}
		}
	}
}

// ------------------------------
// Calcular el error de salida del out de la capa de salida con respecto a un vector objetivo y devolverlo
// funcionError=1 => EntropiaCruzada // funcionError=0 => MSE
double PerceptronMulticapa::calcularErrorSalida(double* target, int funcionError) {
	double error=0.0;
	if(funcionError==0){
	for(int i=0;i<pCapas[nNumCapas-1].nNumNeuronas;i++){
			error+=pow(target[i]-pCapas[nNumCapas-1].pNeuronas[i].x,2);
		}
	error=(double)error/pCapas[nNumCapas-1].nNumNeuronas;
	}
	else{
		for(int i=0;i<pCapas[nNumCapas-1].nNumNeuronas;i++){
			error+=-1*log(pCapas[nNumCapas-1].pNeuronas[i].x)*target[i];
		}
	error=(double)error/pCapas[nNumCapas-1].nNumNeuronas;
	}
	return error;
}


// ------------------------------
// Retropropagar el error de salida con respecto a un vector pasado como argumento, desde la última capa hasta la primera
// funcionError=1 => EntropiaCruzada // funcionError=0 => MSE
void PerceptronMulticapa::retropropagarError(double* objetivo, int funcionError) {
	if(pCapas[nNumCapas-1].tipo==0){
		if(funcionError==0){
			for(int i=0;i<pCapas[nNumCapas-1].nNumNeuronas;i++){
				double out=pCapas[nNumCapas-1].pNeuronas[i].x;
				pCapas[nNumCapas-1].pNeuronas[i].dX=-1*(objetivo[i]-out)*(1-out)*out;
			}
		}
		else{
			for(int i=0;i<pCapas[nNumCapas-1].nNumNeuronas;i++){
					double out=pCapas[nNumCapas-1].pNeuronas[i].x;
					pCapas[nNumCapas-1].pNeuronas[i].dX=-1*(objetivo[i]/out)*(1-out)*out;
				}
			}
	}
	else{
			if(funcionError==0){
				for(int j=0;j<pCapas[nNumCapas-1].nNumNeuronas;j++){
					double sumder=0.0;
					for(int i=0;i<pCapas[nNumCapas-1].nNumNeuronas;i++){
						double outj=pCapas[nNumCapas-1].pNeuronas[j].x;
						double outi=pCapas[nNumCapas-1].pNeuronas[i].x;
						if(i!=j)
							sumder+=-1*(objetivo[i]-outi)*-outi*outj;
						else
							sumder+=-1*(objetivo[i]-outj)*(1-outj)*outj;
					}
					pCapas[nNumCapas-1].pNeuronas[j].dX=sumder;
				}
			}
			else{
				for(int j=0;j<pCapas[nNumCapas-1].nNumNeuronas;j++){
					double sumder=0.0;
					for(int i=0;i<pCapas[nNumCapas-1].nNumNeuronas;i++){
						double outj=pCapas[nNumCapas-1].pNeuronas[j].x;
						double outi=pCapas[nNumCapas-1].pNeuronas[i].x;
						if(i!=j)
							sumder+=-1*(objetivo[i]/outi)*(-outi)*outj;
						else
							sumder+=-1*(objetivo[i]/outj)*(1-outj)*outj;
					}
					pCapas[nNumCapas-1].pNeuronas[j].dX=sumder;
				}
			}
	}
    for(int j=nNumCapas-2;j>=1;j--){
    	for(int k=0;k<pCapas[j].nNumNeuronas;k++){
    		double sum=0.0;
    		for(int l=0;l<pCapas[j+1].nNumNeuronas;l++){
    			sum+=pCapas[j+1].pNeuronas[l].dX*pCapas[j+1].pNeuronas[l].w[k+1];
    		}
    		double out=pCapas[j].pNeuronas[k].x;
    		pCapas[j].pNeuronas[k].dX=sum*out*(1-out);
    	}
    }


}

// ------------------------------
// Acumular los cambios producidos por un patrón en deltaW
void PerceptronMulticapa::acumularCambio() {
	for(int i=1;i<nNumCapas;i++){
		for(int j=0;j<pCapas[i].nNumNeuronas;j++){
    		for(int k=1;k<pCapas[i-1].nNumNeuronas+1;k++){
    			pCapas[i].pNeuronas[j].ultimoDeltaW[k]=pCapas[i].pNeuronas[j].deltaW[k];
    			pCapas[i].pNeuronas[j].deltaW[k]+= pCapas[i].pNeuronas[j].dX * pCapas[i-1].pNeuronas[k-1].x;

    		}
			pCapas[i].pNeuronas[j].ultimoDeltaW[0]=pCapas[i].pNeuronas[j].deltaW[0];
    		pCapas[i].pNeuronas[j].deltaW[0]+=pCapas[i].pNeuronas[j].dX;
		}
	}

}

// ------------------------------
// Actualizar los pesos de la red, desde la primera capa hasta la última
void PerceptronMulticapa::ajustarPesos() {
	double eta=dEta;
	for(int i=1;i<nNumCapas;i++){
		for(int j=0;j<pCapas[i].nNumNeuronas;j++){
    		for(int k=1;k<pCapas[i-1].nNumNeuronas+1;k++){
    			if(!bOnline)
    				pCapas[i].pNeuronas[j].w[k]=pCapas[i].pNeuronas[j].w[k]-((eta*pCapas[i].pNeuronas[j].deltaW[k])/nNumPatronesTrain)-((dMu*(eta*pCapas[i].pNeuronas[j].ultimoDeltaW[k]))/nNumPatronesTrain);
    			else
    				pCapas[i].pNeuronas[j].w[k]=pCapas[i].pNeuronas[j].w[k]-(eta*pCapas[i].pNeuronas[j].deltaW[k])-(dMu*(eta*pCapas[i].pNeuronas[j].ultimoDeltaW[k]));

    		}
			if(!bOnline)
				pCapas[i].pNeuronas[j].w[0]=pCapas[i].pNeuronas[j].w[0]-((eta*pCapas[i].pNeuronas[j].deltaW[0])/nNumPatronesTrain)-((dMu*(eta*pCapas[i].pNeuronas[j].ultimoDeltaW[0]))/nNumPatronesTrain);
			else
				pCapas[i].pNeuronas[j].w[0]=pCapas[i].pNeuronas[j].w[0]-(eta*pCapas[i].pNeuronas[j].deltaW[0])-(dMu*(eta*pCapas[i].pNeuronas[j].ultimoDeltaW[0]));
		}
		eta=pow(dDecremento,-(nNumCapas-i))*eta;
	}
}

// ------------------------------
// Imprimir la red, es decir, todas las matrices de pesos
void PerceptronMulticapa::imprimirRed() {
	for(int i=1;i<nNumCapas;i++){
		std::cout<<"Capa "<<i<<std::endl<<"========"<<std::endl;
		for(int j=0;j<pCapas[i].nNumNeuronas;j++){
    		for(int k=0;k<pCapas[i-1].nNumNeuronas+1;k++){
    			std::cout<<pCapas[i].pNeuronas[j].w[k]<<"\t";
    		}
    		std::cout<<std::endl;
		}
	}
}

// ------------------------------
// Simular la red: propragar las entradas hacia delante, computar el error, retropropagar el error y ajustar los pesos
// entrada es el vector de entradas del patrón, objetivo es el vector de salidas deseadas del patrón.
// El paso de ajustar pesos solo deberá hacerse si el algoritmo es on-line
// Si no lo es, el ajuste de pesos hay que hacerlo en la función "entrenar"
// funcionError=1 => EntropiaCruzada // funcionError=0 => MSE
void PerceptronMulticapa::simularRed(double* entrada, double* objetivo, int funcionError) {
		alimentarEntradas(entrada);
		propagarEntradas();
		retropropagarError(objetivo,funcionError);
		acumularCambio();
		if(bOnline)
			ajustarPesos();
}

// ------------------------------
// Leer una matriz de datos a partir de un nombre de fichero y devolverla
Datos* PerceptronMulticapa::leerDatos(const char *archivo) {
	  std::ifstream fichero (archivo);
	  Datos* datos_fichero=new Datos;
	  fichero>>datos_fichero->nNumEntradas>>datos_fichero->nNumSalidas>>datos_fichero->nNumPatrones;

	  datos_fichero->entradas=new double*[datos_fichero->nNumPatrones];
	  for(int i = 0; i < datos_fichero->nNumPatrones; i++)
		  datos_fichero->entradas[i] = new double[datos_fichero->nNumEntradas];

	  datos_fichero->salidas=new double*[datos_fichero->nNumPatrones];
	  for(int i = 0; i < datos_fichero->nNumPatrones; i++)
		  datos_fichero->salidas[i] = new double[datos_fichero->nNumSalidas];

	  for(int i=0;i<datos_fichero->nNumPatrones;i++){
		  for(int j=0;j<datos_fichero->nNumEntradas;j++){
			  fichero>>datos_fichero->entradas[i][j];
		  }
		  for(int k=0;k<datos_fichero->nNumSalidas;k++){
			  fichero>>datos_fichero->salidas[i][k];
		  }
	  }
	  return datos_fichero;
}


// ------------------------------
// Entrenar la red para un determinado fichero de datos (pasar una vez por todos los patrones)
void PerceptronMulticapa::entrenar(Datos* pDatosTrain, int funcionError) {
	if(bOnline){
		for(int r=0;r<pDatosTrain->nNumPatrones;r++){
			for(int i=1; i < this->nNumCapas; i++){
				for(int j=0; j < this->pCapas[i].nNumNeuronas; j++){
					for(int k=0; k < this->pCapas[i-1].nNumNeuronas + 1; k++){
						this->pCapas[i].pNeuronas[j].deltaW[k] = 0.0;
					}
				}
			}
			simularRed(pDatosTrain->entradas[r],pDatosTrain->salidas[r],funcionError);
		}
	}
	else{
		for(int i=1; i < this->nNumCapas; i++){
			for(int j=0; j < this->pCapas[i].nNumNeuronas; j++){
				for(int k=0; k < this->pCapas[i-1].nNumNeuronas + 1; k++){
					this->pCapas[i].pNeuronas[j].deltaW[k] = 0.0;
				}
			}
		}
		for(int r=0;r<pDatosTrain->nNumPatrones;r++)
			simularRed(pDatosTrain->entradas[r],pDatosTrain->salidas[r],funcionError);
		ajustarPesos();

	}
}

// ------------------------------
// Probar la red con un conjunto de datos y devolver el error cometido
// funcionError=1 => EntropiaCruzada // funcionError=0 => MSE
double PerceptronMulticapa::test(Datos* pDatosTest, int funcionError) {
	double mse=0.0;
	for(int i=0;i<pDatosTest->nNumPatrones;i++){
		alimentarEntradas(pDatosTest->entradas[i]);
		propagarEntradas();
		mse+=calcularErrorSalida(pDatosTest->salidas[i],funcionError);
	}
	return mse/pDatosTest->nNumPatrones;
}

// OPCIONAL - KAGGLE
// Imprime las salidas predichas para un conjunto de datos.
// Utiliza el formato de Kaggle: dos columnas (Id y Predicted)
void PerceptronMulticapa::predecir(Datos* pDatosTest)
{
	int i;
	int j;
	int numSalidas = pCapas[nNumCapas-1].nNumNeuronas;
	double * salidas = new double[numSalidas];
	
	cout << "Id,Predicted" << endl;
	
	for (i=0; i<pDatosTest->nNumPatrones; i++){

		alimentarEntradas(pDatosTest->entradas[i]);
		propagarEntradas();
		recogerSalidas(salidas);

		int maxIndex = 0;
		for (j = 0; j < numSalidas; j++)
			if (salidas[j] >= salidas[maxIndex])
				maxIndex = j;
		
		cout << i << "," << maxIndex << endl;

	}
}

// ------------------------------
// Probar la red con un conjunto de datos y devolver el CCR
double PerceptronMulticapa::testClassification(Datos* pDatosTest,int ** matrizConf) {
	double ccr=0.0;
	int numSalidas = pCapas[nNumCapas-1].nNumNeuronas;

	if(matrizConf!=NULL){
		for(int i=0;i<numSalidas;i++)
			for(int k=0;k<numSalidas;k++)
				matrizConf[i][k]=0;
	}
	double * salidas = new double[numSalidas];
	for(int i=0;i<pDatosTest->nNumPatrones;i++){
		alimentarEntradas(pDatosTest->entradas[i]);
		propagarEntradas();
		recogerSalidas(salidas);

		double mayor=salidas[0];
		int indiceMayor=0;
		for(int j=1;j<numSalidas;j++){
			if(mayor<salidas[j]){
				mayor=salidas[j];
				indiceMayor=j;
			}
		}
		if(pDatosTest->salidas[i][indiceMayor]==1){
			ccr++;
		}
		if(matrizConf!=NULL){
			int j=0;
			for(int k=0;k<numSalidas;k++){
				if(pDatosTest->salidas[i][k]==1)
					j=k;
			}
			matrizConf[indiceMayor][j]++;
		}
	}
	
	return (ccr/pDatosTest->nNumPatrones)*100;
}

// ------------------------------
// Ejecutar el algoritmo de entrenamiento durante un número de iteraciones, utilizando pDatosTrain
// Una vez terminado, probar como funciona la red en pDatosTest
// Tanto el error MSE de entrenamiento como el error MSE de test debe calcularse y almacenarse en errorTrain y errorTest
// funcionError=1 => EntropiaCruzada // funcionError=0 => MSE
void PerceptronMulticapa::ejecutarAlgoritmo(Datos * pDatosTrain, Datos * pDatosTest, int maxiter, double *errorTrain, double *errorTest, double *ccrTrain, double *ccrTest, int funcionError,int** matrizconf)
{
	int countTrain = 0;

	// Inicialización de pesos
	pesosAleatorios();
	double testError = 0;
	double minTrainError = 0;
	int numSinMejorar;
	nNumPatronesTrain = pDatosTrain->nNumPatrones;
	double minValidationError=0;
	double numSinMejorarValidacion=0;
	double validationError=0.0;
	Datos *pDatosValidacion=NULL;

	// Generar datos de validación
	if(dValidacion > 0 && dValidacion < 1){
		int* vectordeelegidos=vectorAleatoriosEnterosSinRepeticion(0,pDatosTrain->nNumPatrones-1,round(pDatosTrain->nNumPatrones*dValidacion));
		pDatosValidacion=new Datos;
		pDatosValidacion->nNumPatrones=round(pDatosTrain->nNumPatrones*dValidacion);
		pDatosValidacion->nNumEntradas=pDatosTrain->nNumEntradas;
		pDatosValidacion->nNumSalidas=pDatosTrain->nNumSalidas;

		pDatosValidacion->entradas=new double*[pDatosValidacion->nNumPatrones];
		  for(int i = 0; i < pDatosValidacion->nNumPatrones; i++)
			  pDatosValidacion->entradas[i] = new double[pDatosValidacion->nNumEntradas];

		pDatosValidacion->salidas=new double*[pDatosValidacion->nNumPatrones];
		  for(int i = 0; i < pDatosValidacion->nNumPatrones; i++)
			 pDatosValidacion->salidas[i] = new double[pDatosValidacion->nNumSalidas];

		  double** entrTrain=new double*[pDatosTrain->nNumPatrones-pDatosValidacion->nNumPatrones];
		  		  for(int i = 0; i < pDatosTrain->nNumPatrones-pDatosValidacion->nNumPatrones; i++)
		  			entrTrain[i] = new double[pDatosTrain->nNumEntradas];

		  double** saliTrain=new double*[pDatosTrain->nNumPatrones-pDatosValidacion->nNumPatrones];
		  	  	  for(int i = 0; i < pDatosTrain->nNumPatrones-pDatosValidacion->nNumPatrones; i++)
		  		     saliTrain[i] = new double[pDatosTrain->nNumSalidas];

		  sort(vectordeelegidos, vectordeelegidos+pDatosValidacion->nNumPatrones);

		  for(int i=0,j=0,k=0;i<pDatosTrain->nNumPatrones;i++){
			if(i==vectordeelegidos[j]){
				pDatosValidacion->entradas[j]=pDatosTrain->entradas[i];
				pDatosValidacion->salidas[j]=pDatosTrain->salidas[i];
				j++;
			}
			else{
				entrTrain[k]=pDatosTrain->entradas[i];
				saliTrain[i]=pDatosTrain->salidas[i];
				k++;
			}
		}
		pDatosTrain->nNumPatrones=pDatosTrain->nNumPatrones-pDatosValidacion->nNumPatrones;
		pDatosTrain->salidas=saliTrain;
		pDatosTrain->entradas=entrTrain;
	}


	// Aprendizaje del algoritmo
	do {
		entrenar(pDatosTrain,funcionError);

				double trainError = test(pDatosTrain,funcionError);
		if(dValidacion > 0 && dValidacion < 1){
			if(countTrain==0 || validationError < minValidationError){
						minValidationError = validationError;
						numSinMejorarValidacion = 0;
					}
					else if( (validationError-minValidationError) < 0.001)
						numSinMejorarValidacion = 0;
					else
						numSinMejorarValidacion++;
		}
		if(countTrain==0 || trainError < minTrainError){
			minTrainError = trainError;
			copiarPesos();
			numSinMejorar = 0;
		}
		else if( (trainError-minTrainError) < 0.001)
			numSinMejorar = 0;
		else
			numSinMejorar++;

		if(numSinMejorar>=50){
			cout << "Salida porque no mejora el entrenamiento!!"<< endl;
			restaurarPesos();
			countTrain = maxiter;
		}

		if(numSinMejorarValidacion>=50){
			cout << "Salida porque no mejora el error de validación!!"<< endl;
			restaurarPesos();
			countTrain = maxiter;
		}

		countTrain++;
		testError = test(pDatosTest,funcionError);
		countTrain++;
		cout << "Iteración " << countTrain << "\t Error de entrenamiento: " << trainError << "\t Error de test: " << testError << "\t Error de validacion: " << validationError << endl;
	} while ( countTrain<maxiter );
	if ( (numSinMejorarValidacion!=50) && (numSinMejorar!=50))
		restaurarPesos();

	cout << "PESOS DE LA RED" << endl;
	cout << "===============" << endl;
	imprimirRed();

	cout << "Salida Esperada Vs Salida Obtenida (test)" << endl;
	cout << "=========================================" << endl;
	for(int i=0; i<pDatosTest->nNumPatrones; i++){
		double* prediccion = new double[pDatosTest->nNumSalidas];

		// Cargamos las entradas y propagamos el valor
		alimentarEntradas(pDatosTest->entradas[i]);
		propagarEntradas();
		recogerSalidas(prediccion);
		for(int j=0; j<pDatosTest->nNumSalidas; j++)
			cout << pDatosTest->salidas[i][j] << " -- " << prediccion[j] << " \\\\ " ;
		cout << endl;
		delete[] prediccion;

	}

	*errorTest=test(pDatosTest,funcionError);;
	*errorTrain=minTrainError;
	*ccrTest = testClassification(pDatosTest);
	*ccrTrain = testClassification(pDatosTrain,matrizconf);

}

// OPCIONAL - KAGGLE
//Guardar los pesos del modelo en un fichero de texto.
bool PerceptronMulticapa::guardarPesos(const char * archivo)
{
	// Objeto de escritura de fichero
	ofstream f(archivo);

	if(!f.is_open())
		return false;

	// Escribir el numero de capas, el numero de neuronas en cada capa y el tipo de capa en la primera linea.
	f << nNumCapas;

	for(int i = 0; i < nNumCapas; i++)
	{
		f << " " << pCapas[i].nNumNeuronas;
		f << " " << pCapas[i].tipo;
	}
	f << endl;

	// Escribir los pesos de cada capa
	for(int i = 1; i < nNumCapas; i++)
		for(int j = 0; j < pCapas[i].nNumNeuronas; j++)
			for(int k = 0; k < pCapas[i-1].nNumNeuronas + 1; k++)
				f << pCapas[i].pNeuronas[j].w[k] << " ";

	f.close();

	return true;

}


// OPCIONAL - KAGGLE
//Cargar los pesos del modelo desde un fichero de texto.
bool PerceptronMulticapa::cargarPesos(const char * archivo)
{
	// Objeto de lectura de fichero
	ifstream f(archivo);

	if(!f.is_open())
		return false;

	// Número de capas y de neuronas por capa.
	int nl;
	int *npl;
	int *tipos;

	// Leer número de capas.
	f >> nl;

	npl = new int[nl];
	tipos = new int[nl];

	// Leer número de neuronas en cada capa y tipo de capa.
	for(int i = 0; i < nl; i++)
	{
		f >> npl[i];
		f >> tipos[i];
	}

	// Inicializar vectores y demás valores.
	inicializar(nl, npl, tipos);

	// Leer pesos.
	for(int i = 1; i < nNumCapas; i++)
		for(int j = 0; j < pCapas[i].nNumNeuronas; j++)
			for(int k = 0; k < pCapas[i-1].nNumNeuronas + 1; k++)
				f >> pCapas[i].pNeuronas[j].w[k];

	f.close();
	delete[] npl;

	return true;
}
