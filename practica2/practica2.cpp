//============================================================================
// Introducción a los Modelos Computacionales
// Name        : practica1.cpp
// Author      : Pedro A. Gutiérrez
// Version     :
// Copyright   : Universidad de Córdoba
//============================================================================


#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <iostream>
#include <ctime>    // Para cojer la hora time()
#include <cstdlib>  // Para establecer la semilla srand() y generar números aleatorios rand()
#include <string.h>
#include <math.h>
#include "imc/PerceptronMulticapa.h"

using namespace imc;
using namespace std;

int main(int argc, char **argv) {
	// Procesar los argumentos de la línea de comandos
	  bool Tflag = 0,tflag=0,iflag=0, wflag = 0,sflag=0, pflag = 0,lflag=0,hflag=0,eflag=0,mflag=0,fflag=0,vflag=0,dflag=0,oflag=0;
	    char *Tvalue = NULL, *wvalue = NULL,*tvalue=NULL,*ivalue=NULL,*lvalue=NULL,*hvalue=NULL,*fvalue=NULL,*evalue=NULL,*mvalue=NULL,*vvalue=NULL,*dvalue=NULL;
	    int c;


    opterr = 0;

    // a: opción que requiere un argumento
    // a:: el argumento requerido es opcional
    while ((c = getopt(argc, argv, "t:T:w:i:l:h:e:m:v:d:posf:")) != -1)
    {
        // Se han añadido los parámetros necesarios para usar el modo opcional de predicción (kaggle).
        // Añadir el resto de parámetros que sean necesarios para la parte básica de las prácticas.
       switch(c){
        	case 't':
        		tflag = true;
        		tvalue = optarg;
        		break;
            case 'T':
                Tflag = true;
                Tvalue = optarg;
                break;
            case 'i':
                iflag = true;
                ivalue = optarg;
                break;
            case 'l':
                lflag = true;
                lvalue = optarg;
                break;
            case 'w':
                wflag = true;
                wvalue = optarg;
                break;
            case 'h':
                hflag = true;
                hvalue = optarg;
                break;
            case 'e':
                eflag = true;
                evalue = optarg;
                break;
            case 'm':
                mflag = true;
                mvalue = optarg;
                break;
            case 'v':
                vflag = true;
                vvalue = optarg;
                break;
            case 'd':
                dflag = true;
                dvalue = optarg;
                break;
            case 'f':
                fflag = true;
                fvalue = optarg;
                break;
            case 'p':
                pflag = true;
                break;
            case 'o':
            	oflag = true;
            	break;
            case 's':
            	sflag = true;
            	break;
            case '?':
                if (optopt == 'T' || optopt == 'w' || optopt == 'p')
                    fprintf (stderr, "La opción -%c requiere un argumento.\n", optopt);
                else if (isprint (optopt))
                    fprintf (stderr, "Opción desconocida `-%c'.\n", optopt);
                else
                    fprintf (stderr,
                             "Caracter de opción desconocido `\\x%x'.\n",
                             optopt);
                return EXIT_FAILURE;
            default:
                return EXIT_FAILURE;
        }
    }

    if (!pflag) {

        ////////////////////////////////////////
        // MODO DE ENTRENAMIENTO Y EVALUACIÓN //
        ///////////////////////////////////////

    	// Objeto perceptrón multicapa
    	PerceptronMulticapa mlp;
        if (!tflag){
              fprintf (stderr, "La opción -t es necesaría para la ejecución.\n");
              return EXIT_FAILURE;
        }
    	int iteraciones=1000;
    	int capas=1;
    	int neuronas=5;
    	int error=0;
    	if(!Tflag)
    		Tvalue=tvalue;
    	if(iflag)
    		iteraciones=atoi(ivalue);
    	if(lflag)
    		capas=atoi(lvalue);
    	if(hflag)
    		neuronas=atoi(hvalue);
    	if(eflag)
    		mlp.dEta=atof(evalue);
    	if(mflag)
    		mlp.dMu=atof(mvalue);
    	if(vflag)
    		mlp.dValidacion=atof(vvalue);
    	if(dflag)
    		mlp.dDecremento=atof(dvalue);
    	if(fflag)
    		error=atoi(fvalue);

    	mlp.bOnline=oflag;
        // Lectura de datos de entrenamiento y test: llamar a mlp.leerDatos(...)
    	Datos* pDatosTrain=mlp.leerDatos(tvalue);
        Datos* pDatosTest=mlp.leerDatos(Tvalue);
        // Inicializar vector topología
        int *topologia = new int[capas+2];
        int *tipoCapas = new int[capas+2];
        topologia[0] = pDatosTrain->nNumEntradas;
        tipoCapas[0] = 0;
        for(int i=1; i<(capas+2-1); i++){
        	tipoCapas[i]=0;
        	topologia[i] = neuronas;
        }
        topologia[capas+2-1] = pDatosTrain->nNumSalidas;
        if(sflag)
        	tipoCapas[capas+2-1]=1;
        else
        	tipoCapas[capas+2-1]=0;
        // Inicializar red con vector de topología
        mlp.inicializar(capas+2,topologia,tipoCapas);


        // Semilla de los números aleatorios
        int semillas[] = {1,2,3,4,5};
        double *errores = new double[5];
        double *erroresTrain = new double[5];
        double *erroresTest = new double[5];
        double *ccrs = new double[5];
        double *ccrsTrain = new double[5];
        double mejorErrorTest = 1.0;
        for(int i=0; i<5; i++){
        	cout << "**********" << endl;
        	cout << "SEMILLA " << semillas[i] << endl;
        	cout << "**********" << endl;
    		srand(semillas[i]);
    		mlp.ejecutarAlgoritmo(pDatosTrain,pDatosTest,iteraciones,&(erroresTrain[i]),&(errores[i]),&(ccrsTrain[i]),&(ccrs[i]),error);
    		cout << "Finalizamos => CCR de test final: " << ccrs[i] << endl;

            // (Opcional - Kaggle) Guardamos los pesos cada vez que encontremos un modelo mejor.
            if(wflag && erroresTest[i] <= mejorErrorTest)
            {
                mlp.guardarPesos(wvalue);
                mejorErrorTest = erroresTest[i];
            }

        }


        double mediaErrorTest = 0, desviacionTipicaErrorTest = 0;
        double mediaErrorTrain = 0, desviacionTipicaErrorTrain = 0;
        double mediaCCR = 0, desviacionTipicaCCR = 0;
        double mediaCCRTrain = 0, desviacionTipicaCCRTrain = 0;

        // Calcular medias y desviaciones típicas de entrenamiento y test
		//Calcular medias y desviaciones típicas de entrenamiento y test
		for(int i=0; i<5; i++){
			mediaCCR+= ccrs[i];
			mediaCCRTrain= ccrsTrain[i];
			mediaErrorTrain += erroresTrain[i];
			mediaErrorTest += erroresTest[i];
		}
		mediaCCRTrain/=5;
		mediaCCR/=5;
		mediaErrorTest/=5;
		mediaErrorTrain/=5;

		double auxTest=0;
		double auxTrain=0;
		double auxCCRTest=0;
		double auxCCRTrain=0;
		for(int i=0;i<5;i++){
			auxCCRTest += pow(ccrs[i]-mediaCCR,2);
			auxCCRTrain += pow(ccrsTrain[i]-mediaCCRTrain,2);
			auxTest += pow(erroresTest[i]-mediaErrorTest,2);
			auxTrain += pow(erroresTrain[i]-mediaErrorTrain,2);
		}
		desviacionTipicaCCRTrain= sqrt(auxCCRTrain/4);
		desviacionTipicaCCR = sqrt(auxCCRTest/4);
		desviacionTipicaErrorTest = sqrt(auxTest/4);
		desviacionTipicaErrorTrain = sqrt(auxTrain/4);

        cout << "HEMOS TERMINADO TODAS LAS SEMILLAS" << endl;

    	cout << "INFORME FINAL" << endl;
    	cout << "*************" << endl;
        cout << "Error de entrenamiento (Media +- DT): " << mediaErrorTrain << " +- " << desviacionTipicaErrorTrain << endl;
        cout << "Error de test (Media +- DT): " << mediaErrorTest << " +- " << desviacionTipicaErrorTest << endl;
        cout << "CCR de entrenamiento (Media +- DT): " << mediaCCRTrain << " +- " << desviacionTipicaCCRTrain << endl;
        cout << "CCR de test (Media +- DT): " << mediaCCR << " +- " << desviacionTipicaCCR << endl;
    	return EXIT_SUCCESS;
    } else {

        /////////////////////////////////
        // MODO DE PREDICCIÓN (KAGGLE) //
        ////////////////////////////////

        // Desde aquí hasta el final del fichero no es necesario modificar nada.
        
        // Objeto perceptrón multicapa
        PerceptronMulticapa mlp;

        // Inicializar red con vector de topología
        if(!wflag || !mlp.cargarPesos(wvalue))
        {
            cerr << "Error al cargar los pesos. No se puede continuar." << endl;
            exit(-1);
        }

        // Lectura de datos de entrenamiento y test: llamar a mlp.leerDatos(...)
        Datos *pDatosTest;
        pDatosTest = mlp.leerDatos(Tvalue);
        if(pDatosTest == NULL)
        {
            cerr << "El conjunto de datos de test no es válido. No se puede continuar." << endl;
            exit(-1);
        }

        mlp.predecir(pDatosTest);

        return EXIT_SUCCESS;

    }
}

