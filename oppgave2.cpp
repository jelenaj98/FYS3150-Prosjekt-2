#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <string>
#include "time.h"
#include<armadillo>

#include <UnitTest++/UnitTest++.h>

#ifndef UNITTEST
    #define private public
    #define protected public
#endif

using namespace std;
using namespace arma;

/*************************

Jacobi's method

**************************/
class Jacobi  {
    protected:
        double DiagConst, NondiagConst, Step, RMax, RMin;
        int Dim;

        int iterations;
        double const tolerance = 1.0e-10;
    
        // initial matrix
        mat A;
    
        //Results - eigenvalues, eigvector, armadillo_eigenvalues
        vec eigenvalues, armadillo_eigenvalues;
        mat eigvector;

    public:
        Jacobi();
        Jacobi(int Dim, double RMax);  // for unit testing
        void print();
        void run();
        
    private:
        void calculateEigenvalues();
        void calculateArmadilloEigenvalues();
        void inputValues();
        void initValues();
        virtual void initialMatrix(); // Each child object will call its own implementation of the fuction initialMatrix
        double maxofdiag(int*,int*);
        void rotate (int k,int l);
        void fillEigenvalues();
};

void Jacobi::calculateArmadilloEigenvalues() {
    eig_sym(armadillo_eigenvalues, A);
}

void Jacobi::inputValues() {
    cout <<"Dim = ";
    cin >> Dim;

    cout <<"RMax = ";
    cin >> RMax;
}

Jacobi::Jacobi() { 
    inputValues();

    initValues();
}


Jacobi::Jacobi(int Dim, double RMax) { 
    this->Dim = Dim;
    this->RMax = RMax;

    initValues();
}

void Jacobi::initValues() {
    A = zeros<mat>(Dim,Dim);
    eigvector = zeros<mat>(Dim,Dim);
    eigenvalues = zeros<vec>(Dim);
    armadillo_eigenvalues = zeros<vec>(Dim);


    RMin = 0.0;
    Step    = RMax/ Dim;
    DiagConst = 2.0 / (Step*Step);
    NondiagConst =  -1.0 / (Step*Step);
}

void Jacobi::run() {
    initialMatrix();
    calculateEigenvalues();
    calculateArmadilloEigenvalues();
}

// Setting up tridiagonal matrix and diagonalization using Armadillo
void Jacobi::initialMatrix() {
    A(0,0) = DiagConst;
    A(0,1) = NondiagConst;
    for(int i = 1; i < Dim-1; i++) {
        A(i,i-1)    = NondiagConst;
        A(i,i)    = DiagConst;
        A(i,i+1)    = NondiagConst;
    }
    A(Dim-1,Dim-2) = NondiagConst;
    A(Dim-1,Dim-1) = DiagConst;

    A.print("Start matrix");
}

// Function to find the maximum matrix element
double Jacobi::maxofdiag (int*k,int*l ){
    double max = 0.0;
    for(int i = 0; i < Dim; i++ ){
        for(int j = i + 1; j < Dim; j++ ){
            if( fabs(A(i,j)) > max ){
                max = fabs(A(i,j));
                *l = i;
                *k = j;
            }
        }
    }

    return max;
}

// Function to find the values of cos and sin
void Jacobi::rotate (int k,int l){
    double s, c;
    if( A(k,l) != 0.0 ){
        double t, tau;
        tau = (A(l,l) - A(k,k))/(2*A(k,l));
        if( tau > 0 ){
            t = 1.0/(tau + sqrt(1.0 + tau*tau));
        }
        else{
            t = -1.0/( -tau + sqrt(1.0 + tau*tau));
        }
        c = 1/sqrt(1+t*t);
        s = c*t;
    }
    else{
        c = 1.0;
        s = 0.0;
    }
    double a_kk, a_ll, a_ik, a_il, r_ik, r_il;
    a_kk = A(k,k);
    a_ll = A(l,l); // changing the matrix elements with indices k and l
    A(k,k) = c*c*a_kk - 2.0*c*s*A(k,l) + s*s*a_ll;
    A(l,l) = s*s*a_kk + 2.0*c*s*A(k,l) + c*c*a_ll;
    A(k,l) = 0.0; // hard-coding of the zeros
    A(l,k) = 0.0; // and then we change the remaining elements
    for(int i = 0; i < Dim; i++ ){
        if( i != k && i != l ){
            a_ik = A(i,k);
            a_il = A(i,l);
            A(i,k) = c*a_ik - s*a_il;
            A(k,i) = A(i,k);
            A(i,l) = c*a_il + s*a_ik;
            A(l,i) = A(i,l);
        } // Computing the new eigenvectors
        r_ik = eigvector(i,k);
        r_il = eigvector(i,l);
        eigvector(i,k) = c*r_ik - s*r_il;
        eigvector(i,l) = c*r_il + s*r_ik;
    }
}

// Calculate eigen values using Jacobi method
void Jacobi::calculateEigenvalues() {
    int k, l;
    
    double max_number_iterations = (double) Dim*(double) Dim*(double) Dim;
    iterations = 0;
    double max_offdiag = maxofdiag ( &k, &l);
    while( fabs(max_offdiag) > tolerance && (double) iterations < max_number_iterations ){
        max_offdiag = maxofdiag ( &k, &l);
        rotate ( k, l);
        iterations++;
    }

    fillEigenvalues();
}

// Fill eigenvalues from the main diagonal of matrix A 
void Jacobi::fillEigenvalues() {
    //The eigenvalues of A will be on the diagonalof A, with eigenvalue i being A[i][i].
    for(int i = 0;i < Dim;i++) {
        eigenvalues[i] = A(i,i);
    }

    eigenvalues = sort(eigenvalues);
}

void Jacobi::print() {
    int dim = 5;
    if(Dim < dim)
        dim = Dim;
    cout << "RESULTS - The lowest five eigenvalues:" << endl;
    cout << setiosflags(ios::showpoint | ios::uppercase);
    cout <<"Number of Eigenvalues = " <<  Dim << endl;
    cout <<"Number of iterations: " << iterations << endl;
    cout << setw(30) << "Numerical eigenvalues" << setw(25) << "Armadillo eigenvalues" << setw(21) <<  "Difference" << endl;
    for(int i = 0; i < dim; i++) {
        cout <<  setprecision(12) << setw(23) << eigenvalues(i) << setw(25) << armadillo_eigenvalues(i) << setw(30) << fabs(eigenvalues(i)-armadillo_eigenvalues(i)) << endl;
  }
}

/*
TEST Jacobi
*/
TEST(MaxOfDiag) {
    int k, l;

    Jacobi my(5,1);

    my.A(0,3) = 3.0;
    my.A(1,4) = 5.6;
    
    CHECK_EQUAL(5.6, my.maxofdiag(&k,&l));
};


TEST(SortEigenValues) {
    
    // start matrix
    int dim = 5;
    double rmax = 1.0;
    Jacobi my(dim,rmax);
    for(int i = 0;i  < dim;i++) {
        for(int j = 0;j  < dim;j++) {
            if(i == j) {
                my.A(i,j) = (double)(10.0-i);
            } 
        }
    }

    // calculate eigenvalues
    my.fillEigenvalues();

    // expected values
    vec expectedResults {6.0, 7.0, 8.0, 9.0, 10.0};

    // Check and compare expected and calculated values
    for(int i = 0;i < dim;i++)
        CHECK_EQUAL(my.eigenvalues(i), expectedResults(i) );
};

/*************************

CLASS HarmonicOscillator_1e

**************************/
class HarmonicOscillator_1e : public Jacobi {
    protected:
        int lOrbital;
        double OrbitalFactor;
    
    // local memory for r and the potential w[r] 
        vec r, w;

        double potential(double);

    public:
        HarmonicOscillator_1e();
        
    private:    
        void initialMatrix();
        void inputValues();
};

HarmonicOscillator_1e::HarmonicOscillator_1e() : Jacobi () {

    inputValues();
    
    OrbitalFactor = lOrbital * (lOrbital + 1.0);

    // local memory for r and the potential w[r] 
    r = zeros<vec>(Dim);
    w = zeros<vec>(Dim);
    for(int i = 0; i < Dim; i++) {
        r(i) = RMin + (i+1) * Step;
        w(i) = potential(r(i)) + OrbitalFactor/(r(i) * r(i));
    }
}

void HarmonicOscillator_1e::inputValues() {
    cout <<"Orbital momentum = ";
    cin >> lOrbital;
}


/*
  The function potential()
  calculates and return the value of the 
  potential for a given argument x.
  The potential here is for the hydrogen atom
*/
double HarmonicOscillator_1e::potential(double x) {
    return x*x;
}

// Setting up tridiagonal matrix and diagonalization using Armadillo
void HarmonicOscillator_1e::initialMatrix() {

    A(0,0) = DiagConst + w(0);
    A(0,1) = NondiagConst;
    for(int i = 1; i < Dim-1; i++) {
        A(i,i-1) = NondiagConst;
        A(i,i) = DiagConst + w(i);
        A(i,i+1) = NondiagConst;
    }
    
    A(Dim-1,Dim-2) = NondiagConst;
    A(Dim-1,Dim-1) = DiagConst + w(Dim-1);

    A.print("Start matrix");
}



/*************************

CLASS HarmonicOscillator_2e

**************************/
class HarmonicOscillator_2e : public HarmonicOscillator_1e {
    double omega;
    
    public:
    HarmonicOscillator_2e();    
        
    private:
        void inputValues();
        void initialMatrix();
};


HarmonicOscillator_2e::HarmonicOscillator_2e() : HarmonicOscillator_1e() {
    inputValues();
    
    // local memory for r and the potential w[r] 
    r = zeros<vec>(Dim);
    w = zeros<vec>(Dim);
    for(int i = 0; i < Dim; i++) {
        r(i) = RMin + (i+1) * Step;
        w(i) = omega*omega*potential(r(i)) + 1/(r(i));
    }
}

void HarmonicOscillator_2e::inputValues() {
    cout <<"Omega = ";
    cin >> omega;
}


// Setting up tridiagonal matrix and diagonalization using Armadillo
void HarmonicOscillator_2e::initialMatrix() {
    A(0,0) = DiagConst + w(0);
    A(0,1) = NondiagConst;
    for(int i = 1; i < Dim-1; i++) {
        A(i,i-1) = NondiagConst;
        A(i,i) = DiagConst + w(i);
        A(i,i+1) = NondiagConst;
    }
    
    A(Dim-1,Dim-2) = NondiagConst;
    A(Dim-1,Dim-1) = DiagConst + w(Dim-1);

    A.print("Start matrix");
}

int main() {
    #define UNITTEST
    if(UnitTest::RunAllTests() != 0)
         exit(-1);
    #undef UNITTEST
    
    Jacobi J;
    J.run();
    J.print();

    HarmonicOscillator_1e ho1e;
    ho1e.run();
    ho1e.print();

    HarmonicOscillator_2e ho2e;
    ho2e.run();
    ho2e.print();

    return 0;
}