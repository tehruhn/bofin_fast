#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <map>
#include <iterator>
#include <cmath>
#include <complex>
#include <iostream>
#include <functional>
#include <numeric>

using namespace std;

// ######## UTILITY FUNCTIONS

// interfaces with python for the bosonic part
pair<map<pair<int, int >, complex<double> >, int > boson_py_interface(complex<double>*, int, complex<double>*, int, int, complex<double>*, int, complex<double>*, int, int, int, int, int);

// interfaces with python for the fermionic part
pair<map<pair<int, int >, complex<double> >, int > fermion_py_interface(complex<double>*, int, complex<double>*, int, int, complex<double>*, int, complex<double>*, int, int*, int, int, int);

// adds or subtracts value in tuple at position k
vector<int> add_at_idx(vector<int> seq, int k, int val);

// calculates previous hierarchy index for 'n'
vector<int> prevhe(vector<int> current_he, int k, int ncut);

// calculates next hierarchy index for 'n'
vector<int> nexthe(vector<int> current_he, int k, int ncut);

// enumerates all states for n sites with k possible vals
vector<vector<int> > state_number_enumerate(vector<int> dims, int k);

// actually enumerates all the states
void generate_states(int pos, int sum, vector<int> ans, int n, int k, vector<int>& dims, vector<vector<int> >& states);

// for identity kron from left
vector<vector<complex<double> > > spre(vector<vector<complex<double> > > A);

// for identity kron from right
vector<vector<complex<double> > > spost(vector<vector<complex<double> > > A); 
 
// returns HEOM state dictionaries
pair<int, pair< map<int, vector<int> > , map< vector<int>, int> > > _heom_state_dictionaries(vector<int> dims, int excitations);

// return conjugate transpose
vector<vector<complex<double> > > dagger(vector<vector<complex<double> > >);

// returns transpose
vector<vector<complex<double> > > transpose(vector<vector<complex<double> > > mat);

// Runge Kutta
vector<vector<complex<double> > > RungeKutta(float x0, vector<complex<double> > y0, float x, float h, map<pair<int,int> , complex<double> > sparse_matrix);

// given function for Runge Kutta
// multiply with matrix of form map<pair<int,int> , complex<double> > L_helems
vector<complex<double> > comp_function(float t, vector<complex<double> > rho, const map<pair<int,int> , complex<double> >& sparse_matrix);

// HEOM solver class
class HEOM{

	// ######## PUBLIC ATTRIBUTES
	public:
	// system hamiltonian
	vector<vector<complex<double> > > hamiltonian;
	// coupling operator
	vector<vector<vector<complex<double> > > > coupling;
	// both ck and vk are column vectors, but not so in implementation
	// list of amplitudes in correlation fn expansion
	vector<complex<double> > ck;
	// list of frequencies in correlation fn expansion
	vector<complex<double> > vk;
	// hierarchy cutoff
	int ncut;
	// matsubara frequencies cutoffs
	int kcut;
	// number of real vals in list
	int nr;
	// number of imaginary vals in list
	int ni;
	// sparse L_helems matrix
	int nhe;
	map<pair<int,int> , complex<double> > L_helems;
	vector<int> xs;
	vector<int> ys;
	vector<complex<double> > data;
	vector<int> offsets;
	int dict_size;
	long long int total_nhe;

	//  ######## PRIVATE ATTRIBUTES
	private:
	map< int, vector<int> > idx2he;
	map< vector<int>, int> he2idx;
	int N;
	pair<long long int, int> hshape;
	// Liouvillian
	pair<int, int> grad_shape;
	vector<vector<complex<double> > > L;
	vector<vector<vector<complex<double> > > > spreQ;
	vector<vector<vector<complex<double> > > > spostQ;
	vector<vector<vector<complex<double> > > > spreQdag;
	vector<vector<vector<complex<double> > > > spostQdag;

	// ######## MEMBER FUNCTIONS
	public:
	// constructor	
	HEOM();

	// setters for attributes
	void set_hamiltonian(vector<vector<complex<double> > >);
	void set_coupling(vector<vector<vector<complex<double> > > >);
	void set_ck(vector<complex<double> >);
	void set_vk(vector<complex<double> >);
	void set_ncut(int ncut1);
	void set_nr(int nr1);
	void set_ni(int ni1);
	void set_offsets(vector<int>);

	// initialize private variables for bosonic case
	void boson_initialize();
	// initialize private variables when given Liouvillian for bosonic case
	void boson_initializeWithL();
	// initialize private variables for fermionic case
	void fermion_initialize();
	// initialize private variables when given Liouvillian for fermionic case
	void fermion_initializeWithL();
	// given index list, populates graph of next and previous elements
	void populate(vector<int> heidx_list);

	// get gradient term for hierarchy ADM at level n for bosonic case
	void boson_grad_n(vector<int> he_n);

	// get previous gradient for bosonic case
	void boson_grad_prev(vector<int> he_n, int k, vector<int> prev_he);

	// get next gradient for bosonic case
	void boson_grad_next(vector<int> he_n, int k, vector<int> next_he);

	// make the rhs for bosonic case
	void boson_rhs();

	// get gradient term for hierarchy ADM at level n for fermionic case
	void fermion_grad_n(vector<int> he_n);

	// get previous gradient for fermionic case
	void fermion_grad_prev(vector<int> he_n, int k, vector<int> prev_he, int idx);

	// get next gradient for fermionic case
	void fermion_grad_next(vector<int> he_n, int k, vector<int> next_he, int idx);

	// make the rhs for fermionic case
	void fermion_rhs();

};

#endif