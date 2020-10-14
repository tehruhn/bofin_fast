#include "utilities.h"

#include <vector>
#include <map>
#include <iterator>
#include <cmath>
#include <complex>
#include <iostream>
#include <functional>
#include <numeric>
#include <chrono>
// my favorite macro of all time, in case anyone looks at this comment
// makes STL container traversal a breeze
// the code would look at least 10x uglier without this
#define traverse(container, it) \
for(auto it = container.begin(); it != container.end(); it++)

using namespace std;
using namespace std::chrono;

// adds or subtracts value in tuple at position k
vector<int> add_at_idx(vector<int> seq, int k, int val){
	seq[k] += val;
	return seq;
}

// calculates previous hierarchy index for 'n'
vector<int> prevhe(vector<int> current_he, int k, int ncut){
	vector<int> nprev = add_at_idx(current_he, k , -1);
	if(nprev[k] < 0){
		vector<int> ans(0);
		return ans;
	}
	else{
		return nprev;
	}
}

// calculates next hierarchy index for 'n'
vector<int> nexthe(vector<int> current_he, int k, int ncut){
	vector<int> nnext = add_at_idx(current_he, k, 1);
	int total = accumulate(nnext.begin(), nnext.end(), 0);
	if(total > ncut){
		vector<int> ans(0);
		return ans;
	}
	else{
		return nnext;
	}
}

// // enumerates all states for n sites with possible range given in vector
// vector<vector<int> > state_number_enumerate(vector<int> dims, int excitations){
// 	vector<vector<int> > arr;
// 	int n = (int)dims.size();
// 	for(int i = 0; i < n; i++){
// 		vector<int> temp(dims[i]);
// 		std::iota(temp.begin(), temp.end(), 0);
// 		arr.push_back(temp);
// 	}
// 	// computes combinations
// 	vector<int> temp(n);
// 	vector<vector<int> > states;
//     vector<int> indices(n, 0);
//     int next;
//     while(1){ 
//   		temp.clear();
//         for (int i = 0; i < n; i++) 
//             temp.push_back(arr[i][indices[i]]);  
//         long long int total = accumulate(temp.begin(), temp.end(), 0);
//         if(total <= excitations)
// 		    states.push_back(temp);
//         next = n - 1; 
//         while (next >= 0 &&  
//               (indices[next] + 1 >= (int)arr[next].size())) 
//             next--; 
//         if (next < 0) 
//             break; 
//         indices[next]++; 
//         for (int i = next + 1; i < n; i++) 
//             indices[i] = 0; 
//     }
//     return states;
// }

// enumerates all states for n sites with possible range given in vector
vector<vector<int> > state_number_enumerate(vector<int> dims, int excitations){
	
	int n = (int)dims.size();
	int k = excitations;
	vector<vector<int> > states;
	vector<int> ans(0);
	generate_states(0, 0, ans, n, k, dims, states);
	return states;
}

// actually enumerates all the states
void generate_states(int pos, int sum, vector<int> ans, int n, int k, vector<int>& dims, vector<vector<int> >& states){
	if(pos == n){
		states.push_back(ans);
		return;
	}
	for(int i = 0; i < dims[pos]; i++){
		if(sum + i > k)
			return;
		ans.push_back(i);
		generate_states(pos+1, sum+i, ans, n, k, dims, states);
		ans.pop_back();
	}
}

// returns HEOM state dictionaries
pair<int, pair< map<int, vector<int> > , map< vector<int>, int> > > _heom_state_dictionaries(vector<int> dims, int excitations){

	int nstates = 0;
	map< int, vector<int> > idx2state;
	map< vector<int>, int> state2idx;
	vector<vector<int> > states = state_number_enumerate(dims, excitations);
	for(int i = 0; i < (int)states.size(); i++){
		vector<int> temp = states[i];
		// for(int kk = 0; kk < (int)states[i].size(); kk++){
		// 	cout << states[i][kk] << " ";
		// }
		// cout << endl;
		state2idx[temp] = nstates;
		idx2state[nstates] = temp;
		nstates++;
	}
	// cout << "TEST" << nstates << endl;
	pair<int, pair< map<int, vector<int> > , map< vector<int>, int> > > ans;
	ans.first = nstates;
	ans.second.first = idx2state;
	ans.second.second = state2idx;
	return ans;

}

// interfaces with python, for the bosonic solver
pair<map<pair<int, int >, complex<double> >, int > boson_py_interface(complex<double>* H1, int szh, complex<double>* C1, int szc1, int szc, complex<double>* ck1, int szck, complex<double>* vk1, int szvk, int nc, int nr, int ni, int k){
	// process H into a matrix of doubles
	vector<vector<complex<double> > > H;
	for(int i = 0; i < szh; i++){
		vector<complex<double> > temp(H1+(i*szh), H1+(i*szh)+szh);
		H.push_back(temp);
	}
	// process C into a list of matrices of doubles
	vector<vector<vector<complex<double> > > > C;
	int num_ops = szc1/szc;
	if(num_ops == 1){
		vector<vector<complex<double> > > C_temp;
		for(int i = 0; i < szc; i++){
			vector<complex<double> > temp(C1+(i*szc), C1+(i*szc)+szc);
			C_temp.push_back(temp);
		}
		for(int i = 0; i < szck; i++){
			C.push_back(C_temp);
		}
	}
	else{
		for(int i = 0; i < num_ops; i++){
			int offset = i*szc*szc;
			vector<vector<complex<double> > > C_temp;
			for(int j = 0; j < szc; j++){
				vector<complex<double> > temp(C1+ offset +(j*szc), C1+ offset +(j*szc)+szc);
				C_temp.push_back(temp);
			}
			C.push_back(C_temp);
		}
	}

	vector<complex<double> > ck(ck1, ck1+szck);
	vector<complex<double> > vk(vk1, vk1+szvk);
	HEOM tester = HEOM();
	tester.set_hamiltonian(H);
	tester.set_coupling(C);
	tester.set_ck(ck);
	tester.set_vk(vk);
	tester.set_ncut(nc);
	tester.set_nr(nr);
	tester.set_ni(ni);
	if(k == 1)
		tester.boson_initialize();
	else
		tester.boson_initializeWithL();
	tester.boson_rhs();
	return make_pair(tester.L_helems, tester.nhe);
}




// constructor	
HEOM::HEOM(){

}


// setters for attributes
void HEOM::set_hamiltonian(vector<vector<complex<double> > > hamiltonian1){
	hamiltonian = hamiltonian1;
}
void HEOM::set_coupling(vector<vector<vector<complex<double> > > > coupling1){
	coupling = coupling1;
}
void HEOM::set_ck(vector<complex<double> > ck1){
	ck = ck1;
}
void HEOM::set_vk(vector<complex<double> > vk1){
	vk = vk1;
}
void HEOM::set_ncut(int ncut1){
	ncut = ncut1;
}
void HEOM::set_nr(int nr1){
	nr = nr1;
}
void HEOM::set_ni(int ni1){
	ni = ni1;
}
void HEOM::set_offsets(vector<int> offsets1){
	offsets =offsets1;
}

// initialize private variables for bosonic case
void HEOM::boson_initialize(){

	kcut = nr + ni + ((int)ck.size() - (nr + ni))/2;
	// initialize dicts
	vector<int> dms(kcut, ncut+1);

	auto start = high_resolution_clock::now(); 
	pair<int, pair< map<int, vector<int> > , map< vector<int>, int> > > ans = _heom_state_dictionaries(dms, ncut);
	auto stop = high_resolution_clock::now();
	auto duration = duration_cast<microseconds>(stop - start); 
  
	// cout << duration.count() << endl;  

	nhe = ans.first;
	he2idx = ans.second.second;
	idx2he = ans.second.first;
	// cout << nhe << endl;
	N = (int)hamiltonian.size();
	// total density matrices in hierarchy
	int numr = kcut + ncut;
	long long int fact = 1;
	for(int i = 1; i <= numr; i++){
		fact *= i;
	} 
	for(int i = 1; i <= ncut; i++){
		fact /= i;
	}
	for(int i = 1; i <= kcut; i++){
		fact /= i;
	}
	total_nhe = fact;
	grad_shape = make_pair(N*N, N*N);
	for(int i = 0; i < (int)coupling.size(); i++){
		spreQ.push_back(spre(coupling[i]));
	}
	for(int i = 0; i < (int)coupling.size(); i++){
		spostQ.push_back(spost(coupling[i]));
	}
	vector<vector<complex<double> > > spreH = spre(hamiltonian);
	vector<vector<complex<double> > > spostH = spost(hamiltonian);
	vector<vector<complex<double> > > tempmat((int)spreH.size(), vector<complex<double> >((int)spreH[0].size(), complex<double>(0,0)));
	for(int i = 0; i < (int)spreH.size(); i++){
		for(int j = 0; j < (int)spreH[0].size(); j++){
			tempmat[i][j] = complex<double>(0, -1)*(spreH[i][j] - spostH[i][j]);
		} 
	}
	L = tempmat;
}


// initialize private variables when given Liouvillian for bosonic case
void HEOM::boson_initializeWithL(){
	kcut = nr + ni + ((int)ck.size() - (nr + ni))/2;

	// initialize dicts
	vector<int> dms(kcut, ncut+1);
	pair<int, pair< map<int, vector<int> > , map< vector<int>, int> > > ans = _heom_state_dictionaries(dms, ncut);
	nhe = ans.first;
	he2idx = ans.second.second;
	idx2he = ans.second.first;
	N = int(sqrt((int)hamiltonian.size()));
	
	// total density matrices in hierarchy
	int numr = kcut + ncut;
	long long int fact = 1;
	for(int i = 1; i <= numr; i++){
		fact *= i;
	} 
	for(int i = 1; i <= ncut; i++){
		fact /= i;
	}
	for(int i = 1; i <= kcut; i++){

		fact /= i;
	}

	total_nhe = fact;

	grad_shape = make_pair(N*N, N*N);
	L = hamiltonian;
	for(int i = 0; i < (int)coupling.size(); i++){
		spreQ.push_back(spre(coupling[i]));
	}
	for(int i = 0; i < (int)coupling.size(); i++){
		spostQ.push_back(spost(coupling[i]));
	}
}


// kron with identity to the right
// put left matrix into right matrix
vector<vector<complex<double> > > spre(vector<vector<complex<double> > > A){
	vector<vector<complex<double> > > ans((int)A.size()*(int)A.size(), vector<complex<double> >((int)A.size()*(int)A.size(), 0));
	// now A kron identity
	for(int i = 0; i < (int)A.size()*(int)A.size(); i+=(int)A.size()){
		for(int j = 0; j < (int)A.size(); j++){
			for(int k = 0; k < (int)A.size(); k++){
				ans[i+j][i+k] = A[j][k];
			}
		}
	}
	return ans;

}

// kron with identity to the left
// put left matrix into right matrix
vector<vector<complex<double> > > spost(vector<vector<complex<double> > > A){
	A = transpose(A);
	vector<vector<complex<double> > > ans(A.size()*A.size(), vector<complex<double> >(A.size()*A.size(), 0));
	// now identity kron A
	for(int i = 0; i < (int)A.size(); i++){
		for(int j = 0; j < (int)A.size(); j++){
			for(int k = 0; k < (int)A.size(); k++){
				ans[i*(int)A.size()+k][j*(int)A.size()+k] = A[i][j]; 
			}
		}
	}
	return ans;

}

// get gradient term for hierarchy ADM at level n for bosonic case
void HEOM::boson_grad_n(vector<int> he_n){
	complex<double> gradient_sum = 0;
	int skip = 0;
	for(int i = 0; i < (int)vk.size(); i++){
		if(i < nr+ni){
			gradient_sum += (complex<double>(he_n[i], 0)*vk[i]);
		}
		else{
			if(skip){
				skip = 0; 
				continue;
			}
			else{
				int tot_fixed = nr + ni;
				int extra = (i+1-tot_fixed);
				int idx = tot_fixed + (extra/2) + (extra%2) -1;
				gradient_sum += (complex<double>(he_n[idx], 0)*vk[i]);
				skip = 1;
			}
		}
	}
	gradient_sum = complex<double>(-1, 0)*gradient_sum;
	vector<vector<complex<double> > > Lt = L;
	for(int i = 0; i < grad_shape.first; i++){
		Lt[i][i] += gradient_sum;
	}
	int nidx = he2idx[he_n];
	int block = grad_shape.first;
	int pos = nidx*block;
	for(int i = 0; i < grad_shape.first; i++){
		for(int j = 0; j < grad_shape.first; j++){
			if(norm(Lt[i][j]) != 0){
				L_helems[make_pair(pos+i, pos+j)] = Lt[i][j];
			}
		}
	}
	// cout << L_helems.size() << endl;
	// cout << L_helems[make_pair(30,30)] << endl;
}

// get previous gradient for bosonic case
void HEOM::boson_grad_prev(vector<int> he_n, int k, vector<int> prev_he){
	int nk = he_n[k];
	float norm_prev;
	vector<vector<complex<double> > > op1(grad_shape.first, vector<complex<double> >(grad_shape.first));
	if(k < nr){
		norm_prev = float(nk);
		for(int i = 0; i < grad_shape.first; i++){
			for(int j = 0; j < grad_shape.first; j++){
				op1[i][j] = complex<double>(0, -1)*complex<double>(norm_prev, 0)*ck[k]*(spreQ[k][i][j] - spostQ[k][i][j]);
			}
		}
	}
	else if( k >= nr && k < nr+ni){
		norm_prev = float(nk);
		for(int i = 0; i < grad_shape.first; i++){
			for(int j = 0; j < grad_shape.first; j++){
				op1[i][j] = complex<double>(0, -1)*complex<double>(norm_prev, 0)*complex<double>(0, 1)*ck[k]*(spreQ[k][i][j] + spostQ[k][i][j]);
			}
		}
	}
	else{
		norm_prev = float(nk);
		int k1 = nr + ni + 2*(k-(nr+ni));
		for(int i = 0; i < grad_shape.first; i++){
			for(int j = 0; j < grad_shape.first; j++){
				complex<double> term1 = (complex<double>(0, -1)*(ck[k1] * (spreQ[k][i][j] - spostQ[k][i][j])));
				complex<double> term2 = (ck[k1+1]*(spreQ[k][i][j] + spostQ[k][i][j]));
				op1[i][j] = complex<double>(norm_prev, 0)*(term1 + term2); 
			}
		} 
	}
	int rowidx = he2idx[he_n];
	int colidx = he2idx[prev_he];
	int block = grad_shape.first;
	int rowpos = rowidx*block;
	int colpos = colidx*block;
	for(int i = 0; i < grad_shape.first; i++){
		for(int j = 0; j < grad_shape.first; j++){
			if(norm(op1[i][j]) != 0){
				L_helems[make_pair(rowpos+i, colpos+j)] = op1[i][j];
			}
		}
	}
}

// get next gradient for bosonic case
void HEOM::boson_grad_next(vector<int> he_n, int k, vector<int> next_he){
	vector<vector<complex<double> > > op2(grad_shape.first, vector<complex<double> >(grad_shape.first));
	float norm_next;
	norm_next = 1;
	for(int i = 0; i < grad_shape.first; i++){
		for(int j = 0; j < grad_shape.first; j++){
			op2[i][j] = complex<double>(0, -1)*complex<double>(norm_next, 0)*(spreQ[k][i][j] - spostQ[k][i][j]);
		}
	}
	int rowidx = he2idx[he_n];
	int colidx = he2idx[next_he];
	int block = grad_shape.first;
	int rowpos = rowidx*block;
	int colpos = colidx*block;
	for(int i = 0; i < grad_shape.first; i++){
		for(int j = 0; j < grad_shape.first; j++){
			if(norm(op2[i][j]) != 0){
				L_helems[make_pair(rowpos+i, colpos+j)] = op2[i][j];
			}
		}
	}
}

// make the rhs for bosonic case
void HEOM::boson_rhs(){
	while(nhe < total_nhe){
		vector<int> heidxlist;
		traverse(idx2he, it){
			heidxlist.push_back(it->first);
		}
		populate(heidxlist);
	}
	vector<int> keylist;
	traverse(idx2he, it){
		keylist.push_back(it->first);
	}
	for(int i = 0; i < (int)keylist.size(); i++){
		vector<int> he_n = idx2he[keylist[i]];
		boson_grad_n(he_n);
		vector<int> next_he;
		vector<int> prev_he;
		for(int k = 0; k < kcut; k++){
			next_he = nexthe(he_n, k, ncut);
			prev_he = prevhe(he_n, k, ncut);
			if(((int)next_he.size() > 0) && (he2idx.find(next_he) != he2idx.end())){
				boson_grad_next(he_n, k , next_he);
			}
			if(((int)prev_he.size() > 0) && (he2idx.find(prev_he) != he2idx.end())){
				boson_grad_prev(he_n, k , prev_he);
			}
		}
	}
	// cout << "Dict size is : " << (int)L_helems.size() << endl;
	dict_size = L_helems.size();
	traverse(L_helems, it){
		xs.push_back(it->first.first);
		ys.push_back(it->first.second);
		data.push_back(it->second);
	}
	// cout << "checks :" << endl;
	// cout << (int)xs.size() << " " << (int)ys.size() << " " << (int)data.size() << endl;
}	

// given index list, populates graph of next and previous elements
void HEOM::populate(vector<int> heidx_list){
	for(int i = 0; i < (int)heidx_list.size(); i++){
		for(int k = 0; k < kcut; k++){
			vector<int> he_current = idx2he[i];
			vector<int> he_next = nexthe(he_current, k, ncut);
			vector<int> he_prev = prevhe(he_current, k, ncut);
			if(((int)he_next.size() > 0) && (he2idx.find(he_next)) == he2idx.end()){
				he2idx[he_next] = nhe;
				idx2he[nhe] = he_next;
				nhe++;
			}
			if(((int)he_prev.size() > 0) && (he2idx.find(he_prev)) == he2idx.end()){
				he2idx[he_prev] = nhe;
				idx2he[nhe] = he_prev;
				nhe++;
			}
		}
	}
}

// ##

// interfaces with python for the fermionic part
pair<map<pair<int, int >, complex<double> >, int > fermion_py_interface(complex<double>* H1, int szh, complex<double>* C1, int szc1, int szc, complex<double>* flat_ck1, int szck, complex<double>* flat_vk1, int szvk, int* len_list, int sz, int nc, int k){
	// process H into a matrix of doubles
	vector<vector<complex<double> > > H;
	for(int i = 0; i < szh; i++){
		vector<complex<double> > temp(H1+(i*szh), H1+(i*szh)+szh);
		H.push_back(temp);
	}
	vector<vector<vector<complex<double> > > > C;
	int num_ops = szc1/szc;
	if(num_ops == 1){
		vector<vector<complex<double> > > C_temp;
		for(int i = 0; i < szc; i++){
			vector<complex<double> > temp(C1+(i*szc), C1+(i*szc)+szc);
			C_temp.push_back(temp);
		}
		for(int i = 0; i < szck; i++){
			C.push_back(C_temp);
		}
	}
	else{
		for(int i = 0; i < num_ops; i++){
			int offset = i*szc*szc;
			vector<vector<complex<double> > > C_temp;
			for(int j = 0; j < szc; j++){
				vector<complex<double> > temp(C1+ offset +(j*szc), C1+ offset +(j*szc)+szc);
				C_temp.push_back(temp);
			}
			C.push_back(C_temp);
		}
	}
	vector<complex<double> > ck(flat_ck1, flat_ck1+szck);
	vector<complex<double> > vk(flat_vk1, flat_vk1+szvk);
	vector<int> szs(len_list, len_list+sz);
	vector<int> offsets;
	offsets.push_back(0);
	int curr_sum = 0;
	for(int i = 0; i < (int)szs.size(); i++){
		offsets.push_back(curr_sum+szs[i]);
		curr_sum += szs[i];
	}
	HEOM tester = HEOM();
	tester.set_hamiltonian(H);
	tester.set_coupling(C);
	tester.set_ck(ck);
	tester.set_vk(vk);
	tester.set_offsets(offsets);
	tester.set_ncut(nc);
	if(k == 1)
		tester.fermion_initialize();
	else
		tester.fermion_initializeWithL();
	tester.fermion_rhs();
	return make_pair(tester.L_helems, tester.nhe);
}


// initialize private variables for fermionic case
void HEOM::fermion_initialize(){

	kcut = offsets.size() - 1;
	// cout << "kcut " << kcut << endl;
	// initialize dicts
	vector<int> dms(ck.size(), 2);
	// cout << "cutoff: " << ncut << endl;
	pair<int, pair< map<int, vector<int> > , map< vector<int>, int> > > ans = _heom_state_dictionaries(dms, ncut);
	nhe = ans.first;
	he2idx = ans.second.second;
	idx2he = ans.second.first;
	N = (int)hamiltonian.size();
	
	// total density matrices in hierarchy
	int numr = kcut + ncut;
	long long int fact = 1;
	for(int i = 1; i <= numr; i++){
		fact *= i;
	} 
	for(int i = 1; i <= ncut; i++){
		fact /= i;
	}
	for(int i = 1; i <= kcut; i++){
		fact /= i;
	}
	total_nhe = fact;
	// cout << "N: " << N << endl;
	grad_shape = make_pair(N*N, N*N);
	for(int i = 0; i < (int)coupling.size(); i++){
		spreQ.push_back(spre(coupling[i]));
		spreQdag.push_back(spre(dagger(coupling[i])));
	}
	for(int i = 0; i < (int)coupling.size(); i++){
		spostQdag.push_back(spost(dagger(coupling[i])));
		spostQ.push_back(spost(coupling[i]));
	}
	vector<vector<complex<double> > > spreH = spre(hamiltonian);
	vector<vector<complex<double> > > spostH = spost(hamiltonian);
	vector<vector<complex<double> > > tempmat((int)spreH.size(), vector<complex<double> >((int)spreH[0].size(), complex<double>(0,0)));
	for(int i = 0; i < (int)spreH.size(); i++){
		for(int j = 0; j < (int)spreH[0].size(); j++){
			tempmat[i][j] = complex<double>(0, -1)*(spreH[i][j] - spostH[i][j]);
		} 
	}
	L = tempmat;
	// for(int k = 0; k < spreQdag.size(); k++){
	// 	for(int i = 0; i < spreQdag[0].size(); i++){
	// 		for(int j = 0; j < spreQdag[0][0].size(); j++){
	// 			cout << spreQdag[k][i][j] << " ";
	// 		}
	// 		cout << endl;
	// 	}
	// 	cout << endl;
	// }
}


// initialize private variables when given Liouvillian for fermionic case
void HEOM::fermion_initializeWithL(){

	kcut = offsets.size() -1;

	// cout << "kcut " << kcut << endl;
	// initialize dicts
	vector<int> dms(kcut, 2);
	pair<int, pair< map<int, vector<int> > , map< vector<int>, int> > > ans = _heom_state_dictionaries(dms, ncut);
	nhe = ans.first;
	he2idx = ans.second.second;
	idx2he = ans.second.first;
	N = int(sqrt((int)hamiltonian.size()));
	
	// total density matrices in hierarchy
	int numr = kcut + ncut;
	long long int fact = 1;
	for(int i = 1; i <= numr; i++){
		fact *= i;
	} 
	for(int i = 1; i <= ncut; i++){
		fact /= i;
	}
	for(int i = 1; i <= kcut; i++){

		fact /= i;
	}
	total_nhe = fact;
	grad_shape = make_pair(N*N, N*N);
	L = hamiltonian;
	for(int i = 0; i < (int)coupling.size(); i++){
		spreQ.push_back(spre(coupling[i]));
		spreQdag.push_back(spre(dagger(coupling[i])));
	}
	for(int i = 0; i < (int)coupling.size(); i++){
		spostQdag.push_back(spost(dagger(coupling[i])));
		spostQ.push_back(spost(coupling[i]));
	}
}


// get gradient term for hierarchy ADM at level n for bosonic case
void HEOM::fermion_grad_n(vector<int> he_n){
	complex<double> gradient_sum = 0;
	// cout << he_n.size() << endl;
	// cout << "-------";
	for(int i = 0; i < (int)vk.size(); i++){
			gradient_sum += (complex<double>(he_n[i], 0)*vk[i]);
	}
	gradient_sum = complex<double>(-1, 0)*gradient_sum;
	// cout << "grad_sum: " << gradient_sum << endl;
	vector<vector<complex<double> > > Lt = L;
	for(int i = 0; i < grad_shape.first; i++){
		Lt[i][i] += gradient_sum;
	}
	int nidx = he2idx[he_n];
	int block = grad_shape.first;
	int pos = nidx*block;
	// for(int i = 0; i < Lt.size(); i++){
	// 	for(int j = 0; j < Lt[0].size(); j++){
	// 		cout << Lt[i][j] << " ";
	// 	}
	// 	cout << endl;
	// }
	for(int i = 0; i < grad_shape.first; i++){
		for(int j = 0; j < grad_shape.first; j++){
			if(norm(Lt[i][j]) != 0){
				L_helems[make_pair(pos+i, pos+j)] = Lt[i][j];
			}
		}
	}
}


// get previous gradient for fermionic case
void HEOM::fermion_grad_prev(vector<int> he_n, int k, vector<int> prev_he, int idx){
	// int nk = he_n[k];
	float norm_prev;
	vector<vector<complex<double> > > op1(grad_shape.first, vector<complex<double> >(grad_shape.first));
	norm_prev = 1;
	int sign1 = 0;
	int n_excite = 2;
	for(int i = 0; i < (int)he_n.size(); i++)
		if(he_n[i] == 1)
			n_excite++;
	sign1 = pow(-1, n_excite-1);
	int upto = offsets[k]+idx;
	int sign2 = 1;
	for(int i = 0 ; i < upto; i++)
		if(prev_he[i])
			sign2 *= -1;
	complex<double> pref = complex<double>(sign2, 0)*complex<double>(0, -1)*complex<double>(norm_prev, 0);
	for(int i = 0; i < grad_shape.first; i++){
		for(int j = 0; j < grad_shape.first; j++){
			if(k%2 == 1){
				op1[i][j] = pref*((ck[offsets[k] + idx]*spreQ[k][i][j]) - (complex<double>(sign1, 0)*conj(ck[offsets[k-1] + idx])*spostQ[k][i][j]));
			}
			else{
				op1[i][j] = pref*((ck[offsets[k] + idx]*spreQ[k][i][j]) - (complex<double>(sign1, 0)*conj(ck[offsets[k+1] + idx])*spostQ[k][i][j]));
			}
		}
	}
	// for(int i = 0; i < op1.size(); i++){
	// 	for(int j = 0; j < op1[0].size(); j++){
	// 		cout << op1[i][j] << " ";
	// 	}
	// 	cout << endl;
	// }
	// cout << endl;
	int rowidx = he2idx[he_n];
	int colidx = he2idx[prev_he];
	int block = grad_shape.first;
	int rowpos = rowidx*block;
	int colpos = colidx*block;
	for(int i = 0; i < grad_shape.first; i++){
		for(int j = 0; j < grad_shape.first; j++){
			if(norm(op1[i][j]) != 0){
				L_helems[make_pair(rowpos+i, colpos+j)] = op1[i][j];
			}
		}
	}
}

// get next gradient for fermionic case
void HEOM::fermion_grad_next(vector<int> he_n, int k, vector<int> next_he, int idx){
	vector<vector<complex<double> > > op2(grad_shape.first, vector<complex<double> >(grad_shape.first));
	float norm_next;
	norm_next = 1;
	int sign1 = 0;
	int n_excite = 2;
	for(int i = 0; i < (int)he_n.size(); i++)
		if(he_n[i] == 1)
			n_excite++;
	sign1 = pow(-1, n_excite-1);
	int upto = offsets[k]+idx;
	int sign2 = 1;
	for(int i = 0 ; i < upto; i++)
		if(next_he[i])
			sign2 *= -1;
	complex<double> pref = complex<double>(sign2, 0)*complex<double>(0, -1)*complex<double>(norm_next, 0); 
	for(int i = 0; i < grad_shape.first; i++){
		for(int j = 0; j < grad_shape.first; j++){
			op2[i][j] = pref*(spreQdag[k][i][j] + (complex<double>(sign1, 0)*spostQdag[k][i][j]));		
		}
	}
	// cout << "lol " << sign1 << endl;
	// for(int i = 0; i < op2.size(); i++){
	// 	for(int j = 0; j < op2[0].size(); j++){
	// 		cout << op2[i][j] << " ";
	// 	}
	// 	cout << endl;
	// }
	// cout << endl;
	// cout << op2.size() <<" " << op2[0].size()<< endl;
	int rowidx = he2idx[he_n];
	int colidx = he2idx[next_he];
	int block = grad_shape.first;
	int rowpos = rowidx*block;
	int colpos = colidx*block;
	for(int i = 0; i < grad_shape.first; i++){
		for(int j = 0; j < grad_shape.first; j++){
			if(norm(op2[i][j]) != 0){
				L_helems[make_pair(rowpos+i, colpos+j)] = op2[i][j];
			}
		}
	}
}

// make the rhs for fermionic case
void HEOM::fermion_rhs(){
	// cout << "total_nhe: "<< total_nhe << endl;
	while(nhe < total_nhe){
		vector<int> heidxlist;
		traverse(idx2he, it){
			heidxlist.push_back(it->first);
		}
		populate(heidxlist);
	}
	vector<int> keylist;
	traverse(idx2he, it){
		keylist.push_back(it->first);
	}
	for(int i = 0; i < (int)keylist.size(); i++){
		vector<int> he_n = idx2he[keylist[i]];
		// for(int aa = 0; aa < he_n.size(); aa++)
		// 	cout << he_n[aa] << " ";
		// cout << endl;
		fermion_grad_n(he_n);
		for(int k = 0; k < kcut; k++){
			int start = offsets[k];
			int end = offsets[k+1];
			// cout << "endval" << end << endl;
			int num_elems = end-start;
			// cout << "numelems: " << num_elems << endl;
			for(int m = 0; m < num_elems; m++){
				vector<int> next_he = nexthe(he_n, offsets[k]+m, ncut);
				vector<int> prev_he = prevhe(he_n, offsets[k]+m, ncut);
				// for(int aa = 0; aa < next_he.size(); aa++)
				// 	cout << next_he[aa] << " ";
				// cout << endl;
				// for(int aa = 0; aa < prev_he.size(); aa++)
				// 	cout << prev_he[aa] << " ";
				// cout << endl;
				if(((int)next_he.size() > 0) && (he2idx.find(next_he) != he2idx.end())){
					// cout << "gn: " << k << " " << m << endl;
					fermion_grad_next(he_n, k , next_he, m);
				}
				if(((int)prev_he.size() > 0) && (he2idx.find(prev_he) != he2idx.end())){
					// cout << "gp: " << k << " " << m << endl;
					fermion_grad_prev(he_n, k , prev_he, m);
				}
			}
		}
	}
	// cout << "Dict size is : " << (int)L_helems.size() << endl;
	dict_size = L_helems.size();
	traverse(L_helems, it){
		xs.push_back(it->first.first);
		ys.push_back(it->first.second);
		data.push_back(it->second);
	}
	// cout << "checks :" << endl;
	// cout << (int)xs.size() << " " << (int)ys.size() << " " << (int)data.size() << endl;
}	

// returns complex conjugate transpose
vector<vector<complex<double> > > dagger(vector<vector<complex<double> > > mat){
	vector<vector<complex<double> > > ans;
	int sz1 = mat.size();
	int sz2 = mat[0].size();
	for(int i = 0; i < sz1; i++){
		vector<complex<double> > temp;
		for(int j = 0; j < sz2; j++){
			temp.push_back(0);
		}
		ans.push_back(temp);
	}
	for(int i = 0; i < sz1; i++){
		for(int j = 0; j < sz2; j++){
			ans[j][i] = conj(mat[i][j]);
		}
	}
	return ans;
}

// returns transpose
vector<vector<complex<double> > > transpose(vector<vector<complex<double> > > mat){
	vector<vector<complex<double> > > ans;
	int sz1 = mat.size();
	int sz2 = mat[0].size();
	for(int i = 0; i < sz1; i++){
		vector<complex<double> > temp;
		for(int j = 0; j < sz2; j++){
			temp.push_back(0);
		}
		ans.push_back(temp);
	}
	for(int i = 0; i < sz1; i++){
		for(int j = 0; j < sz2; j++){
			ans[j][i] = mat[i][j];
		}
	}
	return ans;
}

// to avoid compiler errors
int main(){
    return 0;
}

// given function for Runge Kutta
// multiply with matrix of form map<pair<int,int> , complex<double> > L_helems
vector<complex<double> > comp_function(float t, vector<complex<double> > rho, const map<pair<int,int> , complex<double> >& sparse_matrix){
	vector<complex<double> > opt_st(rho.size(), complex<double>(0,0));
	traverse(sparse_matrix, it){
		if(norm(rho[it->first.second]) != 0)
			opt_st[it->first.first] += ((it->second)*(rho[it->first.second]));
	}
	// outputs.push_back(opt_st);
	return opt_st;
}

// Runge Kutta
vector<vector<complex<double> > > RungeKutta(float x0, vector<complex<double> > y0, float x, float h, map<pair<int,int> , complex<double> > sparse_matrix){
    // number of iterations
    int n = (int)((x - x0) / h);  
    vector<complex<double> > k1, k2, k3, k4; 
    vector<complex<double> > y = y0; 
    int sz = y.size();

    vector<vector<complex<double> > > all_states;

    // find next value using Runge Kutta
    for (int i = 0; i < n; i++){ 

    	// first value
        k1 = comp_function(x0, y, sparse_matrix);
        for(int j = 0; j < sz; j++){
        	k1[j] *= h;
        }

        // second value
        vector<complex<double> > y1(sz, complex<double>(0,0));
        for(int j = 0; j < sz; j++){
        	y1[j] = y[j] + k1[j]*0.5;
        }
        k2 = comp_function(x0 + 0.5*h, y1, sparse_matrix);
        for(int j = 0; j < sz; j++){
        	k2[j] *= h;
        }

        // third value
        vector<complex<double> > y2(sz, complex<double>(0,0));
        for(int j = 0; j < sz; j++){
        	y2[j] = y[j] + k2[j]*0.5;
        }
        k3 = comp_function(x0 + 0.5*h, y2, sparse_matrix);
		for(int j = 0; j < sz; j++){
        	k3[j] *= h;
        }

        // final value 
        vector<complex<double> > y3(sz, complex<double>(0,0));
        for(int j = 0; j < sz; j++){
        	y3[j] = y[j] + k3[j];
        }
        k4 = comp_function(x0 + h, y3, sparse_matrix);
		for(int j = 0; j < sz; j++){
        	k4[j] *= h;
        }
        
        // update y
        for(int j = 0; j < sz; j++){
        	y[j] = y[j] + (complex<double>(1/6, 0))*(k1[j] + complex<double>(2, 0)*k2[j] + complex<double>(2, 0)*k3[j] + k4[j]);
        }  
        
        // update x
        x0 = x0 + h; 

        all_states.push_back(y);
    }

    return all_states; 
} 
