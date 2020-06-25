#include "utilities.h"
#include <chrono>
// #include <bits/stdc++.h>

#include <vector>
#include <map>
#include <iterator>
#include <cmath>
#include <complex>
#include <iostream>
#include <functional>
#include <numeric>
// my favorite macro of all time, in case anyone looks at this comment
// makes STL container traversal a breeze
// the code would look at least 10x uglier without this
#define traverse(container, it) \
for(typeof(container.begin()) it = container.begin(); it != container.end(); it++)

using namespace std;

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

// enumerates all states for n sites with possible range given in vector
vector<vector<int> > state_number_enumerate(vector<int> dims, int excitations){
	vector<vector<int> > arr;
	int n = (int)dims.size();
	for(int i = 0; i < n; i++){
		vector<int> temp(dims[i]);
		std::iota(temp.begin(), temp.end(), 0);
		arr.push_back(temp);
	}
	// computes combinations
	vector<int> temp(n);
	vector<vector<int> > states;
    vector<int> indices(n, 0);
    int next;
    while(1){ 
  		temp.clear();
        for (int i = 0; i < n; i++) 
            temp.push_back(arr[i][indices[i]]);  
        long long int total = accumulate(temp.begin(), temp.end(), 0);
        if(total <= excitations)
		    states.push_back(temp);
        next = n - 1; 
        while (next >= 0 &&  
              (indices[next] + 1 >= (int)arr[next].size())) 
            next--; 
        if (next < 0) 
            break; 
        indices[next]++; 
        for (int i = next + 1; i < n; i++) 
            indices[i] = 0; 
    }
    // cout << states.size() << endl; 
    return states;
}

// returns HEOM state dictionaries
pair<int, pair< map<int, vector<int> > , map< vector<int>, int> > > _heom_state_dictionaries(vector<int> dims, int excitations){

	int nstates = 0;
	map< int, vector<int> > idx2state;
	map< vector<int>, int> state2idx;
	vector<vector<int> > states = state_number_enumerate(dims, excitations);
	for(int i = 0; i < (int)states.size(); i++){
		vector<int> temp = states[i];
		state2idx[temp] = nstates;
		idx2state[nstates] = temp;
		nstates++;
	}
	pair<int, pair< map<int, vector<int> > , map< vector<int>, int> > > ans;
	ans.first = nstates;
	ans.second.first = idx2state;
	ans.second.second = state2idx;
	// cout << nstates << " " << idx2state.size() << " " << state2idx.size() << endl;
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
	// cout << num_ops << endl;
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
	// for(int i = 0; i < C.size();i++){
	// 	for(int j = 0; j < C[0].size();j++){
	// 		for(int k = 0; k < C[0][0].size(); k++)
	// 			cout << C[i][j][k] << " ";
	// 		cout << endl;
	// 	}
	// 	cout << endl;
	// }
	// cout << nc << " " << nr << " " << ni << " " << k << endl;
	// cout << "num coup ops: " << C.size() << endl;
	vector<complex<double> > ck(ck1, ck1+szck);
	// for(int i = 0; i < ck.size(); i++)
	// 	cout << ck[i] << " " ;
	// cout << endl; 
	vector<complex<double> > vk(vk1, vk1+szvk);
	// for(int i = 0; i < vk.size(); i++)
	// 	cout << vk[i] << " " ;
	// cout << endl;
	HEOM tester = HEOM();
	// cout << "starting"<< endl;
	tester.set_hamiltonian(H);
	// cout << "set hamiltonian"<< endl;
	tester.set_coupling(C);
	// cout << "set coupling"<< endl;
	tester.set_ck(ck);
	// cout << "set ck" << endl;
	tester.set_vk(vk);
	// cout << "set vk" << endl;
	tester.set_ncut(nc);
	// cout << "set ncut" << endl;
	tester.set_nr(nr);
	tester.set_ni(ni);
	if(k == 1)
		tester.boson_initialize();
	else
		tester.boson_initializeWithL();
	// cout << "done with init" << endl;
	tester.boson_rhs();
	// xs = (int*)(malloc(sizeof(int)* tester.dict_size));
	// ys = (int*)(malloc(sizeof(int)* tester.dict_size));
	// data = (complex<double>*)(malloc(sizeof(complex<double>)* tester.dict_size));
	// memcpy(xs, &tester.xs[0], sizeof(int)* tester.dict_size);
	// memcpy(ys, &tester.ys[0], sizeof(int)* tester.dict_size);
	// memcpy(data, &tester.data[0], sizeof(complex<double>)* tester.dict_size);
	// return tester.dict_size;
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
	// cout << "Kcut : " << kcut << endl; 

	// initialize dicts
	vector<int> dms(kcut, ncut+1);
	pair<int, pair< map<int, vector<int> > , map< vector<int>, int> > > ans = _heom_state_dictionaries(dms, ncut);
	// cout << "made dict" << endl;
	nhe = ans.first;
	he2idx = ans.second.second;
	idx2he = ans.second.first;
	N = (int)hamiltonian.size();
	// cout << "N : "<< N << endl;
	
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
	// cout << "fact : " << fact << endl;
	total_nhe = fact;
	// hshape = make_pair(total_nhe, N*N);
	grad_shape = make_pair(N*N, N*N);
	// cout << "begin assigning L" << endl;
	// TODO 
	// L_helems
	// spreQ = spre(coupling);
	for(int i = 0; i < (int)coupling.size(); i++){
		spreQ.push_back(spre(coupling[i]));
	}
	// spostQ = spost(coupling);
	for(int i = 0; i < (int)coupling.size(); i++){
		spostQ.push_back(spost(coupling[i]));
		// cout << dagger(coupling[i]).size() << " ";
	}
	// for(int i = 0; i < spostQ.size();i++){
	// 	for(int j = 0; j < spostQ[0].size();j++){
	// 		for(int k = 0; k < spostQ[0][0].size(); k++)
	// 			cout << spostQ[i][j][k] << " ";
	// 		cout << endl;
	// 	}
	// 	cout << endl;
	// }
	// cout << endl;
	vector<vector<complex<double> > > spreH = spre(hamiltonian);
	vector<vector<complex<double> > > spostH = spost(hamiltonian);
	// cout << "done spre and spost ing" << endl;
	vector<vector<complex<double> > > tempmat((int)spreH.size(), vector<complex<double> >((int)spreH[0].size(), complex<double>(0,0)));
	// sets value of L
	// cout << spreQ.size() << endl;
	for(int i = 0; i < (int)spreH.size(); i++){
		for(int j = 0; j < (int)spreH[0].size(); j++){
			tempmat[i][j] = complex<double>(0, -1)*(spreH[i][j] - spostH[i][j]);
			// cout << spreQ[i][j] << " " ;
		} 
		// cout << endl;
	}
	// for(int i = 0; i < spreH.size(); i++){
	// 	for(int j = 0; j < spreH[0].size(); j++){
	// 		// tempmat[i][j] = complex<double>(0, -1)*(spreH[i][j] - spostH[i][j]);
	// 		cout << spostQ[i][j] << " " ;
	// 	} 
	// 	cout << endl;
	// }
	// cout << "came out of loop" << endl;
	L = tempmat;
	// cout << "assigned L" << endl;
	// L_helems
	// is already defined
}


// initialize private variables when given Liouvillian for bosonic case
void HEOM::boson_initializeWithL(){
	kcut = nr + ni + ((int)ck.size() - (nr + ni))/2;

	// initialize dicts
	vector<int> dms(kcut, ncut+1);
	pair<int, pair< map<int, vector<int> > , map< vector<int>, int> > > ans = _heom_state_dictionaries(dms, ncut);
	// cout << "made dict" << endl;
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
	// cout << "fact : " << fact << endl;
	total_nhe = fact;
	// hshape = make_pair(total_nhe, N*N);
	grad_shape = make_pair(N*N, N*N);
	L = hamiltonian;
	// spreQ = spre(coupling);
	for(int i = 0; i < (int)coupling.size(); i++){
		spreQ.push_back(spre(coupling[i]));
	}
	// spostQ = spost(coupling);
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
	// cout << "IN SPRE : " << ans.size() << endl;
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
	// cout << "for this call " << endl;
	for(int i = 0; i < (int)vk.size(); i++){
		if(i < nr+ni){
			gradient_sum += (complex<double>(he_n[i], 0)*vk[i]);
			// cout << "one : " << gradient_sum << " " << he_n[i] << " " << vk[i] << " " << i << endl;
		}
		else{
			if(skip){
				skip = 0;  //this blows my mind.
				continue;
			}
			else{
				// int idx = nr + ni + (i-nr-ni)/2;
				int tot_fixed = nr + ni;
				int extra = (i+1-tot_fixed);
				int idx = tot_fixed + (extra/2) + (extra%2) -1;
				gradient_sum += (complex<double>(he_n[idx], 0)*vk[i]);
				// cout << "two : " << gradient_sum << " " << he_n[idx] << " " << vk[i] << " " << idx << endl;
				skip = 1;
			}
		}
	}
	gradient_sum = complex<double>(-1, 0)*gradient_sum;
	// cout << gradient_sum << endl;
	vector<vector<complex<double> > > Lt = L;
	for(int i = 0; i < grad_shape.first; i++){
		Lt[i][i] += gradient_sum;
	}
	// for(int i = 0; i < grad_shape.first; i++){
	// 	for(int j = 0; j < grad_shape.first; j++){
	// 		cout << Lt[i][j] << " " ;
	// 	}
	// 	cout << endl;
	// }
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
}

// get previous gradient for bosonic case
void HEOM::boson_grad_prev(vector<int> he_n, int k, vector<int> prev_he){
	int nk = he_n[k];
	float norm_prev;
	vector<vector<complex<double> > > op1(grad_shape.first, vector<complex<double> >(grad_shape.first));
	// if(k == 0){
	// 	norm_prev = sqrt(float(nk)/abs(lam));
	// 	for(int i = 0; i < grad_shape.first; i++){
	// 		for(int j = 0; j < grad_shape.first; j++){
	// 			op1[i][j] = complex<double>(0, -1)*complex<double>(norm_prev, 0)*complex<double>(lam, 0)*spostQ[i][j]*complex<double>(-1, 0);
	// 		}
	// 	}
	// }
	// else if(k == 1){
	// 	norm_prev = sqrt(float(nk)/abs(lam));
	// 	for(int i = 0; i < grad_shape.first; i++){
	// 		for(int j = 0; j < grad_shape.first; j++){
	// 			op1[i][j] = complex<double>(0, -1)*complex<double>(norm_prev, 0)*complex<double>(lam, 0)*spreQ[i][j];
	// 		}
	// 	}
	// }
	// else{
	// 	norm_prev = sqrt(float(nk)/abs(ck[k]));
	// 	for(int i = 0; i < grad_shape.first; i++){
	// 		for(int j = 0; j < grad_shape.first; j++){
	// 			op1[i][j] = complex<double>(0, -1)*complex<double>(norm_prev, 0)*complex<double>(abs(ck[k]))*(spreQ[i][j] - spostQ[i][j]);
	// 		}
	// 	}
	// }
	if(k < nr){
		// norm_prev = sqrt(float(nk)/abs(ck[k]));
		norm_prev = float(nk);
		for(int i = 0; i < grad_shape.first; i++){
			for(int j = 0; j < grad_shape.first; j++){
				op1[i][j] = complex<double>(0, -1)*complex<double>(norm_prev, 0)*ck[k]*(spreQ[k][i][j] - spostQ[k][i][j]);
				// cout << op1[i][j] << " ";
			}
			// cout << endl;
		}
	}
	else if( k >= nr && k < nr+ni){
		// norm_prev = sqrt(float(nk)/abs(ck[k]));
		// cout << "k " << k << endl; 
		norm_prev = float(nk);
		for(int i = 0; i < grad_shape.first; i++){
			for(int j = 0; j < grad_shape.first; j++){
				op1[i][j] = complex<double>(0, -1)*complex<double>(norm_prev, 0)*complex<double>(0, 1)*ck[k]*(spreQ[k][i][j] + spostQ[k][i][j]);
				// cout << op1[i][j] << " ";
			}
			// cout << endl;
		}
	}
	else{
		norm_prev = float(nk);
		// computing actual index
		// cout << "k " << k << endl;
		int k1 = nr + ni + 2*(k-(nr+ni));
		for(int i = 0; i < grad_shape.first; i++){
			for(int j = 0; j < grad_shape.first; j++){
				// op1[i][j] = complex<double>(0, -1)*complex<double>(norm_prev, 0)*(ckR[k] * (spreQ[k][i][j] - spostQ[k][i][j])
				// 								+ i*(ckI[k]) * (spreQ[k][i][j] + spostQ[k][i][j]);
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
	// cout << "entered grad next" << endl;
	// int nk = he_n[k];
	vector<vector<complex<double> > > op2(grad_shape.first, vector<complex<double> >(grad_shape.first));
	float norm_next;
	// if(k < 2)
	// 	norm_next = sqrt(float(lam*(nk+1)));
	// else
	// 	norm_next = sqrt(float(abs(ck[k])*(nk+1)));
	// norm_next = sqrt(float(abs(ck[k])*(nk+1)));
	norm_next = 1;
	for(int i = 0; i < grad_shape.first; i++){
		for(int j = 0; j < grad_shape.first; j++){
			op2[i][j] = complex<double>(0, -1)*complex<double>(norm_next, 0)*(spreQ[k][i][j] - spostQ[k][i][j]);
			// cout << op2[i][j] << " ";
		}
		// cout << endl;
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
	// auto start1 = high_resolution_clock::now(); 
	while(nhe < total_nhe){
		vector<int> heidxlist;
		traverse(idx2he, it){
			heidxlist.push_back(it->first);
		}
		populate(heidxlist);
	}
	// auto stop1 = high_resolution_clock::now();
	// auto duration1 = duration_cast<seconds>(stop1 - start1); 
	// cout << "While loop: ";
	// cout << duration1.count() << endl; 
	// cout << "came out of while loop" << endl;
	vector<int> keylist;
	traverse(idx2he, it){
		keylist.push_back(it->first);
	}
	// cout << "KLS : " << keylist.size() << endl;
	// cout << "added keys" << endl;
	for(int i = 0; i < (int)keylist.size(); i++){
		vector<int> he_n = idx2he[keylist[i]];
		boson_grad_n(he_n);
		vector<int> next_he;
		vector<int> prev_he;
		// cout << "did grad n" << endl;
		for(int k = 0; k < kcut; k++){
			// if(k > nr+ni){
			// 	int tot_fixed = nr + ni;
			// 	int extra = (k+1-tot_fixed);
			// 	int idx = tot_fixed + (extra/2) + (extra%2) -1;
			// 	next_he = nexthe(he_n, idx, ncut);
			// 	prev_he = prevhe(he_n, idx, ncut);
			// }
			// else{
			// 	next_he = nexthe(he_n, k, ncut);
			// 	prev_he = prevhe(he_n, k, ncut);
			// }
			next_he = nexthe(he_n, k, ncut);
			prev_he = prevhe(he_n, k, ncut);
			// cout << "lol " << next_he.size() << " " << kcut << endl; 
			// cout << he2idx[prev_he] << endl;
			// cout << "made 2 vecs" << endl;
			if(((int)next_he.size() > 0) && (he2idx.find(next_he) != he2idx.end())){
				// auto start2 = high_resolution_clock::now(); 
				// cout << "entered condition for grad next" << endl;
				// cout << i << " " << k << endl;
				boson_grad_next(he_n, k , next_he);
				// cout << "did grad next" << endl;
				// auto stop2 = high_resolution_clock::now();
				// auto duration2 = duration_cast<seconds>(stop2 - start2); 
				// cout << "Grad next : " << endl;
				// cout << duration2.count() << endl; 
			}
			if(((int)prev_he.size() > 0) && (he2idx.find(prev_he) != he2idx.end())){
				// cout << i << " " << k << endl;
				// auto start3 = high_resolution_clock::now(); 
				boson_grad_prev(he_n, k , prev_he);
				// auto stop3 = high_resolution_clock::now();
				// auto duration3 = duration_cast<seconds>(stop3 - start3); 
				// cout << "Grad prev : " << endl;
				// cout << duration3.count() << endl;
			}
		}
	}
	cout << "Dict size is : " << (int)L_helems.size() << endl;
	dict_size = L_helems.size();
	traverse(L_helems, it){
		// // rows.push_back(it->first.first);
		// if(it->first.first > curr_row){
		// 	// new row
		// 	int num_times = it->first.first - curr_row;
		// 	// increment current row
		// 	curr_row = it->first.first;
		// 	// add it these many times
		// 	for(int ii = 0; ii < num_times; ii++){
		// 		indptr.push_back(curr_row);
		// 	}
		// }
		// indices.push_back(it->first.second);
		// data.push_back(it->second);
		// count++;
		xs.push_back(it->first.first);
		ys.push_back(it->first.second);
		data.push_back(it->second);
	}
	cout << "checks :" << endl;
	cout << (int)xs.size() << " " << (int)ys.size() << " " << (int)data.size() << endl;
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
	// for(int i = 0; i < (int)H.size(); i++)
	// 	for(int j = 0; j < (int)H[0].size(); j++)
	// 		cout << H[i][j] << " ";
	// 	cout << endl;
	// cout << endl;
	// process C into a list of matrices of doubles
	vector<vector<vector<complex<double> > > > C;
	int num_ops = szc1/szc;
	// cout << num_ops << endl;
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
	// cout << C.size() << endl;
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
	// for(int i = 0; i < (int)offsets.size();i++)
	// 	cout << offsets[i] << " ";
	// cout << endl;
	HEOM tester = HEOM();
	// cout << "starting"<< endl;
	// vector<complex<double> > a;
	// vector<vector<complex<double> > > b;
	// a.push_back(complex<double>(1,-1));
	// a.push_back(complex<double>(2,-2));
	// b.push_back(a);
	// a.clear();
	// a.push_back(complex<double>(3,-3));
	// a.push_back(complex<double>(4,-4));
	// b.push_back(a);
	// vector<vector<complex<double> > > c = tester.dagger(b);
	// for(int i = 0; i < 2; i++){
	// 	for(int j = 0; j < 2; j++){
	// 		cout << c[i][j] << " ";
	// 	}
	// 	cout << endl;
	// }
	tester.set_hamiltonian(H);
	// cout << "set hamiltonian"<< endl;
	tester.set_coupling(C);
	// cout << "set coupling"<< endl;
	tester.set_ck(ck);
	// cout << "set ck" << endl;
	tester.set_vk(vk);
	// cout << "set vk" << endl;
	tester.set_offsets(offsets);
	// cout << "set offsets" << endl;
	tester.set_ncut(nc);
	// cout << "set ncut" << endl;
	if(k == 1)
		tester.fermion_initialize();
	else
		tester.fermion_initializeWithL();
	// cout << "done with init" << endl;
	tester.fermion_rhs();
	// cout << "made rhs" << endl;
// 	// xs = (int*)(malloc(sizeof(int)* tester.dict_size));
// 	// ys = (int*)(malloc(sizeof(int)* tester.dict_size));
// 	// data = (complex<double>*)(malloc(sizeof(complex<double>)* tester.dict_size));
// 	// memcpy(xs, &tester.xs[0], sizeof(int)* tester.dict_size);
// 	// memcpy(ys, &tester.ys[0], sizeof(int)* tester.dict_size);
// 	// memcpy(data, &tester.data[0], sizeof(complex<double>)* tester.dict_size);
// 	// return tester.dict_size;
	return make_pair(tester.L_helems, tester.nhe);
}


// initialize private variables for fermionic case
void HEOM::fermion_initialize(){

	kcut = offsets.size() - 1;
	// cout << "Kcut : " << kcut << endl; 

	// initialize dicts
	vector<int> dms(ck.size(), 2);
	pair<int, pair< map<int, vector<int> > , map< vector<int>, int> > > ans = _heom_state_dictionaries(dms, ncut);
	// cout << "made dict" << endl;
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
	// cout << "fact : " << fact << endl;
	total_nhe = fact;
	// hshape = make_pair(total_nhe, N*N);
	grad_shape = make_pair(N*N, N*N);
	// cout << "begin assigning L" << endl;
	// TODO 
	// L_helems
	// spreQ = spre(coupling);
	for(int i = 0; i < (int)coupling.size(); i++){
		spreQ.push_back(spre(coupling[i]));
		spreQdag.push_back(spre(dagger(coupling[i])));
	}
	// spostQ = spost(coupling);
	for(int i = 0; i < (int)coupling.size(); i++){
		spostQdag.push_back(spost(dagger(coupling[i])));
		// vector<vector<complex<double> > > conjtran = dagger(coupling[i]);
		// for(int j = 0; j < conjtran.size(); j++){
		// 	for(int k = 0; k < conjtran[j].size();k++){
		// 		cout << conjtran[j][k] << " ";
		// 	}
		// 	cout << endl;
		// }
		// cout << endl;
		spostQ.push_back(spost(coupling[i]));
	}
	vector<vector<complex<double> > > spreH = spre(hamiltonian);
	vector<vector<complex<double> > > spostH = spost(hamiltonian);
	// cout << "done spre and spost ing" << endl;
	vector<vector<complex<double> > > tempmat((int)spreH.size(), vector<complex<double> >((int)spreH[0].size(), complex<double>(0,0)));
	// sets value of L
	// cout << spreQ.size() << endl;
	for(int i = 0; i < (int)spreH.size(); i++){
		for(int j = 0; j < (int)spreH[0].size(); j++){
			tempmat[i][j] = complex<double>(0, -1)*(spreH[i][j] - spostH[i][j]);
			// cout << spreQ[i][j] << " " ;
		} 
		// cout << endl;
	}
	// for(int i = 0; i < spostQ.size();i++){
	// 	for(int j = 0; j < spostQ[i].size(); j++){
	// 		for(int k = 0; k < spostQ[i][j].size();k++){
	// 			cout << spostQ[i][j][k] << " ";
	// 		}
	// 		cout << endl;
	// 	}
	// 	cout << endl;
	// }
	// for(int i = 0; i < spreH.size(); i++){
	// 	for(int j = 0; j < spreH[0].size(); j++){
	// 		// tempmat[i][j] = complex<double>(0, -1)*(spreH[i][j] - spostH[i][j]);
	// 		cout << spostQ[i][j] << " " ;
	// 	} 
	// 	cout << endl;
	// }
	// cout << "came out of loop" << endl;
	L = tempmat;
	// cout << "assigned L" << endl;
	// L_helems
	// is already defined
}


// initialize private variables when given Liouvillian for fermionic case
void HEOM::fermion_initializeWithL(){
	kcut = offsets.size() -1;

	// initialize dicts
	vector<int> dms(kcut, 2);
	pair<int, pair< map<int, vector<int> > , map< vector<int>, int> > > ans = _heom_state_dictionaries(dms, ncut);
	// cout << "made dict" << endl;
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
	// cout << "fact : " << fact << endl;
	total_nhe = fact;
	// hshape = make_pair(total_nhe, N*N);
	grad_shape = make_pair(N*N, N*N);
	L = hamiltonian;
	// spreQ = spre(coupling);
	for(int i = 0; i < (int)coupling.size(); i++){
		spreQ.push_back(spre(coupling[i]));
		spreQdag.push_back(spre(dagger(coupling[i])));
	}
	// spostQ = spost(coupling);
	for(int i = 0; i < (int)coupling.size(); i++){
		spostQdag.push_back(spost(dagger(coupling[i])));
		// vector<vector<complex<double> > > conjtran = dagger(coupling[i]);
		spostQ.push_back(spost(coupling[i]));
	}
}


// get gradient term for hierarchy ADM at level n for bosonic case
void HEOM::fermion_grad_n(vector<int> he_n){
	complex<double> gradient_sum = 0;
	// cout << he_n.size() << " " << vk.size();
	for(int i = 0; i < (int)vk.size(); i++){
			gradient_sum += (complex<double>(he_n[i], 0)*vk[i]);
	}
	// cout << gradient_sum << endl;
	gradient_sum = complex<double>(-1, 0)*gradient_sum;
	// cout << gradient_sum << endl;
	vector<vector<complex<double> > > Lt = L;
	for(int i = 0; i < grad_shape.first; i++){
		Lt[i][i] += gradient_sum;
	}
	// for(int i = 0; i < grad_shape.first; i++){
	// 	for(int j = 0; j < grad_shape.first; j++){
	// 		cout << Lt[i][j] << " " ;
	// 	}
	// 	cout << endl;
	// }
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
}


// get previous gradient for fermionic case
void HEOM::fermion_grad_prev(vector<int> he_n, int k, vector<int> prev_he, int idx){
	// int nk = he_n[k];
	float norm_prev;
	vector<vector<complex<double> > > op1(grad_shape.first, vector<complex<double> >(grad_shape.first));

	// norm_prev = sqrt(float(nk)/abs(ck[k]));
	// for(int i = 0; i < he_n.size(); i++){
	// 	cout << he_n[i] << " ";
	// }
	// cout << endl;
	norm_prev = 1;
	int sign1 = 0;
	int n_excite = 2;
	for(int i = 0; i < (int)he_n.size(); i++)
		if(he_n[i] == 1)
			n_excite++;
	// cout << n_excite-1 << endl;
	sign1 = pow(-1, n_excite-1);
	int upto = offsets[k]+idx;
	int sign2 = 1;
	for(int i = 0 ; i < upto; i++)
		if(prev_he[i])
			sign2 *= -1;
	// cout << offsets[k]+idx << endl;
	complex<double> pref = complex<double>(sign2, 0)*complex<double>(0, -1)*complex<double>(norm_prev, 0);
	for(int i = 0; i < grad_shape.first; i++){
		for(int j = 0; j < grad_shape.first; j++){
			if(k%2 == 1){
				op1[i][j] = pref*((ck[offsets[k] + idx]*spreQ[k][i][j]) - (complex<double>(sign1, 0)*conj(ck[offsets[k-1] + idx])*spostQ[k][i][j]));
				// cout << ck[offsets[k] + idx] << " " << spreQ[k][0][1] << " " << complex<double>(sign1, 0) << " " << conj(ck[offsets[k-1] + idx]) << " " << spostQ[k][0][1] << endl; 
				// cout << ck[offsets[k]+idx] << " " << conj(ck[offsets[k-1] + idx]) << endl;
				// cout << offsets[k-1] + idx << endl;
				// cout << k-1 << " " << idx << endl; 
			}
			else{
				op1[i][j] = pref*((ck[offsets[k] + idx]*spreQ[k][i][j]) - (complex<double>(sign1, 0)*conj(ck[offsets[k+1] + idx])*spostQ[k][i][j]));
				// cout << ck[offsets[k] + idx] << " " << spreQ[k][0][1] << " " << complex<double>(sign1, 0) << " " << conj(ck[offsets[k+1] + idx]) << " " << spostQ[k][0][1] << endl;
				// cout << ck[offsets[k]+idx] << " " << conj(ck[offsets[k+1] + idx]) << endl;
				// cout << offsets[k+1] + idx << endl;
				// cout << k+1 << " " << idx << endl; 
			}
			// cout << op1[i][j] << " ";
		}
		// cout << endl;
	}
	// if(k%2 == 1)
	// 	cout << k-1 << " " << idx << endl; 
	// else
	// 	cout << k+1 << " " << idx << endl; 
	// if(k%2)
	// 	cout << conj(ck[offsets[k-1] + idx]) << endl;
	// cout << op1[0][1] << " " << op1[0][2] << endl;
	// cout << sign2 << endl;
	// cout << ck[offsets[k] + idx] << " " << spreQ[k][0][1] << " " << conj(ck[offsets[k-1] + idx])<< " " << spostQ[k][0][1] << endl;
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
	// cout << "entered grad next" << endl;
	// int nk = he_n[k];
	vector<vector<complex<double> > > op2(grad_shape.first, vector<complex<double> >(grad_shape.first));
	float norm_next;
	// if(k < 2)
	// 	norm_next = sqrt(float(lam*(nk+1)));
	// else
	// 	norm_next = sqrt(float(abs(ck[k])*(nk+1)));
	// norm_next = sqrt(float(abs(ck[k])*(nk+1)));
	norm_next = 1;
	int sign1 = 0;
	int n_excite = 2;
	for(int i = 0; i < (int)he_n.size(); i++)
		if(he_n[i] == 1)
			n_excite++;
	// cout << n_excite-1 << endl;
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
			// cout << op2[i][j] << " ";
		}
		// cout << endl;
	}
	// cout << spreQdag[k][0][1] <<" " <<spostQdag[k][0][2] << endl;
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
	// auto start1 = high_resolution_clock::now(); 
	while(nhe < total_nhe){
		vector<int> heidxlist;
		traverse(idx2he, it){
			heidxlist.push_back(it->first);
		}
		populate(heidxlist);
	}
	// auto stop1 = high_resolution_clock::now();
	// auto duration1 = duration_cast<seconds>(stop1 - start1); 
	// cout << "While loop: ";
	// cout << duration1.count() << endl; 
	// cout << "came out of while loop" << endl;
	vector<int> keylist;
	traverse(idx2he, it){
		keylist.push_back(it->first);
	}
	// cout << keylist.size() << endl;
	// cout << "added keys" << endl;
	for(int i = 0; i < (int)keylist.size(); i++){
		vector<int> he_n = idx2he[keylist[i]];
		fermion_grad_n(he_n);
		// cout << he_n.size() << endl;
		// cout << "did grad n" << endl;
		// cout << keylist.size() << endl;
		for(int k = 0; k < kcut; k++){
			int start = offsets[k];
			int end = offsets[k+1];
			int num_elems = end-start;
			// cout << num_elems << endl;
			for(int m = 0; m < num_elems; m++){
				vector<int> next_he = nexthe(he_n, offsets[k]+m, ncut);
				vector<int> prev_he = prevhe(he_n, offsets[k]+m, ncut);
				// cout << he2idx[prev_he] << endl;
				// cout << "made 2 vecs" << endl;
				if(((int)next_he.size() > 0) && (he2idx.find(next_he) != he2idx.end())){
					// auto start2 = high_resolution_clock::now(); 
					// cout << "entered condition for grad next" << endl;
					fermion_grad_next(he_n, k , next_he, m);
					// cout << "did grad next" << endl;
					// auto stop2 = high_resolution_clock::now();
					// auto duration2 = duration_cast<seconds>(stop2 - start2); 
					// cout << "Grad next : " << endl;
					// cout << duration2.count() << endl; 
				}
				if(((int)prev_he.size() > 0) && (he2idx.find(prev_he) != he2idx.end())){
					// auto start3 = high_resolution_clock::now(); 
					// cout << k << " " << m << endl;
					// for(int i = 0; i < he_n.size(); i++){
					// 	cout << he_n[i] << " ";
					// }
					// cout << endl;
					fermion_grad_prev(he_n, k , prev_he, m);
					// auto stop3 = high_resolution_clock::now();
					// auto duration3 = duration_cast<seconds>(stop3 - start3); 
					// cout << "Grad prev : " << endl;
					// cout << duration3.count() << endl;
				}
			}
		}
	}
	cout << "Dict size is : " << (int)L_helems.size() << endl;
	dict_size = L_helems.size();
	traverse(L_helems, it){
		// // rows.push_back(it->first.first);
		// if(it->first.first > curr_row){
		// 	// new row
		// 	int num_times = it->first.first - curr_row;
		// 	// increment current row
		// 	curr_row = it->first.first;
		// 	// add it these many times
		// 	for(int ii = 0; ii < num_times; ii++){
		// 		indptr.push_back(curr_row);
		// 	}
		// }
		// indices.push_back(it->first.second);
		// data.push_back(it->second);
		// count++;
		xs.push_back(it->first.first);
		ys.push_back(it->first.second);
		data.push_back(it->second);
	}
	cout << "checks :" << endl;
	cout << (int)xs.size() << " " << (int)ys.size() << " " << (int)data.size() << endl;
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
	// cout << "returning" << endl;
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
	// cout << "returning" << endl;
	return ans;
}

// to avoid compiler errors
int main(){
    return 0;
}


// // constructor for ODE
// ODE::ODE(){

// }

// // setters for ODE
// void ODE::set_spmat(map<pair<int,int> , complex<double> > &mat1, long long int size1){
// 	size = size1;
// 		// awesome multiplication algorithm for sparse dicts with full vectors
// 	// assumes rhonext has all elements set to zero
// 	sparse_matrix = mat1;
// 	int count = 0;
// 	int curr_row = -1;
// 	traverse(mat1, it){
// 		// rows.push_back(it->first.first);
// 		if(it->first.first > curr_row){
// 			// new row
// 			int num_times = it->first.first - curr_row;
// 			// increment current row
// 			curr_row = it->first.first;
// 			// add it these many times
// 			for(int ii = 0; ii < num_times; ii++){
// 				indptr.push_back(curr_row);
// 			}
// 		}
// 		indices.push_back(it->first.second);
// 		data.push_back(it->second);
// 		count++;
// 	}
// 	// vector<vector<complex<double> > > temp(size, vector<complex<double> >(size));
// 	// outputs = temp;
// }

// void ODE::operator() (const vector<complex<double> > rho, vector<complex<double> > &rhonext, const double){

// 	vector<complex<double> > opt_st(size, complex<double>(0,0));
// 	traverse(sparse_matrix, it){
// 		if(norm(rho[it->first.second]) != 0)
// 			opt_st[it->first.first] += ((it->second)*(rho[it->first.second]));
// 	}
// 	// outputs.push_back(opt_st);
// 	rhonext = opt_st;
// }
