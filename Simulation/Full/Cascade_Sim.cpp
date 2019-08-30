//============================================================================
// Name        : Cascade.cpp
// Author      :
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>
#include <random>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <string>
#include <tuple>
#include <ctime>
#include <unordered_map>
#include <iomanip>
#include "mpi.h"
#include "Data_cascade_Sim.h"

using namespace std;

int n_proc = 256;

int network_size = 1080977;// number of firms
int n_item = 9;// firm information items
const int start_date = 371;
const int end_date = 960;
const int n_dim = 73;//number of parameters. 1(const) + 3 firm characteristics (age, sales, hyoten) + 2(macro) + 2(contagion demand) + 7(area FE, base=Kanto) + 58(industry FE, base=JSIC 5)
const int n_sim = 50;
const int T_expire = 246;

const int contag_dem = 9;
const int contag_sup = 10;

void Simulation(const unordered_map<int, unordered_map<int, double> >& network, const unordered_map<int, unordered_map<int, double> >& network_b,
		unordered_map<int, unordered_map<int, double> >& kigyo_joho, unordered_map<int, unordered_map<int, double> >& macrovar,
		vector<double> parameter, int start_date, int end_date,
		unsigned int seed_value,
		int my_rank, int p);
double z_contag_dem_f(const unordered_map<int, unordered_map<int, double> >& network_b, const unordered_map<int, unordered_map<int, double> >& kigyo_joho, const int& f_date_var, const int& t_date, const int& target_i);
double z_contag_sup_f(const unordered_map<int, unordered_map<int, double> >& network, const unordered_map<int, unordered_map<int, double> >& kigyo_joho, const int& f_date_var, const int& t_date, const int& target_i);


int main(int argc, char** argv) {

	//	Data
	unordered_map<int, unordered_map<int, double> > network_1;
	unordered_map<int, unordered_map<int, double> > network_1_b;
	unordered_map<int, unordered_map<int, double> > kigyo_joho_1;
	unordered_map<int, unordered_map<int, double> > macrovar_1;

	string filename = "wei_TSR_sk_sup.txt";
	data(filename, network_1);
	filename = "wei_TSR_sk_dem.txt";
	data_b(filename, network_1_b);
	filename = "TSR_main_full.txt";
	data_KJ(filename, kigyo_joho_1);
	filename = "macrovar.txt";
	data_macro(filename, macrovar_1);

  double t_start, t_finish;
	int my_rank;
	int p;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &p);

	if(my_rank == 0) cout << "n_proc is " << n_proc << endl;
	if(my_rank == 0) cout << "number of paramters is " << n_dim << endl;
	if(my_rank == 0) cout << "number of items is " << n_item << endl;
	// if(my_rank == 0) cout << "end_date is " << end_date << endl;
	// if(my_rank == 0) cout << "check kigyo joho " << kigyo_joho_1.at(111).at(3) << endl;


	int f_date_var = 7;
	int data_used_var = 8;
	for(int i = 1; i <= network_size; ++i){
		kigyo_joho_1.at(i)[contag_dem] = z_contag_dem_f(network_1_b, kigyo_joho_1, f_date_var, start_date, i);
		kigyo_joho_1.at(i)[contag_sup] = z_contag_sup_f(network_1, kigyo_joho_1, f_date_var, start_date, i);
	}

	vector<double> parameter{
		1.4155127027027e-05,-0.0397331702702703,0.566739554054054,-0.845365337837838,-0.00863835554054054,0.786895378378378,-0.02954065,-0.195711121621622,-0.168230256756757,-0.133644905405405,-0.456200567567568,-0.199015540540541,-0.42354072972973,-0.193785651351351,-0.475547256756757,-0.486726756756757,0.333032310810811,-0.463539783783784,0.660811148648649,0.33005527027027,0.276199040540541,-0.244373945945946,0.484533959459459,-0.283066054054054,-0.00730950740540541,-0.205592917567568,0.607573337837838,-0.0100572644189189,0.0664408971486486,-0.160602247297297,-0.290761972972973,-0.451514121621622,0.034159502972973,0.270929202702703,0.433487743243243,0.0581936689189189,-0.00819277374324324,-0.232512391891892,0.156967378378378,0.243331012162162,0.0855140486486486,0.361596837837838,0.306581756756757,0.00903106993243243,-0.0355704248648649,-0.373463594594595,-0.867516972972973,-0.0767887594594595,0.0484813818918919,0.744981864864865,0.138305945945946,-0.160197135135135,0.0405034405405405,0.219768675675676,0.0650369148648649,0.341761797297297,-0.252170216216216,-0.464581891891892,-0.208392094594595,0.0723478768918919,-0.888732486486487,-1.10920513513514,-0.82146227027027,-0.3698725,-0.0709588021621622,-0.418150432432432,0.0195209993243243,-0.0130057386486486,-0.084594249054054,-0.230424432432432,-0.0826720148648649,1.09119378378378,0.873740310810811
	};

	mt19937 engine_original(123);
	unsigned int seed_v[n_sim];
	for(int i = 0; i < n_sim; ++i) seed_v[i] = engine_original();

	t_start = MPI_Wtime();
	for(int sim_i = 0; sim_i < n_sim; ++sim_i){
		// if(my_rank == 0) cout << "Simulation number " << sim_i << "," << seed_v[sim_i] << endl;
		// initializaion
		for(int i = 1; i <= network_size; ++i){
			if(kigyo_joho_1.at(i).at(data_used_var) == 1.0){
				if(kigyo_joho_1.at(i).at(f_date_var) >= start_date) kigyo_joho_1.at(i).at(f_date_var) = 9999;
			}
		}
		Simulation(network_1, network_1_b, kigyo_joho_1, macrovar_1, parameter, start_date, end_date, seed_v[sim_i], my_rank, p);
	}
	t_finish = MPI_Wtime();
	if(my_rank == 0) cout << "Simulation time is " << t_finish - t_start << endl;
  MPI_Finalize();
	return 0;
}

void Simulation(const unordered_map<int, unordered_map<int, double> >& network, const unordered_map<int, unordered_map<int, double> >& network_b,
		unordered_map<int, unordered_map<int, double> >& kigyo_joho, unordered_map<int, unordered_map<int, double> >& macrovar,
		vector<double> parameter, int start_date, int end_date,
		unsigned int seed_value,
		int my_rank, int p){

	mt19937 engine_rank(seed_value);
	unsigned int seed_array[n_proc];
	for(int i = 0; i < n_proc; ++i) seed_array[i] = engine_rank();
	unsigned int seed_rank = seed_array[my_rank];
	mt19937 engine(seed_rank);

	int log_age = 2;
	int log_sales = 3;
	int hyoten = 4;
	int area_id = 5;
	int ind_id = 6;
	int f_date = 7;
	int data_used = 8;

	//  3 firm characteristics (age, sales, hyoten) + 2(macro) + 7(area FE, base=Kanto) + 58(industry FE, base=JSIC 5) + 2(contagion demand and supply)
	double lambda_0 = parameter[0];//constant
	double beta_f_1 = parameter[1];//firm age
	double beta_f_2 = parameter[2];//firm sales
	double beta_f_3 = parameter[3];//firm hyoten
	double beta_m_1 = parameter[4];
	double beta_m_2 = parameter[5];
	// Fixed effect
	vector<double> area_FE(8, 0);
	for(int i = 1; i <= 7; ++i) area_FE[i] = parameter[(5 + i)];
	vector<double> ind_FE(59, 0);
	for(int i = 1; i <= 58; ++i) ind_FE[i] = parameter[(12 + i)];
	double contagion_dem = parameter[71];//contagion from demand side
	double contagion_sup = parameter[72];//contagion from supply side

	unordered_map<int, int> bank_firm_date_sim;//to store the results

	for(int date = start_date; date <= end_date; ++date){
		vector<int> bankruptcy;
		vector<int> bankruptcy_each;

		int ib = network_size/n_proc;
		int istat, iend;

		istat = my_rank*ib + 1;
		if(my_rank == (n_proc - 1) ){
			iend = network_size + 1;
		}else{
			iend = (my_rank + 1)*ib + 1;
		}

		// if(my_rank == 0) cout << "check_1 " << endl;

		for(int i = istat; i < iend; ++i){

			// if(my_rank == 0 && i < 10) cout << "check_2 " << endl;

			if( !(kigyo_joho.at(i).at(f_date) < date) ){// surviving firms
				if(kigyo_joho.at(i).at(data_used) == 1.0){

					double prob = lambda_0*exp(beta_f_1*kigyo_joho.at(i).at(log_age) + beta_f_2*kigyo_joho.at(i).at(log_sales) + beta_f_3*kigyo_joho.at(i).at(hyoten) +
						beta_m_1*macrovar.at(date).at(1) + beta_m_2*macrovar.at(date).at(2) + contagion_dem*kigyo_joho.at(i).at(contag_dem) + contagion_sup*kigyo_joho.at(i).at(contag_sup) + area_FE.at(kigyo_joho.at(i).at(area_id)) + ind_FE.at(kigyo_joho.at(i).at(ind_id) ) );

					// if(my_rank == 0 && i < 10) cout << "check_3 " << endl;

					discrete_distribution<int> dist {1 - prob, prob};
					int number = dist(engine);
					if(number == 1) bankruptcy_each.push_back(i);
				}
			}
		}

		// if(my_rank == 0) cout << "check_4 " << endl;

		int each_size = bankruptcy_each.size();
		int sbuf_bank[100];//the initialization int sbuf_bank[100] = {}; works here
		for(int i = 0; i < 100; ++i) sbuf_bank[i] = 0;
		for(int i = 0; i < each_size; ++i) sbuf_bank[i] = bankruptcy_each[i];
		int rbuf_bank[n_proc*100];//why rbuf_bank[n_proc*100] = {}; does not work on K-computer
		for(int i = 0; i < n_proc*100; ++i) rbuf_bank[i] = 0;
		MPI_Allgather(sbuf_bank, 100, MPI_INT, rbuf_bank, 100, MPI_INT, MPI_COMM_WORLD);

		// if(my_rank == 0) cout << "check_5 " << endl;

		for(int i = 0; i < n_proc*100; ++i){
			if(rbuf_bank[i] != 0) bankruptcy.push_back(rbuf_bank[i]);
		}

		// if(my_rank == 0) cout << "check_6 " << endl;

		for(auto itr_0 = bankruptcy.begin(); itr_0 != bankruptcy.end(); ++itr_0){
			kigyo_joho.at(*itr_0).at(f_date) = (double)date;
			bank_firm_date_sim[*itr_0] = date;
		}

		// if(my_rank == 0) cout << "check_7 " << endl;

		for(auto itr_1 = bankruptcy.begin(); itr_1 != bankruptcy.end(); ++itr_1){
			// for demand effects
			for(auto itr_2 = network.at(*itr_1).begin(); itr_2 != network.at(*itr_1).end(); ++itr_2){
				int sup_i = itr_2->first;
				if(kigyo_joho.at(sup_i).at(data_used) == 1.0) kigyo_joho.at(sup_i).at(contag_dem) = z_contag_dem_f(network_b, kigyo_joho, f_date, date, sup_i);
			}
			// for supply effects
			for(auto itr_3 = network_b.at(*itr_1).begin(); itr_3 != network_b.at(*itr_1).end(); ++itr_3){
				int cus_i = itr_3->first;
				if(kigyo_joho.at(cus_i).at(data_used) == 1.0) kigyo_joho.at(cus_i).at(contag_sup) = z_contag_sup_f(network, kigyo_joho, f_date, date, cus_i);
			}
		}

		// if(my_rank == 0) cout << "check_8 " << endl;
	}

	if(my_rank == 0){
		// cout << "bankruptcy size " << bank_firm_date_sim.size() << endl;
		for(auto itr_0 = bank_firm_date_sim.begin(); itr_0 != bank_firm_date_sim.end(); ++itr_0){
			int b_firm = itr_0->first;
			cout << b_firm << ", ";
		}
		cout << endl;
	}
}

double z_contag_dem_f(const unordered_map<int, unordered_map<int, double> >& network_b, const unordered_map<int, unordered_map<int, double> >& kigyo_joho, const int& f_date_var, const int& t_date, const int& target_i){
	double z_contag_dem = 0.0;
	for(auto itr = network_b.at(target_i).begin(); itr != network_b.at(target_i).end(); ++itr){
		int cus_i = itr->first;
		if(kigyo_joho.at(cus_i).at(f_date_var) < t_date && kigyo_joho.at(cus_i).at(f_date_var) > (t_date - T_expire) ) z_contag_dem += itr->second;
	}
	return z_contag_dem;
}

double z_contag_sup_f(const unordered_map<int, unordered_map<int, double> >& network, const unordered_map<int, unordered_map<int, double> >& kigyo_joho, const int& f_date_var, const int& t_date, const int& target_i){
	double z_contag_sup = 0.0;
	for(auto itr = network.at(target_i).begin(); itr != network.at(target_i).end(); ++itr){
		int sup_i = itr->first;
		if(kigyo_joho.at(sup_i).at(f_date_var) < t_date && kigyo_joho.at(sup_i).at(f_date_var) > (t_date - T_expire) ) z_contag_sup += itr->second;
	}
	return z_contag_sup;
}
