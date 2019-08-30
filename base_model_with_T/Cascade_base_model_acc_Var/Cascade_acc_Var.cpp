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
#include <omp.h>
#include "Data_cascade_acc_Var.h"

using namespace std;

int n_proc = 48;

int network_size = 1080977;// number of firms
int n_item = 15;// firm information items
const int start_date = 371;
const int end_date = 960;
const int n_dim = 79;//number of parameters. 1(const) + 3 firm characteristics (age, sales, hyoten) + 2(macro) + 2(contagion demand) + 7(area FE, base=Kanto) + 58(industry FE, base=JSIC 5) + 6(financial var)
const int T_expire = 246;

double dd_logLike_l_l(const unordered_map<int, unordered_map<int, double> >& network, const unordered_map<int, unordered_map<int, double> >& network_b,
		const unordered_map<int, unordered_map<int, double> >& kigyo_joho, const unordered_map<int, unordered_map<int, double> >& macrovar,
		vector<double> parameter, int start_date, int end_date,
		int my_rank, int p);
double dd_logLike_l_b(const unordered_map<int, unordered_map<int, double> >& network, const unordered_map<int, unordered_map<int, double> >& network_b,
		const unordered_map<int, unordered_map<int, double> >& kigyo_joho, const unordered_map<int, unordered_map<int, double> >& macrovar,
		vector<double> parameter, int start_date, int end_date,
		int b_num,
		int my_rank, int p);
double dd_logLike_b_b(const unordered_map<int, unordered_map<int, double> >& network, const unordered_map<int, unordered_map<int, double> >& network_b,
		const unordered_map<int, unordered_map<int, double> >& kigyo_joho, const unordered_map<int, unordered_map<int, double> >& macrovar,
		vector<double> parameter, int start_date, int end_date,
		int b_num_1, int b_num_2,
		int my_rank, int p);
double z_contag_dem_f(const unordered_map<int, unordered_map<int, double> >& network_b, const unordered_map<int, unordered_map<int, double> >& kigyo_joho, const int& f_date_var, const int& t_date, const int& target_i);
double z_contag_sup_f(const unordered_map<int, unordered_map<int, double> >& network, const unordered_map<int, unordered_map<int, double> >& kigyo_joho, const int& f_date_var, const int& t_date, const int& target_i);
double z_target_f(const unordered_map<int, unordered_map<int, double> >& kigyo_joho, const unordered_map<int, unordered_map<int, double> >& macrovar, const int& t_date, const int& target_i, const int& b_num,
	double z_contag_dem_var, double z_contag_sup_var);

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

	filename = "TSR_main_acc.txt";
	data_KJ(filename, kigyo_joho_1);
	filename = "macrovar.txt";
	data_macro(filename, macrovar_1);

	// vector<double> parameter{};
	vector<double> parameter = {
		5.222457625e-06,-0.03745871375,0.52095115,-1.0039425,-0.02602099625,0.03457374305,0.315719125,-0.0297458275,-0.1338698625,-0.22220805,-0.339948375,-0.2372010375,-0.36456635,0.2397413625,-0.473816525,-0.5018988625,0.029666837625,-1.130654,0.66612095,0.1133089175,-1.01010915,-1.437015375,0.19531635,0.1841049875,-0.3556254625,-4.79640625,0.8416832125,-0.335612125,-0.7965163375,-0.22037245,-0.1869862,-0.423286625,0.1239298625,0.1687435875,0.5700249,-0.10323534375,-0.01900797925,-0.3370038375,0.33973605,1.045645575,0.66771825,0.1968916125,0.593521375,0.163638175,-0.315964775,-0.08222906875,-0.6770815625,-0.361611325,-0.6642215,1.262899,0.1092958125,-0.2714389,0.03964135875,0.214682025,-0.5450025,0.584171825,-0.43936475,0.10690063375,-0.0259847525,0.2165582375,-0.477111775,-1.0290683375,-0.2039264,0.12108314875,0.0626484775,0.3856201125,-0.4703851375,-0.29553325,0.1744700375,0.4730120125,0.08538289,-0.1518026125,-1.023720375,0.1576371625,-0.097843755,-0.1416594625,0.1859176,1.535071375,0.9520319125
	};

	double t_start, t_finish;
	int my_rank, p;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &p);

	t_start = MPI_Wtime();
  vector<vector<double> > dd_matrix(n_dim, vector<double>(n_dim, 0));
  dd_matrix[0][0] = dd_logLike_l_l(network_1, network_1_b, kigyo_joho_1, macrovar_1, parameter, start_date, end_date, my_rank, p);
  for(int i = 1; i < n_dim; ++i) dd_matrix[0][i] = dd_logLike_l_b(network_1, network_1_b, kigyo_joho_1, macrovar_1, parameter, start_date, end_date, i, my_rank, p);
  for(int i = 1; i < n_dim; ++i){
  	for(int j = i; j < n_dim; ++j) dd_matrix[i][j] = dd_logLike_b_b(network_1, network_1_b, kigyo_joho_1, macrovar_1, parameter, start_date, end_date, i, j, my_rank, p);
  }
  for(int i = 1; i < n_dim; ++i) dd_matrix[i][0] = dd_matrix[0][i];
  for(int i = 1; i < n_dim; ++i){
  	for(int j = 1; j < i; ++j) dd_matrix[i][j] = dd_matrix[j][i];
  }

	if(my_rank == 0){
		cout << "dd_matrix " << endl;
	  for(int i = 0; i < n_dim; ++i){
	  	for(int j = 0; j < n_dim; ++j) cout << dd_matrix[i][j] << ", ";
	  	cout << endl;
	  }
	}

	t_finish = MPI_Wtime();
	if(my_rank == 0) cout << "Elapsed time is " << t_finish - t_start << endl;

	MPI_Finalize();

	return 0;
}

double dd_logLike_l_l(const unordered_map<int, unordered_map<int, double> >& network, const unordered_map<int, unordered_map<int, double> >& network_b,
		const unordered_map<int, unordered_map<int, double> >& kigyo_joho, const unordered_map<int, unordered_map<int, double> >& macrovar,
		vector<double> parameter, int start_date, int end_date,
		int my_rank, int p){

	double logLike = 0.0;
	double r_v_logLike;

	int f_date = 13;
	int data_used = 14;

	//  3 firm characteristics (age, sales, hyoten) + 2(macro) + 7(area FE, base=Kanto) + 58(industry FE, base=JSIC 5) + 2(contagion demand and supply) + 6(financial var)
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

	double beta_acc_1 = parameter[71];//EBITA
	double beta_acc_2 = parameter[72];//CHAT
	double beta_acc_3 = parameter[73];//PAA
	double beta_acc_4 = parameter[74];//REA
	double beta_acc_5 = parameter[75];//LCTAT
	double beta_acc_6 = parameter[76];//FAT

	double contagion_dem = parameter[77];//contagion from demand side
	double contagion_sup = parameter[78];//contagion from supply side

	int ib = network_size/n_proc;
	int istat, iend;

	istat = my_rank*ib + 1;
	if(my_rank == (n_proc - 1) ){
		iend = network_size + 1;
	}else{
		iend = (my_rank + 1)*ib + 1;
	}

	#pragma omp parallel for reduction(+:logLike) schedule(dynamic,1)
	for(int i = istat; i < iend; ++i){
		double ind_logLike = 0.0;
		if(kigyo_joho.at(i).at(data_used) == 1.0){

			if(kigyo_joho.at(i).at(f_date) > end_date){
				// no contribution to dd_log likelihood
			}else if(kigyo_joho.at(i).at(f_date) >= start_date && kigyo_joho.at(i).at(f_date) <= end_date ){
				double lam_2 = lambda_0*lambda_0;
				ind_logLike += - 1/(lam_2);
			}
		}
		logLike += ind_logLike;
	}

	MPI_Allreduce(&logLike, &r_v_logLike, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	return r_v_logLike;
}

double dd_logLike_l_b(const unordered_map<int, unordered_map<int, double> >& network, const unordered_map<int, unordered_map<int, double> >& network_b,
		const unordered_map<int, unordered_map<int, double> >& kigyo_joho, const unordered_map<int, unordered_map<int, double> >& macrovar,
		vector<double> parameter, int start_date, int end_date,
		int b_num,
		int my_rank, int p){

	double logLike = 0.0;
	double r_v_logLike;

	// int pre_bank = 1;
	int log_age = 2;
	int log_sales = 3;
	int hyoten = 4;
	int area_id = 5;
	int ind_id = 6;
	int EBITA = 7;
	int CHAT = 8;
	int PAA = 9;
	int REA = 10;
	int LCTAT = 11;
	int FAT = 12;
	int f_date = 13;
	int data_used = 14;

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

	double beta_acc_1 = parameter[71];//EBITA
	double beta_acc_2 = parameter[72];//CHAT
	double beta_acc_3 = parameter[73];//PAA
	double beta_acc_4 = parameter[74];//REA
	double beta_acc_5 = parameter[75];//LCTAT
	double beta_acc_6 = parameter[76];//FAT

	double contagion_dem = parameter[77];//contagion from demand side
	double contagion_sup = parameter[78];//contagion from supply side

	int ib = network_size/n_proc;
	int istat, iend;

	istat = my_rank*ib + 1;
	if(my_rank == (n_proc - 1) ){
		iend = network_size + 1;
	}else{
		iend = (my_rank + 1)*ib + 1;
	}

	#pragma omp parallel for reduction(+:logLike) schedule(dynamic,1)
	for(int i = istat; i < iend; ++i){

		double ind_logLike = 0.0;
		if(kigyo_joho.at(i).at(data_used) == 1.0){

			// area_FE, ind_FE
			int i_area = kigyo_joho.at(i).at(area_id);
			int i_ind = kigyo_joho.at(i).at(ind_id);

			// Z. ratio of bankrupt customers. Initialization
			double z_contag_dem = 0.0;
			double z_contag_sup = 0.0;

			// bankruptcies among customers
			unordered_map<int, vector<int> > set_f_date_cus;
			for(auto itr = network_b.at(i).begin(); itr != network_b.at(i).end(); ++itr){
				int cus_i = itr->first;
				int cus_i_f_date = (int)kigyo_joho.at(cus_i).at(f_date);
				if(cus_i_f_date < end_date) set_f_date_cus[cus_i_f_date].push_back(cus_i);
			}

			unordered_map<int, vector<int> > set_f_date_sup;
			for(auto itr = network.at(i).begin(); itr != network.at(i).end(); ++itr){
				int sup_i = itr->first;
				int sup_i_f_date = (int)kigyo_joho.at(sup_i).at(f_date);
				if(sup_i_f_date < end_date) set_f_date_sup[sup_i_f_date].push_back(sup_i);
			}

			if(kigyo_joho.at(i).at(f_date) > end_date){

				z_contag_dem = z_contag_dem_f(network_b, kigyo_joho, f_date, (start_date - 1), i);
				z_contag_sup = z_contag_sup_f(network, kigyo_joho, f_date, (start_date - 1), i);

				for(int t = start_date; t <= end_date; ++t){

					auto itr_cus = set_f_date_cus.find(t-1);
					if(itr_cus != set_f_date_cus.end() ){
						for(auto itr_b = set_f_date_cus.at(t-1).begin(); itr_b != set_f_date_cus.at(t-1).end(); ++itr_b) z_contag_dem += network_b.at(i).at(*itr_b);
					}
					auto itr_cus_exp = set_f_date_cus.find(t-1 - T_expire);
					if(itr_cus_exp != set_f_date_cus.end() ){
						for(auto itr_b = set_f_date_cus.at(t-1 - T_expire).begin(); itr_b != set_f_date_cus.at(t-1 - T_expire).end(); ++itr_b) z_contag_dem -= network_b.at(i).at(*itr_b);
					}

					auto itr_sup = set_f_date_sup.find(t-1);
					if(itr_sup != set_f_date_sup.end() ){
						for(auto itr_b = set_f_date_sup.at(t-1).begin(); itr_b != set_f_date_sup.at(t-1).end(); ++itr_b) z_contag_sup += network.at(i).at(*itr_b);
					}
					auto itr_sup_exp = set_f_date_sup.find(t-1 - T_expire);
					if(itr_sup_exp != set_f_date_sup.end() ){
						for(auto itr_b = set_f_date_sup.at(t-1 - T_expire).begin(); itr_b != set_f_date_sup.at(t-1 - T_expire).end(); ++itr_b) z_contag_sup -= network.at(i).at(*itr_b);
					}

					double z_target = z_target_f(kigyo_joho, macrovar, t, i, b_num, z_contag_dem, z_contag_sup);

					ind_logLike += -z_target*exp(beta_f_1*kigyo_joho.at(i).at(log_age) + beta_f_2*kigyo_joho.at(i).at(log_sales) + beta_f_3*kigyo_joho.at(i).at(hyoten) +
						beta_acc_1*kigyo_joho.at(i).at(EBITA) + beta_acc_2*kigyo_joho.at(i).at(CHAT) + beta_acc_3*kigyo_joho.at(i).at(PAA) + beta_acc_4*kigyo_joho.at(i).at(REA) + beta_acc_5*kigyo_joho.at(i).at(LCTAT) + beta_acc_6*kigyo_joho.at(i).at(FAT) +
						beta_m_1*macrovar.at(t).at(1) + beta_m_2*macrovar.at(t).at(2) + contagion_dem*z_contag_dem + contagion_sup*z_contag_sup + area_FE[i_area] + ind_FE[i_ind] );
				}

			}else if(kigyo_joho.at(i).at(f_date) >= start_date && kigyo_joho.at(i).at(f_date) <= end_date ){

				z_contag_dem = z_contag_dem_f(network_b, kigyo_joho, f_date, (start_date - 1), i);
				z_contag_sup = z_contag_sup_f(network, kigyo_joho, f_date, (start_date - 1), i);

				int bank_date = (int)kigyo_joho.at(i).at(f_date);

				for(int t = start_date; t < bank_date; ++t){

					auto itr_cus = set_f_date_cus.find(t-1);
					if(itr_cus != set_f_date_cus.end() ){
						for(auto itr_b = set_f_date_cus.at(t-1).begin(); itr_b != set_f_date_cus.at(t-1).end(); ++itr_b) z_contag_dem += network_b.at(i).at(*itr_b);
					}
					auto itr_cus_exp = set_f_date_cus.find(t-1 - T_expire);
					if(itr_cus_exp != set_f_date_cus.end() ){
						for(auto itr_b = set_f_date_cus.at(t-1 - T_expire).begin(); itr_b != set_f_date_cus.at(t-1 - T_expire).end(); ++itr_b) z_contag_dem -= network_b.at(i).at(*itr_b);
					}

					auto itr_sup = set_f_date_sup.find(t-1);
					if(itr_sup != set_f_date_sup.end() ){
						for(auto itr_b = set_f_date_sup.at(t-1).begin(); itr_b != set_f_date_sup.at(t-1).end(); ++itr_b) z_contag_sup += network.at(i).at(*itr_b);
					}
					auto itr_sup_exp = set_f_date_sup.find(t-1 - T_expire);
					if(itr_sup_exp != set_f_date_sup.end() ){
						for(auto itr_b = set_f_date_sup.at(t-1 - T_expire).begin(); itr_b != set_f_date_sup.at(t-1 - T_expire).end(); ++itr_b) z_contag_sup -= network.at(i).at(*itr_b);
					}

					double z_target = z_target_f(kigyo_joho, macrovar, t, i, b_num, z_contag_dem, z_contag_sup);

					ind_logLike += -z_target*exp(beta_f_1*kigyo_joho.at(i).at(log_age) + beta_f_2*kigyo_joho.at(i).at(log_sales) + beta_f_3*kigyo_joho.at(i).at(hyoten) +
						beta_acc_1*kigyo_joho.at(i).at(EBITA) + beta_acc_2*kigyo_joho.at(i).at(CHAT) + beta_acc_3*kigyo_joho.at(i).at(PAA) + beta_acc_4*kigyo_joho.at(i).at(REA) + beta_acc_5*kigyo_joho.at(i).at(LCTAT) + beta_acc_6*kigyo_joho.at(i).at(FAT) +
						beta_m_1*macrovar.at(t).at(1) + beta_m_2*macrovar.at(t).at(2) + contagion_dem*z_contag_dem + contagion_sup*z_contag_sup + area_FE[i_area] + ind_FE[i_ind] );
				}
			}
		}
		logLike += ind_logLike;
	}

	MPI_Allreduce(&logLike, &r_v_logLike, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

	return r_v_logLike;
}

double dd_logLike_b_b(const unordered_map<int, unordered_map<int, double> >& network, const unordered_map<int, unordered_map<int, double> >& network_b,
		const unordered_map<int, unordered_map<int, double> >& kigyo_joho, const unordered_map<int, unordered_map<int, double> >& macrovar,
		vector<double> parameter, int start_date, int end_date,
		int b_num_1, int b_num_2,
		int my_rank, int p){
	double logLike = 0.0;
	double r_v_logLike;

	// int pre_bank = 1;
	int log_age = 2;
	int log_sales = 3;
	int hyoten = 4;
	int area_id = 5;
	int ind_id = 6;
	int EBITA = 7;
	int CHAT = 8;
	int PAA = 9;
	int REA = 10;
	int LCTAT = 11;
	int FAT = 12;
	int f_date = 13;
	int data_used = 14;

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

	double beta_acc_1 = parameter[71];//EBITA
	double beta_acc_2 = parameter[72];//CHAT
	double beta_acc_3 = parameter[73];//PAA
	double beta_acc_4 = parameter[74];//REA
	double beta_acc_5 = parameter[75];//LCTAT
	double beta_acc_6 = parameter[76];//FAT

	double contagion_dem = parameter[77];//contagion from demand side
	double contagion_sup = parameter[78];//contagion from supply side

	int ib = network_size/n_proc;
	int istat, iend;

	istat = my_rank*ib + 1;
	if(my_rank == (n_proc - 1) ){
		iend = network_size + 1;
	}else{
		iend = (my_rank + 1)*ib + 1;
	}

	#pragma omp parallel for reduction(+:logLike) schedule(dynamic,1)
	for(int i = istat; i < iend; ++i){

		double ind_logLike = 0.0;
		if(kigyo_joho.at(i).at(data_used) == 1.0){

			// area_FE, ind_FE
			int i_area = kigyo_joho.at(i).at(area_id);
			int i_ind = kigyo_joho.at(i).at(ind_id);

			// Z. ratio of bankrupt customers. Initialization
			double z_contag_dem = 0.0;
			double z_contag_sup = 0.0;

			// bankruptcies among customers
			unordered_map<int, vector<int> > set_f_date_cus;
			for(auto itr = network_b.at(i).begin(); itr != network_b.at(i).end(); ++itr){
				int cus_i = itr->first;
				int cus_i_f_date = (int)kigyo_joho.at(cus_i).at(f_date);
				if(cus_i_f_date < end_date) set_f_date_cus[cus_i_f_date].push_back(cus_i);
			}

			unordered_map<int, vector<int> > set_f_date_sup;
			for(auto itr = network.at(i).begin(); itr != network.at(i).end(); ++itr){
				int sup_i = itr->first;
				int sup_i_f_date = (int)kigyo_joho.at(sup_i).at(f_date);
				if(sup_i_f_date < end_date) set_f_date_sup[sup_i_f_date].push_back(sup_i);
			}


			if(kigyo_joho.at(i).at(f_date) > end_date){

				z_contag_dem = z_contag_dem_f(network_b, kigyo_joho, f_date, (start_date - 1), i);
				z_contag_sup = z_contag_sup_f(network, kigyo_joho, f_date, (start_date - 1), i);

				for(int t = start_date; t <= end_date; ++t){

					auto itr_cus = set_f_date_cus.find(t-1);
					if(itr_cus != set_f_date_cus.end() ){
						for(auto itr_b = set_f_date_cus.at(t-1).begin(); itr_b != set_f_date_cus.at(t-1).end(); ++itr_b) z_contag_dem += network_b.at(i).at(*itr_b);
					}
					auto itr_cus_exp = set_f_date_cus.find(t-1 - T_expire);
					if(itr_cus_exp != set_f_date_cus.end() ){
						for(auto itr_b = set_f_date_cus.at(t-1 - T_expire).begin(); itr_b != set_f_date_cus.at(t-1 - T_expire).end(); ++itr_b) z_contag_dem -= network_b.at(i).at(*itr_b);
					}

					auto itr_sup = set_f_date_sup.find(t-1);
					if(itr_sup != set_f_date_sup.end() ){
						for(auto itr_b = set_f_date_sup.at(t-1).begin(); itr_b != set_f_date_sup.at(t-1).end(); ++itr_b) z_contag_sup += network.at(i).at(*itr_b);
					}
					auto itr_sup_exp = set_f_date_sup.find(t-1 - T_expire);
					if(itr_sup_exp != set_f_date_sup.end() ){
						for(auto itr_b = set_f_date_sup.at(t-1 - T_expire).begin(); itr_b != set_f_date_sup.at(t-1 - T_expire).end(); ++itr_b) z_contag_sup -= network.at(i).at(*itr_b);
					}

					double z_target_1 = z_target_f(kigyo_joho, macrovar, t, i, b_num_1, z_contag_dem, z_contag_sup);
					double z_target_2 = z_target_f(kigyo_joho, macrovar, t, i, b_num_2, z_contag_dem, z_contag_sup);

					ind_logLike += -lambda_0*z_target_1*z_target_2*exp(beta_f_1*kigyo_joho.at(i).at(log_age) + beta_f_2*kigyo_joho.at(i).at(log_sales) + beta_f_3*kigyo_joho.at(i).at(hyoten) +
						beta_acc_1*kigyo_joho.at(i).at(EBITA) + beta_acc_2*kigyo_joho.at(i).at(CHAT) + beta_acc_3*kigyo_joho.at(i).at(PAA) + beta_acc_4*kigyo_joho.at(i).at(REA) + beta_acc_5*kigyo_joho.at(i).at(LCTAT) + beta_acc_6*kigyo_joho.at(i).at(FAT) +
						beta_m_1*macrovar.at(t).at(1) + beta_m_2*macrovar.at(t).at(2) + contagion_dem*z_contag_dem + contagion_sup*z_contag_sup + area_FE[i_area] + ind_FE[i_ind] );
				}

			}else if(kigyo_joho.at(i).at(f_date) > (start_date -1) && kigyo_joho.at(i).at(f_date) < (end_date + 1) ){

				z_contag_dem = z_contag_dem_f(network_b, kigyo_joho, f_date, (start_date - 1), i);
				z_contag_sup = z_contag_sup_f(network, kigyo_joho, f_date, (start_date - 1), i);

				int bank_date = (int)kigyo_joho.at(i).at(f_date);

				for(int t = start_date; t < bank_date; ++t){

					auto itr_cus = set_f_date_cus.find(t-1);
					if(itr_cus != set_f_date_cus.end() ){
						for(auto itr_b = set_f_date_cus.at(t-1).begin(); itr_b != set_f_date_cus.at(t-1).end(); ++itr_b) z_contag_dem += network_b.at(i).at(*itr_b);
					}
					auto itr_cus_exp = set_f_date_cus.find(t-1 - T_expire);
					if(itr_cus_exp != set_f_date_cus.end() ){
						for(auto itr_b = set_f_date_cus.at(t-1 - T_expire).begin(); itr_b != set_f_date_cus.at(t-1 - T_expire).end(); ++itr_b) z_contag_dem -= network_b.at(i).at(*itr_b);
					}

					auto itr_sup = set_f_date_sup.find(t-1);
					if(itr_sup != set_f_date_sup.end() ){
						for(auto itr_b = set_f_date_sup.at(t-1).begin(); itr_b != set_f_date_sup.at(t-1).end(); ++itr_b) z_contag_sup += network.at(i).at(*itr_b);
					}
					auto itr_sup_exp = set_f_date_sup.find(t-1 - T_expire);
					if(itr_sup_exp != set_f_date_sup.end() ){
						for(auto itr_b = set_f_date_sup.at(t-1 - T_expire).begin(); itr_b != set_f_date_sup.at(t-1 - T_expire).end(); ++itr_b) z_contag_sup -= network.at(i).at(*itr_b);
					}

					double z_target_1 = z_target_f(kigyo_joho, macrovar, t, i, b_num_1, z_contag_dem, z_contag_sup);
					double z_target_2 = z_target_f(kigyo_joho, macrovar, t, i, b_num_2, z_contag_dem, z_contag_sup);

					ind_logLike += -lambda_0*z_target_1*z_target_2*exp(beta_f_1*kigyo_joho.at(i).at(log_age) + beta_f_2*kigyo_joho.at(i).at(log_sales) + beta_f_3*kigyo_joho.at(i).at(hyoten) +
						beta_acc_1*kigyo_joho.at(i).at(EBITA) + beta_acc_2*kigyo_joho.at(i).at(CHAT) + beta_acc_3*kigyo_joho.at(i).at(PAA) + beta_acc_4*kigyo_joho.at(i).at(REA) + beta_acc_5*kigyo_joho.at(i).at(LCTAT) + beta_acc_6*kigyo_joho.at(i).at(FAT) +
						beta_m_1*macrovar.at(t).at(1) + beta_m_2*macrovar.at(t).at(2) + contagion_dem*z_contag_dem + contagion_sup*z_contag_sup + area_FE[i_area] + ind_FE[i_ind] );
				}

			}
		}
		logLike += ind_logLike;
	}

	MPI_Allreduce(&logLike, &r_v_logLike, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

	return r_v_logLike;
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

double z_target_f(const unordered_map<int, unordered_map<int, double> >& kigyo_joho, const unordered_map<int, unordered_map<int, double> >& macrovar, const int& t_date, const int& target_i, const int& b_num,
	double z_contag_dem_var, double z_contag_sup_var){

	int log_age = 2;
	int log_sales = 3;
	int hyoten = 4;
	int area_id = 5;
	int ind_id = 6;
	int EBITA = 7;
	int CHAT = 8;
	int PAA = 9;
	int REA = 10;
	int LCTAT = 11;
	int FAT = 12;

	int target_i_area = kigyo_joho.at(target_i).at(area_id);
	int target_i_ind = kigyo_joho.at(target_i).at(ind_id);

	double z_target_var = 0.0;
	if(b_num == 1){
		z_target_var = kigyo_joho.at(target_i).at(log_age);
	}else if(b_num == 2){
		z_target_var = kigyo_joho.at(target_i).at(log_sales);
	}else if(b_num == 3){
		z_target_var = kigyo_joho.at(target_i).at(hyoten);
	}else if(b_num == 4){
		z_target_var = macrovar.at(t_date).at(1);
	}else if(b_num == 5){
		z_target_var = macrovar.at(t_date).at(2);
	}else if(b_num >= 6 && b_num <= 12){
		if(target_i_area == b_num - 5) z_target_var = 1.0;
	}else if(b_num >= 13 && b_num <= 70){
		if(target_i_ind == b_num - 12) z_target_var = 1.0;
	}else if(b_num == 71){
		z_target_var = kigyo_joho.at(target_i).at(EBITA);
	}else if(b_num == 72){
		z_target_var = kigyo_joho.at(target_i).at(CHAT);
	}else if(b_num == 73){
		z_target_var = kigyo_joho.at(target_i).at(PAA);
	}else if(b_num == 74){
		z_target_var = kigyo_joho.at(target_i).at(REA);
	}else if(b_num == 75){
		z_target_var = kigyo_joho.at(target_i).at(LCTAT);
	}else if(b_num == 76){
		z_target_var = kigyo_joho.at(target_i).at(FAT);
	}else if(b_num == 77){
		z_target_var = z_contag_dem_var;
	}else if(b_num == 78){
		z_target_var = z_contag_sup_var;
	}
	return(z_target_var);
}
