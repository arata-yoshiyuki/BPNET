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
#include "Data_cascade_Var.h"

using namespace std;

int n_proc = 48;

int network_size = 1080977;// number of firms
int n_item = 9;// firm information items
const int start_date = 371;
const int end_date = 960;
const int n_dim = 73;//number of parameters. 1(const) + 3 firm characteristics (age, sales, hyoten) + 2(macro) + 2(contagion demand) + 7(area FE, base=Kanto) + 58(industry FE, base=JSIC 5)
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

	filename = "TSR_main_full.txt";
	data_KJ(filename, kigyo_joho_1);
	filename = "macrovar.txt";
	data_macro(filename, macrovar_1);

	vector<double> parameter{
		1.25674256756757e-05,-0.0433104418918919,0.5618335,-0.847737418918919,-0.00699151554054054,0.82515377027027,0.00465551775675676,-0.163518324324324,-0.137191162162162,-0.0993510986486486,-0.411272905405405,-0.150992864864865,-0.380718027027027,-0.11697215,-0.383927432432432,-0.402780891891892,0.434707486486486,-0.306981283783784,0.756867905405405,0.470840081081081,0.413427472972973,-0.139075972972973,0.606657459459459,-0.117968993243243,0.150159081081081,-0.00880390113243243,0.729846702702703,0.107748074324324,0.164540310810811,-0.0222464943243243,-0.197991540540541,-0.281044297297297,0.153067405405405,0.402452743243243,0.577995513513514,0.1803935,0.0373278848918919,-0.147354675675676,0.259610175675676,0.369185337837838,0.191667689189189,0.550849364864865,0.394723702702703,0.0875925918918919,0.0857207878378378,-0.196473621621622,-0.665720702702703,0.0456518864864865,0.194931945945946,0.844135121621622,0.25714972972973,-0.0468817554054054,0.153142135135135,0.343945256756757,0.228507527027027,0.428425702702703,-0.172067918918919,-0.352142459459459,-0.108157997297297,0.131826444594595,-0.761067162162162,-0.980494013513514,-0.6888575,-0.244478945945946,0.0704011162162162,-0.291198432432432,0.189322743243243,0.121262472972973,-0.0530391135135135,-0.0564409918918919,0.0482154337837838,1.12592189189189,0.855282445945946
	};

	double t_start, t_finish;
	int my_rank, p;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &p);

	t_start = MPI_Wtime();
  vector<vector<double> > dd_matrix(n_dim, vector<double>(n_dim, 0));
  dd_matrix[0][0] = dd_logLike_l_l(network_1, network_1_b, kigyo_joho_1, macrovar_1, parameter, start_date, end_date, my_rank, p);
  for(int i = 1; i < n_dim; ++i){
  	dd_matrix[0][i] = dd_logLike_l_b(network_1, network_1_b, kigyo_joho_1, macrovar_1, parameter, start_date, end_date, i, my_rank, p);
  }
  for(int i = 1; i < n_dim; ++i){
  	for(int j = i; j < n_dim; ++j){
  		dd_matrix[i][j] = dd_logLike_b_b(network_1, network_1_b, kigyo_joho_1, macrovar_1, parameter, start_date, end_date, i, j, my_rank, p);
  	}
  }
  for(int i = 1; i < n_dim; ++i){
  	dd_matrix[i][0] = dd_matrix[0][i];
  }
  for(int i = 1; i < n_dim; ++i){
  	for(int j = 1; j < i; ++j){
  		dd_matrix[i][j] = dd_matrix[j][i];
  	}
  }

	if(my_rank == 0){
		cout << "dd_matrix " << endl;
	  for(int i = 0; i < n_dim; ++i){
	  	for(int j = 0; j < n_dim; ++j){
	  		cout << dd_matrix[i][j] << ", ";
	  	}
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
		z_target_var = z_contag_dem_var;
	}else if(b_num == 72){
		z_target_var = z_contag_sup_var;
	}
	return(z_target_var);
}
