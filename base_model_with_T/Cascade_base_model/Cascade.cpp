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
#include "Data_cascade.h"

using namespace std;

int n_proc = 48;

int n_time = 1500;//for Nelder_Mead iteratioin
int network_size = 1080977;// number of firms
int n_item = 9;// firm information items
const int start_date = 371;
const int end_date = 960;
const int n_dim = 73;//number of parameters. 1(const) + 3 firm characteristics (age, sales, hyoten) + 2(macro) + 2(contagion demand) + 7(area FE, base=Kanto) + 58(industry FE, base=JSIC 5)
const int T_expire = 246;

double logLike_f(const unordered_map<int, unordered_map<int, double> >& network, const unordered_map<int, unordered_map<int, double> >& network_b,
		const unordered_map<int, unordered_map<int, double> >& kigyo_joho, const unordered_map<int, unordered_map<int, double> >& macrovar,
		vector<double> parameter, int start_date, int end_date,
		int my_rank, int p);
vector<double> Nelder_Mead(vector<double> initial, vector<double> unit, int dim,
		const unordered_map<int, unordered_map<int, double> >& network, const unordered_map<int, unordered_map<int, double> >& network_b,
		const unordered_map<int, unordered_map<int, double> >& kigyo_joho, const unordered_map<int, unordered_map<int, double> >& macrovar,
		int start_date, int end_date,
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

	// vector<double> test_para(n_dim, 0);
	// test_para[0] = 1.24573135135135e-05;


  double t_start, t_finish;
	int my_rank;
	int p;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &p);

	if(my_rank == 0) cout << "n_proc is " << n_proc << endl;
	if(my_rank == 0) cout << "number of paramters is " << n_dim << endl;
	if(my_rank == 0) cout << "number of items is " << n_item << endl;

	// t_start = MPI_Wtime();
	// double logLike = logLike_f(network_1, network_1_b, kigyo_joho_1, macrovar_1, test_para, start_date, end_date, my_rank, p);
	// t_finish = MPI_Wtime();
  // if(my_rank == 0) cout << "Single logLike_f Elapsed time is " << t_finish - t_start << endl;
	// if(my_rank == 0) cout << "logLike " << logLike << endl;


	vector<double> parameter{
		1.22788324324324e-05,-0.040641322972973,0.561598094594595,-0.845437689189189,-0.006411355,0.869282256756757,0.00429297255405405,-0.166109162162162,-0.143105716216216,-0.101124405405405,-0.426832905405405,-0.151744959459459,-0.38076977027027,-0.0227171758972973,-0.341294094594595,-0.35488327027027,0.463665189189189,-0.198738810810811,0.785399364864865,0.530347297297297,0.493592067567568,-0.0520058945945946,0.65254827027027,-0.10114597027027,0.186309581081081,0.0942289594594595,0.816933202702703,0.171410837837838,0.239703756756757,0.0524058810810811,-0.152083972972973,-0.216442621621622,0.175986905405405,0.473263783783784,0.627113540540541,0.244652621621622,0.13752227027027,-0.10354637027027,0.300126959459459,0.438622081081081,0.22998827027027,0.615923310810811,0.450131135135135,0.164291175675676,0.127760121621622,-0.135320021621622,-0.552975891891892,0.100819341891892,0.255303689189189,0.884217932432432,0.306553027027027,-0.00726214689189189,0.197074337837838,0.378172094594595,0.285115459459459,0.468647918918919,-0.13337772972973,-0.315420364864865,-0.0742486459459459,0.278834297297297,-0.715063972972973,-0.932708445945946,-0.575305405405405,-0.188487243243243,0.121167067567568,-0.235813216216216,0.187499486486486,0.161635675675676,0.0135863908783784,-0.0312058054054054,0.091474972972973,1.94383621621622,1.20872351351351
	};

	vector<double> test_unit{
		3.75999999999994e-08,0.0030368,0.00412299999999999,0.00355800000000006,0.00201301,0.073671,0.02195312,0.014004,0.009071,0.0078532,0.01701,0.019413,0.013142,0.115205,0.01273,0.011331,0.019176,0.079552,0.02023,0.051368,0.040824,0.0391783,0.026343,0.048334,0.032148,0.0764759,0.0540350000000001,0.033876,0.049631,0.0705089,0.02084,0.060877,0.034841,0.043926,0.0250600000000001,0.033907,0.064983,0.0442554,0.025488,0.086699,0.018432,0.055149,0.039958,0.050677,0.019838,0.1333064,0.16267,0.0653786,0.043908,0.017532,0.018573,0.01883889,0.02301,0.013693,0.041872,0.018709,0.019981,0.024159,0.013884,0.117924,0.025014,0.0392089999999999,0.087785,0.017141,0.054012,0.029116,0.038725,0.023304,0.0285855,0.0403141,0.0320301,0.0425800000000001,0.0453599999999998
	};

	if(my_rank == 0) cout << "start_date is " << start_date << ", " << "end_date is " << end_date << endl;

	t_start = MPI_Wtime();
  vector<double> test = Nelder_Mead(parameter, test_unit, n_dim, network_1, network_1_b, kigyo_joho_1, macrovar_1, start_date, end_date, my_rank, p);
  t_finish = MPI_Wtime();
  if(my_rank == 0) cout << "Elapsed time is " << t_finish - t_start << endl;

  MPI_Finalize();
	return 0;
}


double logLike_f(const unordered_map<int, unordered_map<int, double> >& network, const unordered_map<int, unordered_map<int, double> >& network_b,
		const unordered_map<int, unordered_map<int, double> >& kigyo_joho, const unordered_map<int, unordered_map<int, double> >& macrovar,
		vector<double> parameter, int start_date, int end_date,
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

	if(lambda_0 < 0.0){
		cout << "EMERGENCY" << endl;
		lambda_0 = 0.00000001;
	}

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

		// int digit_1 = i % 5;
		// if(digit_1 != 1) continue;
		// if(kigyo_joho.at(i).at(log_sales) >= 4.5) continue;

		double ind_logLike = 0.0;
		if(kigyo_joho.at(i).at(data_used) == 1.0){

			// area_FE, ind_FE
			int i_area = kigyo_joho.at(i).at(area_id);
			int i_ind = kigyo_joho.at(i).at(ind_id);

			// Z. ratio of bankrupt customers. Initialization
			double z_contag_dem = 0.0;
			double z_contag_sup = 0.0;

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

				if(set_f_date_cus.size() == 0 && set_f_date_sup.size() == 0){
					ind_logLike += (
						(392 - 371 + 1)*exp(beta_m_1*macrovar.at(371).at(1) + beta_m_2*macrovar.at(371).at(2) ) +
						(410 - 393 + 1)*exp(beta_m_1*macrovar.at(393).at(1) + beta_m_2*macrovar.at(393).at(2) ) +
						(431 - 411 + 1)*exp(beta_m_1*macrovar.at(411).at(1) + beta_m_2*macrovar.at(411).at(2) ) +
						(450 - 432 + 1)*exp(beta_m_1*macrovar.at(432).at(1) + beta_m_2*macrovar.at(432).at(2) ) +
						(469 - 450 + 1)*exp(beta_m_1*macrovar.at(450).at(1) + beta_m_2*macrovar.at(450).at(2) ) +
						(491 - 470 + 1)*exp(beta_m_1*macrovar.at(470).at(1) + beta_m_2*macrovar.at(470).at(2) ) +
						// 2015 fiscal year
						(512 - 492 + 1)*exp(beta_m_1*macrovar.at(492).at(1) + beta_m_2*macrovar.at(492).at(2) ) +
						(530 - 513 + 1)*exp(beta_m_1*macrovar.at(513).at(1) + beta_m_2*macrovar.at(513).at(2) ) +
						(552 - 531 + 1)*exp(beta_m_1*macrovar.at(531).at(1) + beta_m_2*macrovar.at(531).at(2) ) +
						(574 - 553 + 1)*exp(beta_m_1*macrovar.at(553).at(1) + beta_m_2*macrovar.at(553).at(2) ) +
						(595 - 575 + 1)*exp(beta_m_1*macrovar.at(575).at(1) + beta_m_2*macrovar.at(575).at(2) ) +
						(614 - 596 + 1)*exp(beta_m_1*macrovar.at(596).at(1) + beta_m_2*macrovar.at(596).at(2) ) +
						(635 - 615 + 1)*exp(beta_m_1*macrovar.at(615).at(1) + beta_m_2*macrovar.at(615).at(2) ) +
						(654 - 636 + 1)*exp(beta_m_1*macrovar.at(636).at(1) + beta_m_2*macrovar.at(636).at(2) ) +
						(675 - 655 + 1)*exp(beta_m_1*macrovar.at(655).at(1) + beta_m_2*macrovar.at(655).at(2) ) +
						(694 - 676 + 1)*exp(beta_m_1*macrovar.at(676).at(1) + beta_m_2*macrovar.at(676).at(2) ) +
						(714 - 695 + 1)*exp(beta_m_1*macrovar.at(695).at(1) + beta_m_2*macrovar.at(695).at(2) ) +
						(736 - 715 + 1)*exp(beta_m_1*macrovar.at(715).at(1) + beta_m_2*macrovar.at(715).at(2) ) +
						// 2016 fiscal year
						(756 - 737 + 1)*exp(beta_m_1*macrovar.at(737).at(1) + beta_m_2*macrovar.at(737).at(2) ) +
						(775 - 757 + 1)*exp(beta_m_1*macrovar.at(757).at(1) + beta_m_2*macrovar.at(757).at(2) ) +
						(797 - 776 + 1)*exp(beta_m_1*macrovar.at(776).at(1) + beta_m_2*macrovar.at(776).at(2) ) +
						(817 - 798 + 1)*exp(beta_m_1*macrovar.at(798).at(1) + beta_m_2*macrovar.at(798).at(2) ) +
						(839 - 818 + 1)*exp(beta_m_1*macrovar.at(818).at(1) + beta_m_2*macrovar.at(818).at(2) ) +
						(859 - 840 + 1)*exp(beta_m_1*macrovar.at(840).at(1) + beta_m_2*macrovar.at(840).at(2) ) +
						(879 - 860 + 1)*exp(beta_m_1*macrovar.at(860).at(1) + beta_m_2*macrovar.at(860).at(2) ) +
						(899 - 880 + 1)*exp(beta_m_1*macrovar.at(880).at(1) + beta_m_2*macrovar.at(880).at(2) ) +
						(920 - 900 + 1)*exp(beta_m_1*macrovar.at(900).at(1) + beta_m_2*macrovar.at(900).at(2) ) +
						(940 - 921 + 1)*exp(beta_m_1*macrovar.at(921).at(1) + beta_m_2*macrovar.at(921).at(2) ) +
						(960 - 941 + 1)*exp(beta_m_1*macrovar.at(941).at(1) + beta_m_2*macrovar.at(941).at(2) )
					)*( -lambda_0*exp(beta_f_1*kigyo_joho.at(i).at(log_age) + beta_f_2*kigyo_joho.at(i).at(log_sales) + beta_f_3*kigyo_joho.at(i).at(hyoten) + area_FE[i_area] + ind_FE[i_ind] ) );

				}else{

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

						ind_logLike += -lambda_0*exp(beta_f_1*kigyo_joho.at(i).at(log_age) + beta_f_2*kigyo_joho.at(i).at(log_sales) + beta_f_3*kigyo_joho.at(i).at(hyoten) +
							beta_m_1*macrovar.at(t).at(1) + beta_m_2*macrovar.at(t).at(2) + contagion_dem*z_contag_dem + contagion_sup*z_contag_sup + area_FE[i_area] + ind_FE[i_ind] );
					}
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

					ind_logLike += -lambda_0*exp(beta_f_1*kigyo_joho.at(i).at(log_age) + beta_f_2*kigyo_joho.at(i).at(log_sales) + beta_f_3*kigyo_joho.at(i).at(hyoten) +
						beta_m_1*macrovar.at(t).at(1) + beta_m_2*macrovar.at(t).at(2) + contagion_dem*z_contag_dem + contagion_sup*z_contag_sup + area_FE[i_area] + ind_FE[i_ind] );
				}

				z_contag_dem = z_contag_dem_f(network_b, kigyo_joho, f_date, bank_date, i);
				z_contag_sup = z_contag_sup_f(network, kigyo_joho, f_date, bank_date, i);
				ind_logLike += log(lambda_0) + beta_f_1*kigyo_joho.at(i).at(log_age) + beta_f_2*kigyo_joho.at(i).at(log_sales) + beta_f_3*kigyo_joho.at(i).at(hyoten) +
					beta_m_1*macrovar.at(bank_date).at(1) + beta_m_2*macrovar.at(bank_date).at(2) + contagion_dem*z_contag_dem + contagion_sup*z_contag_sup + area_FE[i_area] + ind_FE[i_ind];
			}
		}
		logLike += ind_logLike;
	}

	MPI_Allreduce(&logLike, &r_v_logLike, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

	return r_v_logLike;
}

vector<double> Nelder_Mead(vector<double> initial, vector<double> unit, int dim,
		const unordered_map<int, unordered_map<int, double> >& network, const unordered_map<int, unordered_map<int, double> >& network_b,
		const unordered_map<int, unordered_map<int, double> >& kigyo_joho, const unordered_map<int, unordered_map<int, double> >& macrovar,
		int start_date, int end_date,
		int my_rank, int p){

	double reflection = 1.0;
	double expansion = 2.0;
	double contraction = 0.5;
	double shrink = 0.5;

	vector<vector<double> > simplex(dim + 1, vector<double>(dim, 0) );

	for(int i = 0; i < dim + 1; ++i){
		for(int j = 0; j < dim; ++j){
			simplex[i][j] = initial[j];
		}
	}
	for(int i  = 1; i < dim + 1; ++i){
		simplex[i][i - 1] += unit[i - 1];
	}

	for(int time = 0; time < n_time; ++time){
		vector<double> f_values(dim + 1);
		for(int i = 0; i < dim + 1; ++i){
			f_values[i] = - logLike_f(network, network_b, kigyo_joho, macrovar, simplex[i], start_date, end_date, my_rank, p);
		}
		vector<double> sort_f_values(f_values);
		sort(sort_f_values.begin(), sort_f_values.end());
		double f_value_h =  sort_f_values[dim];
		double f_value_s =  sort_f_values[dim - 1];
		double f_value_l =  sort_f_values[0];
		if(my_rank == 0) cout << "f_value_l " << f_value_l << endl;

		auto itr = find(f_values.begin(), f_values.end(), f_value_h);
		int x_h_index = itr - f_values.begin();
		itr = find(f_values.begin(), f_values.end(), f_value_l);
		int x_l_index = itr - f_values.begin();

		vector<double> x_g(dim, 0);
		for(int j = 0; j < dim; ++j){
			for(int i = 0; i < dim + 1; ++i){
				x_g[j] += simplex[i][j];
			}
		}

		//		x_h_index is removed
		for(int j = 0; j < dim; ++j){
			x_g[j] -= simplex[x_h_index][j];
			x_g[j] = x_g[j]/dim;
		}

		vector<double> x_r(dim, 0.0);
		for(int j = 0; j < dim; ++j){
			x_r[j] = (1 + reflection)*x_g[j] - reflection*simplex[x_h_index][j];
		}
		//		double f_value_r = obj_function(x_r);
		double f_value_r = - logLike_f(network, network_b, kigyo_joho, macrovar, x_r, start_date, end_date, my_rank, p);

		//  4
		if(f_value_r < f_value_l){
			vector<double> x_e(dim, 0.0);
			for(int j = 0; j < dim; ++j){
				x_e[j] = expansion*x_r[j] + (1 - expansion)*x_g[j];
			}
			double f_value_e = - logLike_f(network, network_b, kigyo_joho, macrovar, x_e, start_date, end_date, my_rank, p);
			if(f_value_e < f_value_r){
				for(int j = 0; j < dim; ++j){
					simplex[x_h_index][j] = x_e[j];
				}
			}
			if(f_value_e >= f_value_r){
				for(int j = 0; j < dim; ++j){
					simplex[x_h_index][j] = x_r[j];
				}
			}
			//  5
		}else if( (f_value_s < f_value_r) && (f_value_r < f_value_h) ){
			for(int j = 0; j < dim; ++j){
				simplex[x_h_index][j] = x_r[j];
			}
			vector<double> x_c(dim, 0.0);
			for(int j = 0; j < dim; ++j){
				x_c[j] = contraction*simplex[x_h_index][j] + (1 - contraction)*x_g[j];
			}
			double f_value_c = - logLike_f(network, network_b, kigyo_joho, macrovar, x_c, start_date, end_date, my_rank, p);
			if(f_value_c < f_value_h){
				for(int j = 0; j < dim; ++j){
					simplex[x_h_index][j] = x_c[j];
				}
			}else{
				//	Shrink
				for(int i = 0; i < dim + 1; ++i){
					for(int j = 0; j < dim; ++j){
					simplex[i][j] = shrink*(simplex[i][j] + simplex[x_l_index][j]);
					}
				}
			}
			//	6
		}else if(f_value_h <= f_value_r){
			vector<double> x_c(dim, 0.0);
			for(int j = 0; j < dim; ++j){
				x_c[j] = contraction*simplex[x_h_index][j] + (1 - contraction)*x_g[j];
			}
			double f_value_c = - logLike_f(network, network_b, kigyo_joho, macrovar, x_c, start_date, end_date, my_rank, p);
			if(f_value_c < f_value_h){
				for(int j = 0; j < dim; ++j){
					simplex[x_h_index][j] = x_c[j];
				}
			}else{
				for(int i = 0; i < dim + 1; ++i){
					for(int j = 0; j < dim; ++j){
						simplex[i][j] = shrink*(simplex[i][j] + simplex[x_l_index][j]);
					}
				}
			}
		}else{
			for(int j = 0; j < dim; ++j){
				simplex[x_h_index][j] = x_r[j];
			}
		}
		if(time > (n_time - 300) ){
			if(my_rank == 0){
				cout << "time " << time << endl;;
				for(int i = 0; i < dim + 1; ++i){
					for(int j = 0; j < dim; ++j) cout << simplex[i][j] << ",";
					cout << endl;
				}
			}
		}
	}
	return simplex[1];
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
