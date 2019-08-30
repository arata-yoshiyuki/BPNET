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
#include "Data_cascade_acc.h"

using namespace std;

int n_proc = 48;

int n_time = 1500;//for Nelder_Mead iteratioin
int network_size = 1080977;// number of firms
int n_item = 15;// firm information items
const int start_date = 371;
const int end_date = 960;
const int n_dim = 79;//number of parameters. 1(const) + 3 firm characteristics (age, sales, hyoten) + 2(macro) + 2(contagion demand) + 7(area FE, base=Kanto) + 58(industry FE, base=JSIC 5) + 6(financial var)
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

	filename = "TSR_main_acc.txt";
	data_KJ(filename, kigyo_joho_1);
	filename = "macrovar.txt";
	data_macro(filename, macrovar_1);

  double t_start, t_finish;
	int my_rank, p;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &p);

	if(my_rank == 0) cout << "n_proc is " << n_proc << endl;
	if(my_rank == 0) cout << "number of paramters is " << n_dim << endl;
	if(my_rank == 0) cout << "number of items is " << n_item << endl;

	vector<double> parameter{
		6.2089185e-06,-0.05261190875,0.503715625,-0.8488345375,-0.02987403875,-0.29766695,0.2908730375,-0.02086237625,-0.1193521625,-0.2225244,-0.3432124375,-0.2261387,-0.3676229,-0.210914,-0.4479030625,-0.4576824875,-0.07457645125,-1.475557625,0.5294996,-0.17970757,-0.801543675,-1.627490125,0.22403785,0.1371041,-0.4773879875,-11.089275,0.8232522375,-0.3299523,-0.89974205,-0.0077654002375,-0.26192345,-0.5425451,0.07421035,0.10147724875,0.58578755,0.022578903,-0.1187185875,-0.3492300875,0.3836797,1.0031617,0.6677702375,0.2750937375,0.4900733375,-0.09099984,-0.3530641375,-0.2007902625,-0.8007211625,-0.4798181125,-0.6706589625,1.22734625,0.01922754525,-0.3129085625,0.0322027275,0.30542935,-0.4572981375,0.3504920625,-0.483257625,0.05932367,-0.048841791125,-0.105745655,-0.6122874375,-1.133953,-0.2717375125,0.09939038375,0.2019485,0.2970375125,-0.6805512125,-0.4653498,0.167448075,0.373894675,-0.00600983021125,-0.1675264875,-1.02209675,0.1447617,-0.1028842575,-0.1497919625,0.1893423375,2.25314525,0.8668889
	};

	vector<double> test_unit{
		1.6606e-07,0.0095973,0.00964500000000001,0.01976,0.0075281,0.090167,0.020685,0.0467315,0.012752,0.017649,0.033749,0.025701,0.018111,0.162741,0.024658,0.030116,0.0630678,0.31233,0.0938330000000001,0.1543854,0.148574,0.17641,0.119958,0.104256,0.195186,1.576,0.20985,0.061887,0.135323,0.2513942,0.097074,0.073445,0.0716331,0.0809286,0.0826070000000001,0.11218447,0.1728412,0.040655,0.061125,0.154363,0.042439,0.254791,0.083436,0.1116562,0.085691,0.235328,0.367738,0.20727,0.260197,0.06179,0.0709623,0.057379,0.0244867,0.061745,0.119659,0.061752,0.104848,0.0582173,0.04938021,0.2521411,0.046301,0.0636999999999999,0.124185,0.0694265,0.1291,0.028263,0.20435,0.0704520000000001,0.132245,0.134197,0.0850522,0.007385,0.00551999999999997,0.0053,0.00905839999999999,0.00657199999999999,0.00304399999999999,0.0929799999999998,0.1261
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
					)*( -lambda_0*exp(beta_f_1*kigyo_joho.at(i).at(log_age) + beta_f_2*kigyo_joho.at(i).at(log_sales) + beta_f_3*kigyo_joho.at(i).at(hyoten) +
						beta_acc_1*kigyo_joho.at(i).at(EBITA) + beta_acc_2*kigyo_joho.at(i).at(CHAT) + beta_acc_3*kigyo_joho.at(i).at(PAA) + beta_acc_4*kigyo_joho.at(i).at(REA) + beta_acc_5*kigyo_joho.at(i).at(LCTAT) + beta_acc_6*kigyo_joho.at(i).at(FAT) +
						area_FE[i_area] + ind_FE[i_ind] ) );

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
							beta_acc_1*kigyo_joho.at(i).at(EBITA) + beta_acc_2*kigyo_joho.at(i).at(CHAT) + beta_acc_3*kigyo_joho.at(i).at(PAA) + beta_acc_4*kigyo_joho.at(i).at(REA) + beta_acc_5*kigyo_joho.at(i).at(LCTAT) + beta_acc_6*kigyo_joho.at(i).at(FAT) +
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
						beta_acc_1*kigyo_joho.at(i).at(EBITA) + beta_acc_2*kigyo_joho.at(i).at(CHAT) + beta_acc_3*kigyo_joho.at(i).at(PAA) + beta_acc_4*kigyo_joho.at(i).at(REA) + beta_acc_5*kigyo_joho.at(i).at(LCTAT) + beta_acc_6*kigyo_joho.at(i).at(FAT) +
						beta_m_1*macrovar.at(t).at(1) + beta_m_2*macrovar.at(t).at(2) + contagion_dem*z_contag_dem + contagion_sup*z_contag_sup + area_FE[i_area] + ind_FE[i_ind] );
				}

				z_contag_dem = z_contag_dem_f(network_b, kigyo_joho, f_date, bank_date, i);
				z_contag_sup = z_contag_sup_f(network, kigyo_joho, f_date, bank_date, i);
				ind_logLike += log(lambda_0) + beta_f_1*kigyo_joho.at(i).at(log_age) + beta_f_2*kigyo_joho.at(i).at(log_sales) + beta_f_3*kigyo_joho.at(i).at(hyoten) +
					beta_acc_1*kigyo_joho.at(i).at(EBITA) + beta_acc_2*kigyo_joho.at(i).at(CHAT) + beta_acc_3*kigyo_joho.at(i).at(PAA) + beta_acc_4*kigyo_joho.at(i).at(REA) + beta_acc_5*kigyo_joho.at(i).at(LCTAT) + beta_acc_6*kigyo_joho.at(i).at(FAT) +
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
