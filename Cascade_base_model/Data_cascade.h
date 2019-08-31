/*
 * Data_casade.h
 *
 *  Created on: 2017/06/18
 *      Author: arata-
 */

#ifndef DATA_H_
#define DATA_H_

#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <sstream>
#include <algorithm>
#include <unordered_map>

using namespace std;

extern int network_size;
extern int n_item;

void data(string filename, unordered_map<int, unordered_map<int, double> >& network);
void data_b(string filename, unordered_map<int, unordered_map<int, double> >& network);
void data_KJ(string filename, unordered_map<int, unordered_map<int, double> >& kigyo_joho);
void data_macro(string filename, unordered_map<int, unordered_map<int, double> >& macrovar);
vector<string> split(string &s, char delim);

void data(string filename, unordered_map<int, unordered_map<int, double> >& network){
	ifstream ifs(filename);
	string sample;
	char delim = ' ';

	// firm i's out-degree is stored. Kigyo ID starts from 1
	for(int i = 1; i < (network_size + 1); ++i){
		network[i];
	}

	while(getline(ifs, sample) ){
		vector<string> sample_split = split(sample, delim);
		vector<double> test_vec(3);
		for(int j = 0; j < 3; ++j) test_vec[j] = stod(sample_split[j]);
		network[test_vec[0]][test_vec[1]] = test_vec[2];
	}
}

void data_b(string filename, unordered_map<int, unordered_map<int, double> >& network){
	ifstream ifs(filename);
	string sample;
	char delim = ' ';

	// firm i's in-degree is stored. Kigyo ID starts from 1
	for(int i = 1; i < (network_size + 1); ++i){
		network[i];
	}

	while(getline(ifs, sample) ){
		vector<string> sample_split = split(sample, delim);
		vector<double> test_vec(3);
		for(int j = 0; j < 3; ++j) test_vec[j] = stod(sample_split[j]);
		network[test_vec[1]][test_vec[0]] = test_vec[2];
	}
}

void data_KJ(string filename, unordered_map<int, unordered_map<int, double> >& kigyo_joho){
	ifstream ifs(filename);
	string sample;
	char delim = ' ';

	int i = 1;// Kigyo ID starts from 1

	while(getline(ifs, sample) ){
		vector<string> sample_split = split(sample, delim);
		for(int j = 0; j < n_item; ++j) {
			double temp = stod(sample_split[j]);
			kigyo_joho[i][j] = temp;
		}
		i += 1;
	}
}

void data_macro(string filename, unordered_map<int, unordered_map<int, double> >& macrovar){
	ifstream ifs(filename);
	string sample;
	char delim = ' ';
	int num_macro = 3;

	int t = 1;//time ranges from 1 to 970

	// 1 is GDP quarterly and 2 is IIP, 3 Call rate
	while(getline(ifs, sample) ){
		vector<string> sample_split = split(sample, delim);
		for(int j = 0; j < num_macro; ++j) {
			double temp = stod(sample_split[j]);
			macrovar[t][j] = temp;
		}
		t += 1;
	}
}

vector<string> split(string &s, char delim){
    vector<string> elems;
    stringstream ss(s);
    string item;
    while (getline(ss, item, delim)) {
    if (!item.empty()) {
            elems.push_back(item);
        }
    }
    return elems;
}

#endif /* DATA_H_ */
