#include "MLP_Functions.h"
#include <iostream>
#include <random>
using namespace std;

double sigmoid(double x) { // Activation Function
	return 1 / (1 + exp(-x));
}

double deSigmoid(double x) { //Sigmoid func 미분
	return x*(1 - x);
}

class Layer { // 기본 레이어

public:
	int nodeNum, preNodeNum;
	vector<double> value;		//z값
	vector<double> activeValue;	//activation func(z)
	vector<vector<double>> weight; // 가중치(이전 노드 -> 다음노드를 가르키기에 2차원)
	vector<double> preDel;	// backpropagation시 이전 단계에서의 delta값 저장하여 연산을 줄여주기 위한 변수
	double learningRate = 0.0001;	// 학습률
	double bias;	//편향(각 레이어에서 가장 첫번째 weight로 설정)


	Layer(int pre, int next) :nodeNum(next), preNodeNum(pre) {//변수 사이즈 설정
		value.resize(nodeNum);
		activeValue.resize(nodeNum);
		weight.resize(preNodeNum, vector<double>(nodeNum));
		preDel.resize(preNodeNum);
	}

	void init() { //weight 및 bias 초기화
		random_device rd;
		mt19937 gen(rd());
		uniform_real_distribution<double> unif(-1, 1);
		for (int i = 0; i < preNodeNum; i++) {
			for (int j = 0; j < nodeNum; j++) {
				weight[i][j] = unif(gen);

			}
		}
		bias = weight[0][0];
	}

	void feedForward(vector<double> preLayer) {//순전파 과정
		for (int i = 0; i < nodeNum; i++) {
			value[i] = 0;
			activeValue[i] = 0;
			for (int j = 0; j < preNodeNum; j++) {
				value[i] += (weight[j][i] * preLayer[j] + bias);//이전 레이어의 노드의 활성화값에 가중치를 곱하고 편향을 더함
			}
			activeValue[i] = sigmoid(value[i]);//이전 레이어 노드에서 구한 값의 활성화값 산출
		}


	}

	virtual void backPropagation(vector<double> preDelta, vector<double> active, vector<vector<double>> weight) {
		//다음 레이어에서 얻은 변화값, 이전 레이어의 활성화 값, 이전 레이어와 현재 레이어에 연결돼있는 가중치 값 
		double delta = 0;
		for (int i = 0; i < nodeNum; i++) {

			delta = preDelta[i] * deSigmoid(activeValue[i]);

			for (int j = 0; j < preNodeNum; j++) {
				//cout << weight[j][i] << endl;;
				preDel[j] += delta*weight[j][i];
				weight[j][i] -= learningRate*delta*active[j];

			}
			bias -= learningRate*delta;
			preDelta[i] = 0;
		}


	}

};

class OutputLayer : public Layer { //output레이어에서는 loss값을 구해야하고, 역전파가 다른 레이어에서와 조금 다르게 전개됨. 
public:

	OutputLayer(int pre, int next) : Layer(pre, next) {}

	double error(double real) {

		return 0.5*pow((real - activeValue[0]), 2);
	}

	void backPropagation(double realVal, vector<double> active, vector<vector<double>> weight) {//realVal은 실제 얻어야하는 target값
		double delta = 0;
		delta = (activeValue[0] - realVal)*deSigmoid(activeValue[0]);
		for (int i = 0; i < preNodeNum; i++) {

			preDel[i] = delta*weight[i][0];
			weight[i][0] -= learningRate*delta*active[i];

		}
		bias -= learningRate*delta;
	}
};

int getNodeNum(int preNodeNum) {//현재 노드 수보다 작은 2의 제곱수 중에 가장 큰 값을 다음 레이어의 노드 수로 설정
	for (int i = 0; i < preNodeNum; i++) {
		if (pow(2, i) >= preNodeNum) {
			return pow(2, (i - 1));
		}
	}
	return -1;
}

vector<vector<double>> minMaxScaler(vector<vector<double>> x) {
	int row_num = x.size();
	int col_num = x[0].size();
	vector <double> max_val, min_val;
	vector<double> v;
	double min, max;
	for (int i = 0; i < col_num; i++) {
		min = x[0][i];  max = x[0][i];
		for (int j = 0; j < row_num; j++) {
			if (x[j][i] < min) {
				min = x[j][i];

			}
			else if (x[j][i] > max) {
				max = x[j][i];
			}
		}
		min_val.push_back(min);
		max_val.push_back(max);
	}

	for (int i = 0; i < col_num; i++) {
		min = min_val[i];
		max = max_val[i];
		for (int j = 0; j < row_num; j++) {
			x[j][i] = (x[j][i] - min) / (max - min);
		}
	}
	return x;
}

int main() {
	//Regression 
	//Data Load
	string dataPath = "C:/Users/ecminer/Documents/Visual Studio 2015/Projects/ConsoleApplication6/ConsoleApplication6/ProcessDifference_train.csv";
	const char* NameofData = dataPath.c_str();
	vector<vector<double>> train;
	train = readFile(NameofData);
	int epoch = 50;

	//x_train/Y_train Split
	vector<vector<double>> x_train, y_train;
	splitData(train, x_train, y_train);
	x_train = minMaxScaler(x_train);
	y_train = minMaxScaler(y_train);

	//레이어 선언 및 생성
	Layer hidden1(x_train[0].size(), getNodeNum(x_train[0].size())), hidden2(hidden1.nodeNum, getNodeNum(hidden1.nodeNum)), hidden3(hidden2.nodeNum, getNodeNum(hidden2.nodeNum));
	OutputLayer output(hidden3.nodeNum, 1);
	hidden1.init();
	hidden2.init();
	hidden3.init();
	output.init();

	//학습 진행.
	for (int i = 0; i <epoch; i++) {
		double error = 0;
		for (int j = 0; j < x_train.size(); j++) {

			hidden1.feedForward(x_train[j]);
			hidden2.feedForward(hidden1.activeValue);
			hidden3.feedForward(hidden2.activeValue);
			output.feedForward(hidden3.activeValue);

			error = output.error(y_train[j][0]);

			output.backPropagation(y_train[j][0], hidden3.activeValue, output.weight);
			hidden3.backPropagation(output.preDel, hidden2.activeValue, hidden3.weight);
			hidden2.backPropagation(hidden3.preDel, hidden1.activeValue, hidden2.weight);
			hidden1.backPropagation(hidden2.preDel, x_train[j], hidden1.weight);

		}
		cout << i + 1 << "  epoch,     " << "loss: " << error << endl;
	}


	//test수행
	string testPath = "C:/Users/ecminer/Documents/Visual Studio 2015/Projects/ConsoleApplication6/ConsoleApplication6/ProcessDifference_test.csv";
	const char* TestData = testPath.c_str();
	vector<vector<double>> test;
	test = readFile(TestData);

	vector<vector<double>> x_test, y_test;
	splitData(test, x_test, y_test);
	x_test = minMaxScaler(x_test);
	y_test = minMaxScaler(y_test);

	for (int j = 0; j < x_test.size(); j++) {
		double error = 0;
		hidden1.feedForward(x_test[j]);
		hidden2.feedForward(hidden1.activeValue);
		hidden3.feedForward(hidden2.activeValue);
		output.feedForward(hidden3.activeValue);
		error = output.error(y_test[j][0]);
		cout << "test loss: " << error << endl;
	}

	system("pause");


	return 0;
}

