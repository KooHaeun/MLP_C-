#include "MLP_Functions.h"
#include <iostream>
#include <random>
using namespace std;

double sigmoid(double x) { // Activation Function
	return 1 / (1 + exp(-x));
}

double deSigmoid(double x) { //Sigmoid func �̺�
	return x*(1 - x);
}

class Layer { // �⺻ ���̾�

public:
	int nodeNum, preNodeNum;
	vector<double> value;		//z��
	vector<double> activeValue;	//activation func(z)
	vector<vector<double>> weight; // ����ġ(���� ��� -> ������带 ����Ű�⿡ 2����)
	vector<double> preDel;	// backpropagation�� ���� �ܰ迡���� delta�� �����Ͽ� ������ �ٿ��ֱ� ���� ����
	double learningRate = 0.0001;	// �н���
	double bias;	//����(�� ���̾�� ���� ù��° weight�� ����)


	Layer(int pre, int next) :nodeNum(next), preNodeNum(pre) {//���� ������ ����
		value.resize(nodeNum);
		activeValue.resize(nodeNum);
		weight.resize(preNodeNum, vector<double>(nodeNum));
		preDel.resize(preNodeNum);
	}

	void init() { //weight �� bias �ʱ�ȭ
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

	void feedForward(vector<double> preLayer) {//������ ����
		for (int i = 0; i < nodeNum; i++) {
			value[i] = 0;
			activeValue[i] = 0;
			for (int j = 0; j < preNodeNum; j++) {
				value[i] += (weight[j][i] * preLayer[j] + bias);//���� ���̾��� ����� Ȱ��ȭ���� ����ġ�� ���ϰ� ������ ����
			}
			activeValue[i] = sigmoid(value[i]);//���� ���̾� ��忡�� ���� ���� Ȱ��ȭ�� ����
		}


	}

	virtual void backPropagation(vector<double> preDelta, vector<double> active, vector<vector<double>> weight) {
		//���� ���̾�� ���� ��ȭ��, ���� ���̾��� Ȱ��ȭ ��, ���� ���̾�� ���� ���̾ ������ִ� ����ġ �� 
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

class OutputLayer : public Layer { //output���̾���� loss���� ���ؾ��ϰ�, �����İ� �ٸ� ���̾���� ���� �ٸ��� ������. 
public:

	OutputLayer(int pre, int next) : Layer(pre, next) {}

	double error(double real) {

		return 0.5*pow((real - activeValue[0]), 2);
	}

	void backPropagation(double realVal, vector<double> active, vector<vector<double>> weight) {//realVal�� ���� �����ϴ� target��
		double delta = 0;
		delta = (activeValue[0] - realVal)*deSigmoid(activeValue[0]);
		for (int i = 0; i < preNodeNum; i++) {

			preDel[i] = delta*weight[i][0];
			weight[i][0] -= learningRate*delta*active[i];

		}
		bias -= learningRate*delta;
	}
};

int getNodeNum(int preNodeNum) {//���� ��� ������ ���� 2�� ������ �߿� ���� ū ���� ���� ���̾��� ��� ���� ����
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

	//���̾� ���� �� ����
	Layer hidden1(x_train[0].size(), getNodeNum(x_train[0].size())), hidden2(hidden1.nodeNum, getNodeNum(hidden1.nodeNum)), hidden3(hidden2.nodeNum, getNodeNum(hidden2.nodeNum));
	OutputLayer output(hidden3.nodeNum, 1);
	hidden1.init();
	hidden2.init();
	hidden3.init();
	output.init();

	//�н� ����.
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


	//test����
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

