#ifndef NEURALNETWORKRESULT_H
#define NEURALNETWORKRESULT_H

#include <bits/stdc++.h>
#include "json.hpp"
#include "eigen3/Eigen/Core"

using json = nlohmann::json;

typedef float Scalar;

struct NeuralNetworkResult {
private:
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
    typedef Eigen::Array<Scalar, 1, Eigen::Dynamic> RowArray;
    typedef Eigen::RowVectorXi IntegerVector;
    float sigmoid(float X);

public:

    float prob;
    float sum = 0;
    int maxI = 0;
    float max = 0;
    int size = 0;
    int inputLayerSize;
    int hiddenLayerSize;
    int outLayerSize;
    json weights;
    // Construct a network object
    // Network net;
    Matrix W1;
    Vector b1;
    Matrix W2;
    Vector b2;
    Matrix W3;
    Vector b3;

    Matrix Z1;
    Matrix Z2;
    Matrix Z3;

    Matrix out1;
    Matrix out2;
    Matrix out3;

    int batch_size = 1;
    int mini_batch_size = 16;

    // Derivatives
    Matrix dW1;
    Vector db1;
    Matrix dW2;
    Vector db2;
    Matrix dW3;
    Vector db3;

    Matrix dout1;
    Matrix dout2;
    Matrix dout3;

    Matrix dLz3;
    Matrix dLz2;
    Matrix dLz1;

    float result = 0;
    //constructor - destructor
    NeuralNetworkResult();
    NeuralNetworkResult(json model);
    NeuralNetworkResult(int input_size, int hidden_size, int output_size);

    ~NeuralNetworkResult();

    //function declarations for prediction
    double predictRegression(float *X);
    int classifyMC(float *X);
    std::vector<float> classifyML(float *X);

    // functions for model update (forward / backward / param_update)
    void forwardRegression(float *X);
    void forwardRegression(Matrix &X);
    void backwardRegression(float *X, float *y);
    void backwardRegression(Matrix &X, Matrix &y);
    void forwardClassifyMC(Matrix &X);
    void backwardClassifyMC(Matrix &X, Matrix &y);
    void backwardClassifyMC(Matrix &X, IntegerVector &y);
    void updateSGD(float lr, float lambda);
    void checkGradient(Matrix &X, Matrix &y, std::string mode);
    void checkGradient(Matrix &X, IntegerVector &y, std::string mode);

    Scalar MSELoss(Matrix &y_pred, Matrix& y_target);
    Scalar MultiClassEntropy(Matrix &y_pred, IntegerVector& y_target);

    void initModelUpdate();
    void resizeParamBatchsize(int size);
    void getMinibatchData(float* X, float* y, std::vector<double> *queryInput, std::vector<double> *queryOutput, int iter);
    void updateModel(std::vector<Matrix> *queryInput, std::vector<Matrix> *queryOutput, int epochs);
    void updateModel(std::vector<Matrix> *queryInput, std::vector<IntegerVector> *queryOutput, int epochs);
    void checkModelGradient(std::vector<Matrix> *queryInput, std::vector<Matrix> *queryOutput, int epochs, std::string mode);
    void checkModelGradient(std::vector<Matrix> *queryInput, std::vector<IntegerVector> *queryOutput, int epochs, std::string mode);
};



#endif //NEURALNETWORKRESULT_H
