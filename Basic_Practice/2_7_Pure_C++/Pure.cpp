#include "Pure.h"
#include "json.hpp"
#include <tuple>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <iostream>
#include <fstream>

using namespace std;
using namespace std::chrono;
using json = nlohmann::json;

NeuralNetworkResult::NeuralNetworkResult() {}

NeuralNetworkResult::NeuralNetworkResult(json model) {
    sum = 0;
    maxI = 0;
    max = 0;
    inputLayerSize = model["weights"]["W0"].size();
    hiddenLayerSize = model["weights"]["W2"].size();
    outLayerSize = model["weights"]["W5"].size();
    weights = model["weights"];

    // Initialize matrices and then randomize the weights (to normal distribution) at the same time

    // Initialize matrices
    W1.resize(inputLayerSize, hiddenLayerSize);
    size = sizeof(Scalar) * inputLayerSize * hiddenLayerSize;

    b1.resize(hiddenLayerSize);
    size += sizeof(Scalar) * hiddenLayerSize;

    W2.resize(hiddenLayerSize, hiddenLayerSize);
    size += sizeof(Scalar) * hiddenLayerSize * hiddenLayerSize;

    b2.resize(hiddenLayerSize);
    size += sizeof(Scalar) * hiddenLayerSize;

    W3.resize(hiddenLayerSize, outLayerSize);
    size += sizeof(Scalar) * outLayerSize * hiddenLayerSize;

    b3.resize(outLayerSize);
    size += sizeof(Scalar) * outLayerSize;

    // Read weights
    for (auto w = weights.begin(); w != weights.end(); ++w){
        if (w.key() == "W0"){
            for (int i=0; i < inputLayerSize; i++){
                for (int j=0; j < hiddenLayerSize; j++){
                    W1(i,j) = (w.value()[i][j]);
                }
            }
        }
        if (w.key() == "W1"){
            for (int i=0; i < hiddenLayerSize; i++)
                b1(i) = w.value()[i];
        }

        if (w.key() == "W2") {
            for (int i=0; i < hiddenLayerSize; i++){
                for (int j=0; j < hiddenLayerSize; j++){
                    W2(i,j) = w.value()[i][j];
                }
            }
        }
        if (w.key() == "W3"){
            for (int i=0; i < hiddenLayerSize; i++)
                b2(i) = w.value()[i];
        }

        if (w.key() == "W4") {
            for (int i=0; i < hiddenLayerSize; i++){
                for (int j=0; j < outLayerSize; j++){
                    W3(i,j) = w.value()[i][j];
                }
            }
        }
        if (w.key() == "W5") {
            for (int i = 0; i < outLayerSize; i++)
                b3(i) = w.value()[i];
        }
    }

    Z1.resize(hiddenLayerSize, batch_size);
    Z2.resize(hiddenLayerSize, batch_size);
    Z3.resize(outLayerSize, batch_size);

    out1.resize(hiddenLayerSize, batch_size);
    out2.resize(hiddenLayerSize, batch_size);
    out3.resize(outLayerSize, batch_size);

    size += sizeof(Scalar) * hiddenLayerSize * 4;
    size += sizeof(Scalar) * outLayerSize * 2;
}

NeuralNetworkResult::NeuralNetworkResult(int inputLayerSize, int hiddenLayerSize, int outLayerSize) {
    sum = 0;
    maxI = 0;
    max = 0;

    this->inputLayerSize = inputLayerSize;
    this->hiddenLayerSize = hiddenLayerSize;
    this->outLayerSize = outLayerSize;
    // Initialize forward path variables
    // Initialize matrices and then randomize the weights (to normal distribution) at the same time
    W1.resize(inputLayerSize, hiddenLayerSize);
    W1.setRandom(W1.rows(), W1.cols());
    dW1.resize(inputLayerSize, hiddenLayerSize);
    size = sizeof(Scalar) * inputLayerSize * hiddenLayerSize * 2;

    b1.resize(hiddenLayerSize);
    b1.setZero();
    db1.resize(hiddenLayerSize);
    size += sizeof(Scalar) * hiddenLayerSize * 2;

    W2.resize(hiddenLayerSize, hiddenLayerSize);
    W2.setRandom(W2.rows(), W2.cols());
    dW2.resize(hiddenLayerSize, hiddenLayerSize);
    size += sizeof(Scalar) * hiddenLayerSize * hiddenLayerSize * 2;

    b2.resize(hiddenLayerSize);
    b2.setZero();
    db2.resize(hiddenLayerSize);
    size += sizeof(Scalar) * hiddenLayerSize * 2;

    W3.resize(hiddenLayerSize, outLayerSize);
    W3.setRandom(W3.rows(), W3.cols());
    dW3.resize(hiddenLayerSize, outLayerSize);
    size += sizeof(Scalar) * outLayerSize * hiddenLayerSize * 2;

    b3.resize(outLayerSize);
    b3.setZero();
    db3.resize(outLayerSize);
    size += sizeof(Scalar) * outLayerSize * 2;

    Z1.resize(hiddenLayerSize, mini_batch_size);
    dLz1.resize(hiddenLayerSize, mini_batch_size);
    Z2.resize(hiddenLayerSize, mini_batch_size);
    dLz2.resize(hiddenLayerSize, mini_batch_size);
    Z3.resize(outLayerSize, mini_batch_size);
    dLz3.resize(outLayerSize, mini_batch_size);

    out1.resize(hiddenLayerSize, mini_batch_size);
    dout1.resize(hiddenLayerSize, mini_batch_size);
    out2.resize(hiddenLayerSize, mini_batch_size);
    dout2.resize(hiddenLayerSize, mini_batch_size);
    out3.resize(outLayerSize, mini_batch_size);
    dout3.resize(outLayerSize, mini_batch_size);

    size += sizeof(Scalar) * hiddenLayerSize * 4 * 2;
    size += sizeof(Scalar) * outLayerSize * 2 * 2;

//    cout << "W1: " << W1 << endl;
//    cout << "b1: " << b1 << endl;
//    cout << "W2: " << W2 << endl;
//    cout << "b2: " << b2 << endl;
//    cout << "W3: " << W3 << endl;
//    cout << "b3: " << b3 << endl;
}

NeuralNetworkResult::~NeuralNetworkResult() {}

// input is a vector with size of inputLayerSize
double NeuralNetworkResult::predictRegression(float *X){
    // copy X to x_in in Eigen matrix format
    Eigen::MatrixXf x_in = Eigen::Map<Eigen::MatrixXf>(X, inputLayerSize, batch_size);

    // 1st layer
    Z1.noalias() = W1.transpose() * x_in; // z = w' * x + b
    Z1.colwise() += b1;
    out1.array() = Z1.array().cwiseMax(float(0)); // Relu

    // 2nd layer
    Z2.noalias() = W2.transpose() * out1;
    Z2.colwise() += b2;
    out2.array() = Z2.array().cwiseMax(float(0)); // Relu

    // Output layer
    out3.noalias() = W3.transpose() * out2;
    out3.colwise() += b3;

    return double(out3(0,0));
}


int NeuralNetworkResult::classifyMC(float *X) {

    // copy X to x_in in Eigen matrix format
    Eigen::MatrixXf x_in = Eigen::Map<Eigen::MatrixXf>(X, inputLayerSize, batch_size);

    // 1st layer
    Z1.noalias() = W1.transpose() * x_in; // z = w' * x + b
    Z1.colwise() += b1;
    out1.array() = Z1.array().cwiseMax(float(0)); // Relu

    // 2nd layer
    Z2.noalias() = W2.transpose() * out1;
    Z2.colwise() += b2;
    out2.array() = Z2.array().cwiseMax(float(0)); // Relu

    // Output layer
    Z3.noalias() = W3.transpose() * out2;
    Z3.colwise() += b3;
    // Softmax output
    out3.array() = (Z3.rowwise() - Z3.colwise().maxCoeff()).array().exp();
    RowArray colsums = out3.colwise().sum();
    out3.array().rowwise() /= colsums;

    // Copy the data to a long vector
    std::vector<float> res(out3.size());
    std::copy(out3.data(), out3.data() + out3.size(), res.begin());

    int argMax = std::distance(res.begin(), std::max_element(res.begin(), res.end()));

    return argMax;
}


std::vector<float> NeuralNetworkResult::classifyML(float *X) {

    // copy X to x_in in Eigen matrix format
    Eigen::MatrixXf x_in = Eigen::Map<Eigen::MatrixXf>(X, inputLayerSize, batch_size);

    // 1st layer
    Z1.noalias() = W1.transpose() * x_in; // z = w' * x + b
    Z1.colwise() += b1;
    out1.array() = Z1.array().cwiseMax(float(0)); // Relu

    // 2nd layer
    Z2.noalias() = W2.transpose() * out1;
    Z2.colwise() += b2;
    out2.array() = Z2.array().cwiseMax(float(0)); // Relu

    // Output layer
    Z3.noalias() = W3.transpose() * out2;
    Z3.colwise() += b3;
    // multiple sigmoid output
    out3.array() = Scalar(1) / (Scalar(1) + (-Z3.array()).exp());

    // Copy the data to a long vector
    std::vector<float> res(out3.size());
    std::copy(out3.data(), out3.data() + out3.size(), res.begin());
    // int argMax = std::distance(res.begin(), std::max_element(res.begin(), res.end()));
    return res;
}


/////// Fuctions for Model Updates
// forward path for regression model
void NeuralNetworkResult::forwardRegression(float *X){
    // copy X to x_in in Eigen matrix format
    std::cout << X[0] << "  " << X[1] << "  " << X[2] << std::endl;
    Eigen::MatrixXf x_in = Eigen::Map<Eigen::MatrixXf>(X, inputLayerSize, mini_batch_size);
    std::cout << "FORWARD: mapping x_in size: " << x_in.size() << std::endl;
    // 1st layer
    Z1.noalias() = W1.transpose() * x_in; // z = w' * x + b
    Z1.colwise() += b1;
    out1.array() = Z1.array().cwiseMax(float(0)); // Relu

    // 2nd layer
    Z2.noalias() = W2.transpose() * out1;
    Z2.colwise() += b2;
    out2.array() = Z2.array().cwiseMax(float(0)); // Relu

    // Output layer
    out3.noalias() = W3.transpose() * out2;
    out3.colwise() += b3;
}

void NeuralNetworkResult::forwardRegression(Matrix &x_in){
    // 1st layer
    Z1.noalias() = W1.transpose() * x_in; // z = w' * x + b
    Z1.colwise() += b1;
    out1.array() = Z1.array().cwiseMax(float(0)); // Relu

    // 2nd layer
    Z2.noalias() = W2.transpose() * out1;
    Z2.colwise() += b2;
    out2.array() = Z2.array().cwiseMax(float(0)); // Relu

    // Output layer
    out3.noalias() = W3.transpose() * out2;
    out3.colwise() += b3;
}

// backward path for regression model
void NeuralNetworkResult::backwardRegression(float *X, float *y){
    // copy X, y to x_in, y_target in Eigen matrix format
    Eigen::MatrixXf x_in = Eigen::Map<Eigen::MatrixXf>(X, inputLayerSize, mini_batch_size);
    Eigen::MatrixXf y_target = Eigen::Map<Eigen::MatrixXf>(y, outLayerSize, mini_batch_size);

    // Compute the derivative of the input of this layer
    // L = 0.5 * ||yhat - y||^2
    // in = yhat
    // d(L) / d(in) = yhat - y
    // Output layer: [out(=1) x mini_bs]
    dout3.noalias() = (out3 - y_target)*2;

    // 2nd layer
    // derivative for weights: [hidden x output(=1)] = [hidden x mini_bs] * [mini_bs x output(=1)]
    // d(L) / d(W) = [d(L) / d(z)] * a = a2 * delta3^T
    dW3.noalias() = out2 * dout3.transpose() / mini_batch_size;
    // derivative for bias: [h x 1] = [h x mini_bs].mean()
    // d(L) / d(b) = delta
    db3.noalias() = dout3.rowwise().mean();
    // Compute delta2: [h x mini_bs] = [h x out(=1)] * [out(=1) x mini_bs]
    // d(L) / d_in = W * [d(L) / d(z)]
    // compute jacobian d(L) / d(z2)
    dLz2.noalias() = W3 * dout3;
    dout2.array() = (out2.array() > Scalar(0)).select(dLz2, Scalar(0));

    // 1st layer
    dW2.noalias() = out1 * dout2.transpose() / mini_batch_size;
    db2.noalias() = dout2.rowwise().mean();
    dLz1.noalias() = W2 * dout2;
    dout1.array() = (out1.array() > Scalar(0)).select(dLz2, Scalar(0));

    // input layer
    dW1.noalias() = x_in * dout1.transpose() / mini_batch_size;
    db1.noalias() = dout1.rowwise().mean();
}

void NeuralNetworkResult::backwardRegression(Matrix &x_in, Matrix &y_target){
    dout3.noalias() = (out3 - y_target)*2;

    // 2nd layer
    // derivative for weights: [hidden x output(=1)] = [hidden x mini_bs] * [mini_bs x output(=1)]
    // d(L) / d(W) = [d(L) / d(z)] * a = a2 * delta3^T
    dW3.noalias() = out2 * dout3.transpose() / mini_batch_size;
    // derivative for bias: [h x 1] = [h x mini_bs].mean()
    // d(L) / d(b) = delta
    db3.noalias() = dout3.rowwise().mean();
    // Compute delta2: [h x mini_bs] = [h x out(=1)] * [out(=1) x mini_bs]
    // d(L) / d_in = W * [d(L) / d(z)]
    // compute jacobian d(L) / d(z2)
    dLz2.noalias() = W3 * dout3;
    dout2.array() = (out2.array() > Scalar(0)).select(dLz2, Scalar(0));

    // 1st layer
    dW2.noalias() = out1 * dout2.transpose() / mini_batch_size;
    db2.noalias() = dout2.rowwise().mean();
    dLz1.noalias() = W2 * dout2;
    dout1.array() = (out1.array() > Scalar(0)).select(dLz1, Scalar(0));

    // input layer
    dW1.noalias() = x_in * dout1.transpose() / mini_batch_size;
    db1.noalias() = dout1.rowwise().mean();
}

////// forward and backward path for classification (softmax output)
void NeuralNetworkResult::forwardClassifyMC(Matrix &x_in) {
    // 1st layer
    Z1.noalias() = W1.transpose() * x_in; // z = w' * x + b
    Z1.colwise() += b1;
    out1.array() = Z1.array().cwiseMax(float(0)); // Relu

    // 2nd layer
    Z2.noalias() = W2.transpose() * out1;
    Z2.colwise() += b2;
    out2.array() = Z2.array().cwiseMax(float(0)); // Relu

    // Output layer
    Z3.noalias() = W3.transpose() * out2;
    Z3.colwise() += b3;
    // Softmax output
    out3.array() = (Z3.rowwise() - Z3.colwise().maxCoeff()).array().exp();
    RowArray colsums = out3.colwise().sum();
    out3.array().rowwise() /= colsums;
}

void NeuralNetworkResult::backwardClassifyMC(Matrix &x_in, Matrix &y_target) {
    dout3.noalias() = out3 - y_target;

    // 2nd layer
    // derivative for weights: [hidden x output(=1)] = [hidden x mini_bs] * [mini_bs x output(=1)]
    dW3.noalias() = out2 * dout3.transpose() / mini_batch_size;
    // derivative for bias: [h x 1] = [h x mini_bs].mean()
    db3.noalias() = dout3.rowwise().mean();
    // Compute delta2: [h x mini_bs] = [h x out(=1)] * [out(=1) x mini_bs]
    dLz2.noalias() = W3 * dout3;
    dout2.array() = (out2.array() > Scalar(0)).select(dLz2, Scalar(0));

    // 1st layer
    dW2.noalias() = out1 * dout2.transpose() / mini_batch_size;
    db2.noalias() = dout2.rowwise().mean();
    dLz1.noalias() = W2 * dout2;
    dout1.array() = (out1.array() > Scalar(0)).select(dLz1, Scalar(0));

    // input layer
    dW1.noalias() = x_in * dout1.transpose() / mini_batch_size;
    db1.noalias() = dout1.rowwise().mean();
}

void NeuralNetworkResult::backwardClassifyMC(Matrix &x_in, IntegerVector &y_target) {
    dout3.noalias() = out3;
    for (int i = 0; i < mini_batch_size; i++){
        dout3(y_target(i), i) -= 1;
    }

    // 2nd layer
    // derivative for weights: [hidden x output(=1)] = [hidden x mini_bs] * [mini_bs x output(=1)]
    dW3.noalias() = out2 * dout3.transpose() / mini_batch_size;
    // derivative for bias: [h x 1] = [h x mini_bs].mean()
    db3.noalias() = dout3.rowwise().mean();
    // Compute delta2: [h x mini_bs] = [h x out(=1)] * [out(=1) x mini_bs]
    dLz2.noalias() = W3 * dout3;
    dout2.array() = (out2.array() > Scalar(0)).select(dLz2, Scalar(0));

    // 1st layer
    dW2.noalias() = out1 * dout2.transpose() / mini_batch_size;
    db2.noalias() = dout2.rowwise().mean();
    dLz1.noalias() = W2 * dout2;
    dout1.array() = (out1.array() > Scalar(0)).select(dLz1, Scalar(0));

    // input layer
    dW1.noalias() = x_in * dout1.transpose() / mini_batch_size;
    db1.noalias() = dout1.rowwise().mean();
}

// gradient update function for regression and classification model
void NeuralNetworkResult::updateSGD(float lr=0.001, float lambda=0.0) {

    W3.noalias() -= lr * (dW3 + lambda * W3);
    b3.noalias() -= lr * (db3 + lambda * b3);

    W2.noalias() -= lr * (dW2 + lambda * W2);
    b2.noalias() -= lr * (db2 + lambda * b2);

    W1.noalias() -= lr * (dW1 + lambda * W1);
    b1.noalias() -= lr * (db1 + lambda * b1);
}


void NeuralNetworkResult::initModelUpdate(){
    // Initialize matrices and then randomize the weights (to normal distribution) at the same time
    dW1.resize(inputLayerSize, hiddenLayerSize);
    db1.resize(hiddenLayerSize);

    dW2.resize(hiddenLayerSize, hiddenLayerSize);
    db2.resize(hiddenLayerSize);

    dW3.resize(hiddenLayerSize, outLayerSize);
    db3.resize(outLayerSize);

    dLz1.resize(hiddenLayerSize, mini_batch_size);
    dLz2.resize(hiddenLayerSize, mini_batch_size);
    dLz3.resize(outLayerSize, mini_batch_size);

    dout1.resize(hiddenLayerSize, mini_batch_size);
    dout2.resize(hiddenLayerSize, mini_batch_size);
    dout3.resize(outLayerSize, mini_batch_size);
}

void NeuralNetworkResult::resizeParamBatchsize(int size){

//    cout << "in NeuralNetworkResult::resizeParamBatchsize" << endl;
//    cout << "Z1: " << Z1 << endl;
//    cout << "out3: " << out3(0,0) << endl;
//    cout << "inputLayerSize: " << inputLayerSize << endl;
//    cout << "hiddenLayerSize: " << hiddenLayerSize << endl;
//    cout << "outLayerSize: " << outLayerSize << endl;

    Z1.resize(hiddenLayerSize, size);
    dLz1.resize(hiddenLayerSize, size);
    Z2.resize(hiddenLayerSize, size);
    dLz2.resize(hiddenLayerSize, size);
    Z3.resize(outLayerSize, size);
    dLz3.resize(outLayerSize, size);

    out1.resize(hiddenLayerSize, size);
    dout1.resize(hiddenLayerSize, size);
    out2.resize(hiddenLayerSize, size);
    dout2.resize(hiddenLayerSize, size);
    out3.resize(outLayerSize, size);
    dout3.resize(outLayerSize, size);

}

void NeuralNetworkResult::getMinibatchData(float* X, float* y, std::vector<double> *queryInput, std::vector<double> *queryOutput, int iter){
    for (int i = 0; i < mini_batch_size; i++){
        y[i] = (*queryOutput)[iter+i];
        X[2*i] = (*queryInput)[2*(iter+i)];
        X[2*i+1] = (*queryInput)[2*(iter+i) + 1];
    }
}

void NeuralNetworkResult::updateModel(std::vector<Matrix> *queryInput, std::vector<Matrix> *queryOutput, int epochs){

    resizeParamBatchsize(mini_batch_size);
    std::vector<float> loss_vec;
    for (int epoch = 0; epoch < epochs; epoch++){ // per epoch
        int iterations = queryOutput->size();
        for (int iter = 0; iter < iterations; iter++){ // per batch
            // train / update the model
            forwardRegression((*queryInput)[iter]);
            backwardRegression((*queryInput)[iter],(*queryOutput)[iter]);
            if (iter % 100 == 0) {
//                cout << "iter: " << iter << endl;
//                cout << "queryInput: " << (*queryInput)[iter] << endl;
//                cout << "queryOutput: " << (*queryOutput)[iter] << endl;
//                cout << "Loss: " << MSELoss(out3, (*queryOutput)[iter]) << endl;
//                cout << "out3: " << out3 << endl;

                loss_vec.push_back(MSELoss(out3, (*queryOutput)[iter]));
                // cout << MSELoss(out3, (*queryOutput)[iter]) << endl;
            }
            updateSGD();
        }
        // print out some results for each epoch
//        float train_loss = 0.0;
//        for (int i = 0; i < loss_vec.size(); i++) {
//            train_loss += loss_vec[i];
//            // std::cout << "Loss: " << loss_vec[i] << std::endl;
//        }
//        if (loss_vec.size() == 0) {
//            for(auto val: loss_vec) cout << val << endl;
//            printf("==> Epoch: %d, Iteration: %d, Loss: %f \n", epoch+1, (epoch+1)*iterations, MSELoss(out3, (*queryOutput)[iterations-1]));
//        }
//        else {
//            for(auto val: loss_vec) cout << val << endl;
//            printf("==> Epoch: %d, Iteration: %d, Loss: %f \n", epoch+1, (epoch+1)*iterations, train_loss/loss_vec.size());
//
//        }
//        loss_vec.clear();
    }
}

void NeuralNetworkResult::updateModel(std::vector<Matrix> *queryInput, std::vector<IntegerVector> *queryOutput, int epochs){

    resizeParamBatchsize(mini_batch_size);
    std::vector<float> loss_vec;
    for (int epoch = 0; epoch < epochs; epoch++){ // per epoch
        int iterations = queryOutput->size() ;
        for (int iter = 0; iter < iterations; iter++){ // per batch
            // train / update the model
            forwardClassifyMC((*queryInput)[iter]);
            backwardClassifyMC((*queryInput)[iter],(*queryOutput)[iter]);
            if (iter % 100 == 0) {
                loss_vec.push_back(MultiClassEntropy(out3, (*queryOutput)[iter]));
            }
            updateSGD(0.001, 0.0);
        }
        // print out some results for each epoch
        float train_loss = 0.0;
        for (int i = 0; i < loss_vec.size(); i++) {
            train_loss += loss_vec[i];
            // std::cout << "Loss: " << loss_vec[i] << std::endl;
        }
        printf("==> Epoch: %d, Iteration: %d, Loss: %f \n", epoch+1, (epoch+1)*iterations, train_loss/loss_vec.size());
        loss_vec.clear();
    }
}


void NeuralNetworkResult::checkModelGradient(std::vector<Matrix> *queryInput, std::vector<Matrix> *queryOutput, int epochs, std::string mode){
    mini_batch_size = 1;
    std::cout << "inside checkModelGradient" << std::endl;
    initModelUpdate();
    std::cout << "inside checkModelGradient: after initModelUpdate" << std::endl;
    resizeParamBatchsize(mini_batch_size);
    std::cout << "inside checkModelGradient: after resize ParamBatchsize" << std::endl;
    for (int epoch = 0; epoch < epochs; epoch++){ // per epoch
        int iterations = floor(queryOutput->size());
        for (int iter = 0; iter < iterations; iter++){ // per batch
            checkGradient((*queryInput)[iter], (*queryOutput)[iter], mode);
        }
    }
}

void NeuralNetworkResult::checkModelGradient(std::vector<Matrix> *queryInput, std::vector<IntegerVector> *queryOutput, int epochs, std::string mode){
    mini_batch_size = 1;
    initModelUpdate();
    resizeParamBatchsize(mini_batch_size);
    for (int epoch = 0; epoch < epochs; epoch++){ // per epoch
        int iterations = floor(queryOutput->size());
        for (int iter = 0; iter < iterations; iter++){ // per batch
            checkGradient((*queryInput)[iter], (*queryOutput)[iter], mode);
        }
    }
}


Scalar NeuralNetworkResult::MSELoss(Matrix &y_pred, Matrix &y_target){
    // L = ||yhat - y||^2
    return (y_pred - y_target).squaredNorm() / y_pred.cols();
}

// y_target = [0,...,C-1]
Scalar NeuralNetworkResult::MultiClassEntropy(Matrix &y_pred, IntegerVector& y_target){
    // L = -sum( y * log(y_hat))
    //   = - log ( exp^(score_p) / sum_j^C (exp^(score_j)) )
    int mini_batch_size = y_pred.cols();
    Scalar loss = 0.0;
    for (int i = 0; i < mini_batch_size; i++){
        // - std::log(y_pred[target_label_idx])
        loss -= std::log(y_pred(y_target(i),i));
    }
    return loss/mini_batch_size;
}


void NeuralNetworkResult::checkGradient(Matrix &x_in, Matrix &y_target, std::string mode) {
    const Scalar eps = 1e-5;
    std::cout << "inside check Gradient " << std::endl;
    forwardRegression(x_in);
    backwardRegression(x_in, y_target);

    Matrix approx_dW1(W1.rows(), W1.cols());
    Matrix approx_dW2(W2.rows(), W2.cols());
    Matrix approx_dW3(W3.rows(), W3.cols());
    Matrix approx_db1(b1.rows(), b1.cols());
    Matrix approx_db2(b2.rows(), b2.cols());
    Matrix approx_db3(b3.rows(), b3.cols());
//    NNLossFunctions loss = NNLossFunctions();
    std::cout << "loss: " << (out3 - y_target).squaredNorm() / out3.cols() << std::endl;
    std::cout << "loss: " << MSELoss(out3, y_target) << std::endl;

    std::cout << "out3: " << out3 << std::endl;
    std::cout << "y_target: " << y_target << std::endl;
    std::cout << "dout3: " << dout3 << std::endl;

    std::cout << "W1: " << W1 << std::endl;
    std::cout << "W2: " << W2 << std::endl;
    std::cout << "W3: " << W3 << std::endl;
    std::cout << "b1: " << b1 << std::endl;
    std::cout << "b2: " << b2 << std::endl;
    std::cout << "b3: " << b3 << std::endl;

    //     check W1, W2, W3, b1, b2, b3
    Scalar loss_plus, loss_minus, temp;
    for(int i = 0; i < W1.rows(); i++ ) {
        for (int j = 0; j < W1.cols(); j++) {
            temp = W1(i, j);
            W1(i, j) = temp + eps;
            forwardRegression(x_in);
            loss_plus = MSELoss(out3, y_target) ;
            W1(i, j) = temp - eps;
            forwardRegression(x_in);
            loss_minus = MSELoss(out3, y_target) ;
            W1(i, j) = temp;

            approx_dW1(i, j) = (loss_plus - loss_minus) / eps / 2;
        }
    }
    Matrix m_grad_deltaW1(W1.rows(), W1.cols());
    m_grad_deltaW1.noalias() = (approx_dW1 - dW1);
    for(int i = 0; i < W1.rows(); i++ ) {
        for (int j = 0; j < W1.cols(); j++) {
            printf("(%d, %d) grad_check: dJ/d%s error norm = %f / approx_dW=%f / dW1=%f\n", i, j, "W1" , (double)m_grad_deltaW1(i,j), (double)approx_dW1(i,j), (double)dW1(i,j));
            if ( abs(eps - m_grad_deltaW1(i,j)) <= eps*220 ) {
                printf("[ok]\n");
            } else {
                printf("** ERROR **");
            }
        }
    }

    for(int i = 0; i < b1.rows(); i++ ) {
        for (int j = 0; j < b1.cols(); j++) {
            temp = b1(i, j);
            b1(i, j) = temp + eps;
            forwardRegression(x_in);
            loss_plus = MSELoss(out3, y_target) ;
            b1(i, j) = temp - eps;
            forwardRegression(x_in);
            loss_minus = MSELoss(out3, y_target) ;
            b1(i, j) = temp;

            approx_db1(i, j) = (loss_plus - loss_minus) / eps / 2;
        }
    }
    Matrix m_grad_deltab1(b1.rows(), b1.cols());
    m_grad_deltab1.noalias() = (approx_db1 - db1);
    for(int i = 0; i < b1.rows(); i++ ) {
        for (int j = 0; j < b1.cols(); j++) {
            printf("(%d, %d) grad_check: dJ/d%s error norm = %f / approx_dW=%f / db1=%f\n", i, j, "b1" , (double)m_grad_deltab1(i,j), (double)approx_db1(i,j), (double)db1(i,j));
            if ( abs(eps - m_grad_deltab1(i,j)) <= eps*220 ) {
                printf("[ok]\n");
            } else {
                printf("** ERROR **");
            }
        }
    }


    for(int i = 0; i < W2.rows(); i++ ) {
        for (int j = 0; j < W2.cols(); j++) {
            temp = W2(i, j);
            W2(i, j) = temp + eps;
            forwardRegression(x_in);
            loss_plus = MSELoss(out3, y_target) ;
            W2(i, j) = temp - eps;
            forwardRegression(x_in);
            loss_minus = MSELoss(out3, y_target) ;
            W2(i, j) = temp;

            approx_dW2(i, j) = (loss_plus - loss_minus) / eps / 2;
        }
    }
    Matrix m_grad_deltaW2(W2.rows(), W2.cols());
    m_grad_deltaW2.noalias() = (approx_dW2 - dW2);
    for(int i = 0; i < W2.rows(); i++ ) {
        for (int j = 0; j < W2.cols(); j++) {
            printf("(%d, %d) grad_check: dJ/d%s error norm = %f / approx_dW=%f / dW2=%f\n", i, j, "W2" , (double)m_grad_deltaW2(i,j), (double)approx_dW2(i,j), (double)dW2(i,j));
            if ( abs(eps - m_grad_deltaW2(i,j)) <= eps*220 ) {
                printf("[ok]\n");
            } else {
                printf("** ERROR **");
            }
        }
    }

    for(int i = 0; i < b2.rows(); i++ ) {
        for (int j = 0; j < b2.cols(); j++) {
            temp = b2(i, j);
            b2(i, j) = temp + eps;
            forwardRegression(x_in);
            loss_plus = MSELoss(out3, y_target) ;
            b2(i, j) = temp - eps;
            forwardRegression(x_in);
            loss_minus = MSELoss(out3, y_target) ;
            b2(i, j) = temp;

            approx_db2(i, j) = (loss_plus - loss_minus) / eps / 2;
        }
    }
    Matrix m_grad_deltab2(b2.rows(), b2.cols());
    m_grad_deltab2.noalias() = (approx_db2 - db2);
    for(int i = 0; i < b2.rows(); i++ ) {
        for (int j = 0; j < b2.cols(); j++) {
            printf("(%d, %d) grad_check: dJ/d%s error norm = %f / approx_dW=%f / db2=%f\n", i, j, "b2" , (double)m_grad_deltab2(i,j), (double)approx_db2(i,j), (double)db2(i,j));
            if ( abs(eps - m_grad_deltab2(i,j)) <= eps*220 ) {
                printf("[ok]\n");
            } else {
                printf("** ERROR **");
            }
        }
    }

    for(int i = 0; i < W3.rows(); i++ ) {
        for (int j = 0; j < W3.cols(); j++) {
            temp = W3(i, j);
            W3(i, j) = temp + eps;
            forwardRegression(x_in);
            loss_plus = MSELoss(out3, y_target);
            W3(i, j) = temp - eps;
            forwardRegression(x_in);
            loss_minus = MSELoss(out3, y_target);
            W3(i, j) = temp;

            approx_dW3(i, j) = (loss_plus - loss_minus) / eps / 2;
        }
    }
    Matrix m_grad_deltaW3(W3.rows(), W3.cols());
    m_grad_deltaW3.noalias() = (approx_dW3 - dW3);
    for(int i = 0; i < W3.rows(); i++ ) {
        for (int j = 0; j < W3.cols(); j++) {
            printf("(%d, %d) grad_check: dJ/d%s error norm = %f / approx_dW3=%f / dW3=%f\n", i, j, "W3" , (double)m_grad_deltaW3(i,j), (double)approx_dW3(i,j), (double)dW3(i,j));
            if ( abs(eps - m_grad_deltaW3(i,j)) <= eps*220 ) {
                printf("[ok]\n");
            } else {
                printf("** ERROR **");
            }
        }
    }

    for(int i = 0; i < b3.rows(); i++ ) {
        for (int j = 0; j < b3.cols(); j++) {
            temp = b3(i, j);
            b3(i, j) = temp + eps;
            forwardRegression(x_in);
            loss_plus = MSELoss(out3, y_target);
            b3(i, j) = temp - eps;
            forwardRegression(x_in);
            loss_minus = MSELoss(out3, y_target);
            b3(i, j) = temp;

            approx_db3(i, j) = (loss_plus - loss_minus) / eps / 2;
        }
    }
    Matrix m_grad_deltab3(b3.rows(), b3.cols());
    m_grad_deltab3.noalias() = (approx_db3 - db3);
    for(int i = 0; i < b3.rows(); i++ ) {
        for (int j = 0; j < b3.cols(); j++) {
            printf("(%d, %d) grad_check: dJ/d%s error norm = %f / approx_db3=%f / db3=%f\n", i, j, "b3" , (double)m_grad_deltab3(i,j), (double)approx_db3(i,j), (double)db3(i,j));
            if ( abs(eps - m_grad_deltab3(i,j)) <= eps*220 ) {
                printf("[ok]\n");
            } else {
                printf("** ERROR **");
            }
        }
    }
}




void NeuralNetworkResult::checkGradient(Matrix &x_in, IntegerVector &y_target, std::string mode) {
    const Scalar eps = 1e-5;
    std::cout << "Inside mc_classification checkGradient" << std::endl;
    forwardClassifyMC(x_in);
    std::cout << "Inside mc_classification forwardClassifyMC" << std::endl;
    backwardClassifyMC(x_in, y_target);
    std::cout << "Inside mc_classification backwardClassifyMC" << std::endl;

    Matrix approx_dW1(W1.rows(), W1.cols());
    Matrix approx_dW2(W2.rows(), W2.cols());
    Matrix approx_dW3(W3.rows(), W3.cols());
    Matrix approx_db1(b1.rows(), b1.cols());
    Matrix approx_db2(b2.rows(), b2.cols());
    Matrix approx_db3(b3.rows(), b3.cols());
//    NNLossFunctions loss = NNLossFunctions();
//    std::cout << "loss: " << (out3 - y_target).squaredNorm() / out3.cols() << std::endl;
    std::cout << "loss: " << MultiClassEntropy(out3, y_target) << std::endl;
    std::cout << "loss: " << std::log(out3(y_target(0), 0)) << std::endl;
    std::cout << "loss: " << -std::log(out3(y_target(0),0)) << std::endl;


    std::cout << "Z3: " << Z3 << std::endl;
    std::cout << "out3: " << out3 << std::endl;
    std::cout << "y_target: " << y_target << std::endl;
    std::cout << "dout3: " << dout3 << std::endl;

    std::cout << "W1: " << W1 << std::endl;
    std::cout << "W2: " << W2 << std::endl;
    std::cout << "W3: " << W3 << std::endl;
    std::cout << "b1: " << b1 << std::endl;
    std::cout << "b2: " << b2 << std::endl;
    std::cout << "b3: " << b3 << std::endl;

    //     check W1, W2, W3, b1, b2, b3
    Scalar loss_plus, loss_minus, temp;
    for(int i = 0; i < W1.rows(); i++ ) {
        for (int j = 0; j < W1.cols(); j++) {
            temp = W1(i, j);
            W1(i, j) = temp + eps;
            forwardClassifyMC(x_in);
            loss_plus = MultiClassEntropy(out3, y_target) ;
            W1(i, j) = temp - eps;
            forwardClassifyMC(x_in);
            loss_minus = MultiClassEntropy(out3, y_target) ;
            W1(i, j) = temp;

            approx_dW1(i, j) = (loss_plus - loss_minus) / eps / 2;
        }
    }
    Matrix m_grad_deltaW1(W1.rows(), W1.cols());
    m_grad_deltaW1.noalias() = (approx_dW1 - dW1);
    for(int i = 0; i < W1.rows(); i++ ) {
        for (int j = 0; j < W1.cols(); j++) {
            printf("(%d, %d) grad_check: dJ/d%s error norm = %f / approx_dW=%f / dW1=%f\n", i, j, "W1" , (double)m_grad_deltaW1(i,j), (double)approx_dW1(i,j), (double)dW1(i,j));
            if ( abs(eps - m_grad_deltaW1(i,j)) <= eps*220 ) {
                printf("[ok]\n");
            } else {
                printf("** ERROR **");
            }
        }
    }

    for(int i = 0; i < b1.rows(); i++ ) {
        for (int j = 0; j < b1.cols(); j++) {
            temp = b1(i, j);
            b1(i, j) = temp + eps;
            forwardClassifyMC(x_in);
            loss_plus = MultiClassEntropy(out3, y_target) ;
            b1(i, j) = temp - eps;
            forwardClassifyMC(x_in);
            loss_minus = MultiClassEntropy(out3, y_target) ;
            b1(i, j) = temp;

            approx_db1(i, j) = (loss_plus - loss_minus) / eps / 2;
        }
    }
    Matrix m_grad_deltab1(b1.rows(), b1.cols());
    m_grad_deltab1.noalias() = (approx_db1 - db1);
    for(int i = 0; i < b1.rows(); i++ ) {
        for (int j = 0; j < b1.cols(); j++) {
            printf("(%d, %d) grad_check: dJ/d%s error norm = %f / approx_dW=%f / db1=%f\n", i, j, "b1" , (double)m_grad_deltab1(i,j), (double)approx_db1(i,j), (double)db1(i,j));
            if ( abs(eps - m_grad_deltab1(i,j)) <= eps*220 ) {
                printf("[ok]\n");
            } else {
                printf("** ERROR **");
            }
        }
    }


    for(int i = 0; i < W2.rows(); i++ ) {
        for (int j = 0; j < W2.cols(); j++) {
            temp = W2(i, j);
            W2(i, j) = temp + eps;
            forwardClassifyMC(x_in);
            loss_plus = MultiClassEntropy(out3, y_target) ;
            W2(i, j) = temp - eps;
            forwardClassifyMC(x_in);
            loss_minus = MultiClassEntropy(out3, y_target) ;
            W2(i, j) = temp;

            approx_dW2(i, j) = (loss_plus - loss_minus) / eps / 2;
        }
    }
    Matrix m_grad_deltaW2(W2.rows(), W2.cols());
    m_grad_deltaW2.noalias() = (approx_dW2 - dW2);
    for(int i = 0; i < W2.rows(); i++ ) {
        for (int j = 0; j < W2.cols(); j++) {
            printf("(%d, %d) grad_check: dJ/d%s error norm = %f / approx_dW=%f / dW2=%f\n", i, j, "W2" , (double)m_grad_deltaW2(i,j), (double)approx_dW2(i,j), (double)dW2(i,j));
            if ( abs(eps - m_grad_deltaW2(i,j)) <= eps*220 ) {
                printf("[ok]\n");
            } else {
                printf("** ERROR **");
            }
        }
    }

    for(int i = 0; i < b2.rows(); i++ ) {
        for (int j = 0; j < b2.cols(); j++) {
            temp = b2(i, j);
            b2(i, j) = temp + eps;
            forwardClassifyMC(x_in);
            loss_plus = MultiClassEntropy(out3, y_target) ;
            b2(i, j) = temp - eps;
            forwardClassifyMC(x_in);
            loss_minus = MultiClassEntropy(out3, y_target) ;
            b2(i, j) = temp;

            approx_db2(i, j) = (loss_plus - loss_minus) / eps / 2;
        }
    }
    Matrix m_grad_deltab2(b2.rows(), b2.cols());
    m_grad_deltab2.noalias() = (approx_db2 - db2);
    for(int i = 0; i < b2.rows(); i++ ) {
        for (int j = 0; j < b2.cols(); j++) {
            printf("(%d, %d) grad_check: dJ/d%s error norm = %f / approx_dW=%f / db2=%f\n", i, j, "b2" , (double)m_grad_deltab2(i,j), (double)approx_db2(i,j), (double)db2(i,j));
            if ( abs(eps - m_grad_deltab2(i,j)) <= eps*220 ) {
                printf("[ok]\n");
            } else {
                printf("** ERROR **");
            }
        }
    }

    for(int i = 0; i < W3.rows(); i++ ) {
        for (int j = 0; j < W3.cols(); j++) {
            temp = W3(i, j);
            W3(i, j) = temp + eps;
            forwardClassifyMC(x_in);
            loss_plus = MultiClassEntropy(out3, y_target);
            W3(i, j) = temp - eps;
            forwardClassifyMC(x_in);
            loss_minus = MultiClassEntropy(out3, y_target);
            W3(i, j) = temp;

            approx_dW3(i, j) = (loss_plus - loss_minus) / eps / 2;
        }
    }
    Matrix m_grad_deltaW3(W3.rows(), W3.cols());
    m_grad_deltaW3.noalias() = (approx_dW3 - dW3);
    for(int i = 0; i < W3.rows(); i++ ) {
        for (int j = 0; j < W3.cols(); j++) {
            printf("(%d, %d) grad_check: dJ/d%s error norm = %f / approx_dW3=%f / dW3=%f\n", i, j, "W3" , (double)m_grad_deltaW3(i,j), (double)approx_dW3(i,j), (double)dW3(i,j));
            if ( abs(eps - m_grad_deltaW3(i,j)) <= eps*220 ) {
                printf("[ok]\n");
            } else {
                printf("** ERROR **");
            }
        }
    }

    for(int i = 0; i < b3.rows(); i++ ) {
        for (int j = 0; j < b3.cols(); j++) {
            temp = b3(i, j);
            b3(i, j) = temp + eps;
            forwardClassifyMC(x_in);
            loss_plus = MultiClassEntropy(out3, y_target);
            b3(i, j) = temp - eps;
            forwardClassifyMC(x_in);
            loss_minus = MultiClassEntropy(out3, y_target);
            b3(i, j) = temp;

            approx_db3(i, j) = (loss_plus - loss_minus) / eps / 2;
        }
    }
    Matrix m_grad_deltab3(b3.rows(), b3.cols());
    m_grad_deltab3.noalias() = (approx_db3 - db3);
    for(int i = 0; i < b3.rows(); i++ ) {
        for (int j = 0; j < b3.cols(); j++) {
            printf("(%d, %d) grad_check: dJ/d%s error norm = %f / approx_db3=%f / db3=%f\n", i, j, "b3" , (double)m_grad_deltab3(i,j), (double)approx_db3(i,j), (double)db3(i,j));
            if ( abs(eps - m_grad_deltab3(i,j)) <= eps*220 ) {
                printf("[ok]\n");
            } else {
                printf("** ERROR **");
            }
        }
    }
}
