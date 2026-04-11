#include <rapidcsv.h>
#include <Eigen/Dense>
#include <cmath>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <utility>
#include <vector>

#define EIGEN_DEFAULT_TO_ROW_MAJOR
using RowMatrixXf = Eigen::MatrixXf;

/**
 * @brief simple neural network class with one hidden layer (inputDim -> hiddenDim -> outputDim)
 *
 * provides forward propagation, backward propagation, and parameter updates using the adam optimizer
 */

class nn {
 private:
  //model weights
  Eigen::MatrixXf w1;
  Eigen::MatrixXf b1;
  Eigen::MatrixXf w2;
  Eigen::MatrixXf b2;

  Eigen::MatrixXf a1;
  Eigen::MatrixXf z1;
  RowMatrixXf probs;

  //gradients
  Eigen::MatrixXf dJdw2;
  Eigen::MatrixXf dJdw1;
  Eigen::MatrixXf dJdb2;
  Eigen::MatrixXf dJdb1;

  //momentum for adam optimizer
  float iter = 0.0f;
  Eigen::MatrixXf m_dJdw2;
  Eigen::MatrixXf v_dJdw2;
  Eigen::MatrixXf m_dJdw1;
  Eigen::MatrixXf v_dJdw1;
  Eigen::MatrixXf m_dJdb2;
  Eigen::MatrixXf v_dJdb2;
  Eigen::MatrixXf m_dJdb1;
  Eigen::MatrixXf v_dJdb1;


  static RowMatrixXf Linear(const RowMatrixXf &x, const RowMatrixXf &w,
                            const RowMatrixXf &b);
  static RowMatrixXf ReLu(const RowMatrixXf &x);
  static RowMatrixXf softmax(const RowMatrixXf &x);

  void adam(Eigen::MatrixXf &theta, Eigen::MatrixXf &m,
                   Eigen::MatrixXf &v, const Eigen::MatrixXf &g);


 public:
  //adam optimizer hyperparameters
  float lr = 1e-3f;
  float beta1 = 0.9f;
  float beta2 = 0.999f;
  float eps = 1e-8f;
  
  
  nn(int inputDim, int hiddenDim, int outputDim) {
    w1 = Eigen::MatrixXf::Random(hiddenDim, inputDim) * 0.01f;
    w2 = Eigen::MatrixXf::Random(outputDim, hiddenDim) * 0.01f;
    b1 = Eigen::MatrixXf::Zero(hiddenDim, 1);
    b2 = Eigen::MatrixXf::Zero(outputDim, 1);

    m_dJdw1 = Eigen::MatrixXf::Zero(hiddenDim, inputDim);
    v_dJdw1 = Eigen::MatrixXf::Zero(hiddenDim, inputDim);
    m_dJdw2 = Eigen::MatrixXf::Zero(outputDim, hiddenDim);
    v_dJdw2 = Eigen::MatrixXf::Zero(outputDim, hiddenDim);
    m_dJdb1 = Eigen::MatrixXf::Zero(hiddenDim, 1);
    v_dJdb1 = Eigen::MatrixXf::Zero(hiddenDim, 1);
    m_dJdb2 = Eigen::MatrixXf::Zero(outputDim, 1);
    v_dJdb2 = Eigen::MatrixXf::Zero(outputDim, 1);

    dJdw1 = Eigen::MatrixXf::Zero(hiddenDim, inputDim);
    dJdw2 = Eigen::MatrixXf::Zero(outputDim, hiddenDim);
    dJdb1 = Eigen::MatrixXf::Zero(hiddenDim, 1);
    dJdb2 = Eigen::MatrixXf::Zero(outputDim, 1);
  }

  ~nn() = default;

  RowMatrixXf forward(const RowMatrixXf &x);
  static RowMatrixXf oneHotEncoding(const Eigen::VectorXf &y, int numClasses);
  void backward(const RowMatrixXf &x, const Eigen::VectorXf &y);
  void step();
};


/**
 * @brief linear transform: output = w * x + b
 * @param x input matrix (currDim x m)
 * @param w weights (nextDim x currDim)
 * @param b bias (nextDim x 1)
 * @return output of linear transformation matrix (nextDim x m)
 */
RowMatrixXf nn::Linear(const RowMatrixXf &x, const RowMatrixXf &w, const RowMatrixXf &b) {
  const RowMatrixXf bBroadcast = b.replicate(1, x.cols());
  return w * x + bBroadcast;
}


/**
 * @brief rectified linear unit (ReLU)
 * @param x input matrix (nextDim x m)
 * @return output of ReLU matrix (nextDim x m)
 */
RowMatrixXf nn::ReLu(const RowMatrixXf &x) { return x.cwiseMax(0.0f); }


/**
 * @brief softmax (f(z)_i = e^z_i / sumj[e^z_j])
 * @param x input matrix (classes x m)
 * @return output of softmax matrix (classes x m)
 */
RowMatrixXf nn::softmax(const RowMatrixXf &x) {
  const auto colmax = x.colwise().maxCoeff();
  const RowMatrixXf shifted = x - colmax.replicate(x.rows(), 1);
  const RowMatrixXf exps = shifted.array().exp();
  const auto colsum = exps.colwise().sum();
  return exps.array().cwiseQuotient(colsum.replicate(x.rows(), 1).array());
}


/**
 * @brief forward pass
 * @param x input matrix (inputDim x m)
 * @return logits (classes x m)
 */
RowMatrixXf nn::forward(const RowMatrixXf &x) {
  RowMatrixXf h = Linear(x, w1, b1);
  z1 = h;
  h = ReLu(h);
  a1 = h;
  const RowMatrixXf logits = Linear(h, w2, b2);

  probs = softmax(logits);


  return logits;
}


/**
 * @brief one hot encodes all the labels 
 * @param y labels for all m samples (m x 1)
 * @param numClasses number of classes
 * @return Y one hot encoded matrix (numClasses x m)
 */
RowMatrixXf nn::oneHotEncoding(const Eigen::VectorXf &y, int numClasses) {
  const Eigen::Index m = y.size();
  RowMatrixXf Y = RowMatrixXf::Zero(numClasses, m);
  for (Eigen::Index j = 0; j < m; ++j) {
    const int lab = static_cast<int>(y(j));
    if (lab >= 0 && lab < numClasses)
      Y(lab, j) = 1.0f;
  }


  return Y;
}


/**
 * @brief computes gradients via backwards propogation 
 * @param x input matrix (inputDim x m)
 * @param y labels for all m samples (m x 1)
*/
void nn::backward(const RowMatrixXf &x, const Eigen::VectorXf &y) {
  const Eigen::Index m = x.cols();
  const int numClasses = static_cast<int>(w2.rows());
  const RowMatrixXf Y = oneHotEncoding(y, numClasses);

  const RowMatrixXf dJdz2 = (probs - Y) / static_cast<float>(m);
  dJdw2 = dJdz2 * a1.transpose();
  dJdb2 = dJdz2.rowwise().sum();
  const RowMatrixXf dJda1 = w2.transpose() * dJdz2;
  const RowMatrixXf dJdz1 = dJda1.cwiseProduct((z1.array() > 0.f).cast<float>().matrix());
  dJdw1 = dJdz1 * x.transpose();
  dJdb1 = dJdz1.rowwise().sum();
}


/**
 * @brief executes one adam optimizer step for each parameter
 * @param theta parameter (weight or bias) being updated
 * @param m running average of first moment
 * @param v running average of second moment
 * @param g new gradient
*/
void nn::adam(Eigen::MatrixXf &theta, Eigen::MatrixXf &m, Eigen::MatrixXf &v, const Eigen::MatrixXf &g) {
  m = beta1 * m + (1.f - beta1) * g;
  Eigen::MatrixXf mAdj = m / (1.f - std::pow(beta1, iter));
  v = beta2 * v + (1.f - beta2) * g.cwiseProduct(g);
  Eigen::MatrixXf vAdj = v / (1.f - std::pow(beta2, iter));
  theta.array() -= lr * mAdj.array() / (vAdj.array().sqrt() + eps);
}

/**
 * @brief executes adam optimizer step for all parameters
*/
void nn::step() {
  iter += 1.0f;
  adam(w1, m_dJdw1, v_dJdw1, dJdw1);
  adam(w2, m_dJdw2, v_dJdw2, dJdw2);
  adam(b1, m_dJdb1, v_dJdb1, dJdb1);
  adam(b2, m_dJdb2, v_dJdb2, dJdb2);
}






struct TrainTestSplit {
  RowMatrixXf xTrain;
  RowMatrixXf xTest;
  Eigen::VectorXf yTrain;
  Eigen::VectorXf yTest;
};


RowMatrixXf readData(const std::string &path) {
  rapidcsv::Document doc(path, rapidcsv::LabelParams(0, -1));
  const Eigen::Index n = static_cast<Eigen::Index>(doc.GetRowCount());
  const int kCols = 785;
  RowMatrixXf m(n, kCols);
  for (Eigen::Index i = 0; i < n; ++i) {
    const std::vector<float> row = doc.GetRow<float>(static_cast<size_t>(i));
    for (int c = 0; c < kCols; ++c)
      m(i, c) = row[static_cast<size_t>(c)];
  }
  return m;
}


TrainTestSplit prepareTrainTest(RowMatrixXf raw, unsigned rngSeed) {
  const size_t m = static_cast<size_t>(raw.rows());
  std::vector<size_t> perm(m);
  std::iota(perm.begin(), perm.end(), 0);
  std::mt19937 gen(rngSeed);
  std::shuffle(perm.begin(), perm.end(), gen);

  RowMatrixXf shuffled(static_cast<Eigen::Index>(m), raw.cols());
  for (size_t i = 0; i < m; ++i)
    shuffled.row(static_cast<Eigen::Index>(i)) =
        raw.row(static_cast<Eigen::Index>(perm[i]));

  const size_t testN = static_cast<size_t>(m / 3.333);
  const size_t trainN = m - testN;
  const auto tr = static_cast<Eigen::Index>(trainN);

  const RowMatrixXf trainRows = shuffled.topRows(tr);
  const RowMatrixXf testRows = shuffled.bottomRows(shuffled.rows() - tr);

  TrainTestSplit out;
  out.yTrain = trainRows.col(0);
  out.yTest = testRows.col(0);

  out.xTrain = trainRows.rightCols(784).transpose().array() / 255.f;
  out.xTest = testRows.rightCols(784).transpose().array() / 255.f;
  return out;
}


int accuracy(const RowMatrixXf &logits, const Eigen::VectorXf &y) {
  const Eigen::Index m = y.size();
  if (logits.cols() != m)
    return 0;
  int correct = 0;
  for (Eigen::Index j = 0; j < m; ++j) {
    Eigen::Index pred = 0;
    logits.col(j).maxCoeff(&pred);
    if (pred == static_cast<int>(y(j)))
      ++correct;
  }


  return static_cast<int>(100.0 * static_cast<double>(correct) / static_cast<double>(m));
}


std::pair<RowMatrixXf, Eigen::VectorXf> dataLoader(const RowMatrixXf &x,
                                                  const Eigen::VectorXf &y,
                                                  int batchSize) {
  static std::mt19937 gen{std::random_device{}()};
  const Eigen::Index n = x.cols();
  if (n <= 0 || batchSize <= 0)
    return {RowMatrixXf(x.rows(), 0), Eigen::VectorXf(0)};
  std::uniform_int_distribution<Eigen::Index> dist(0, n - 1);

  RowMatrixXf xBatch(x.rows(), batchSize);
  Eigen::VectorXf yBatch(batchSize);
  for (int i = 0; i < batchSize; ++i) {
    const Eigen::Index idx = dist(gen);
    xBatch.col(i) = x.col(idx);
    yBatch(i) = y(idx);
  }


  return {xBatch, yBatch};
}




int main() {
  const std::string csv = "../data/data.csv";

  auto data = readData(csv);
  auto split = prepareTrainTest(std::move(data), 42);

  std::cout << "train samples (cols): " << split.xTrain.cols()
            << "  test samples: " << split.xTest.cols()
            << "  feature rows: " << split.xTrain.rows() << '\n';


  int epochs = 10000;
  int batchSize = 128;
  nn model(784, 64, 10); //initialize object

  //training loop
  for (int epoch = 0; epoch < epochs; epoch++) {
    const auto [xTrainBatch, yTrainBatch] = dataLoader(split.xTrain, split.yTrain, batchSize);

    (void)model.forward(xTrainBatch); //forward propogate
    model.backward(xTrainBatch, yTrainBatch); //backward propogate to get gradients
    model.step(); //parameter update step

    //get accuracy metrics and print
    const int trainAcc = accuracy(model.forward(xTrainBatch), yTrainBatch);
    const int testAcc = accuracy(model.forward(split.xTest), split.yTest);
    std::cout << "epoch " << epoch << "  train acc " << trainAcc << "%  test acc " << testAcc << "%\n";
  }

  return 0;
}
