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
  Eigen::MatrixXf m_dJdw2;
  Eigen::MatrixXf v_dJdw2;
  Eigen::MatrixXf m_dJdw1;
  Eigen::MatrixXf v_dJdw1;
  Eigen::MatrixXf m_dJdb2;
  Eigen::MatrixXf v_dJdb2;
  Eigen::MatrixXf m_dJdb1;
  Eigen::MatrixXf v_dJdb1;

  int iter = 0;

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
    w1 = Eigen::MatrixXf::Random(inputDim, hiddenDim) * 0.01f;
    w2 = Eigen::MatrixXf::Random(hiddenDim, outputDim) * 0.01f;
    b1 = Eigen::MatrixXf::Zero(1, hiddenDim);
    b2 = Eigen::MatrixXf::Zero(1, outputDim);

    m_dJdw1 = Eigen::MatrixXf::Zero(inputDim, hiddenDim);
    v_dJdw1 = Eigen::MatrixXf::Zero(inputDim, hiddenDim);
    m_dJdw2 = Eigen::MatrixXf::Zero(hiddenDim, outputDim);
    v_dJdw2 = Eigen::MatrixXf::Zero(hiddenDim, outputDim);
    m_dJdb1 = Eigen::MatrixXf::Zero(1, hiddenDim);
    v_dJdb1 = Eigen::MatrixXf::Zero(1, hiddenDim);
    m_dJdb2 = Eigen::MatrixXf::Zero(1, outputDim);
    v_dJdb2 = Eigen::MatrixXf::Zero(1, outputDim);

    dJdw1 = Eigen::MatrixXf::Zero(inputDim, hiddenDim);
    dJdw2 = Eigen::MatrixXf::Zero(hiddenDim, outputDim);
    dJdb1 = Eigen::MatrixXf::Zero(1, hiddenDim);
    dJdb2 = Eigen::MatrixXf::Zero(1, outputDim);
    probs = Eigen::MatrixXf::Zero(1, outputDim);
  }

  ~nn() = default;

  RowMatrixXf forward(const RowMatrixXf &x);
  static RowMatrixXf oneHotEncoding(const Eigen::VectorXf &y, int numClasses);
  void backward(const RowMatrixXf &x, const Eigen::VectorXf &y);
  void step();
};


RowMatrixXf nn::Linear(const RowMatrixXf &x, const RowMatrixXf &w,
                       const RowMatrixXf &b) {
  const Eigen::Index n = x.rows();
  const RowMatrixXf bBroadcast = b.replicate(n, 1);
  return x * w + bBroadcast;
}

RowMatrixXf nn::ReLu(const RowMatrixXf &x) { return x.cwiseMax(0.0f); }

RowMatrixXf nn::softmax(const RowMatrixXf &x) {
  const Eigen::VectorXf rowmax = x.rowwise().maxCoeff();
  const RowMatrixXf shifted = x - rowmax.replicate(1, x.cols());
  const RowMatrixXf exps = shifted.array().exp();
  const Eigen::VectorXf sums = exps.rowwise().sum();
  return exps.array().cwiseQuotient(sums.replicate(1, x.cols()).array());
}

RowMatrixXf nn::forward(const RowMatrixXf &x) {
  RowMatrixXf h = Linear(x, w1, b1);
  z1 = h;
  h = ReLu(h);
  a1 = h;
  const RowMatrixXf logits = Linear(h, w2, b2);
  probs = softmax(logits);
  return probs;
}

RowMatrixXf nn::oneHotEncoding(const Eigen::VectorXf &y, int numClasses) {
  RowMatrixXf oneHot = RowMatrixXf::Zero(y.size(), numClasses);
  for (Eigen::Index i = 0; i < y.size(); ++i)
    if (y(i) >= 0 && y(i) < numClasses)
      oneHot(i, static_cast<int>(y(i))) = 1.0f;
  return oneHot;
}

void nn::backward(const RowMatrixXf &x, const Eigen::VectorXf &y) {
  const Eigen::Index m = x.rows();
  const int numClasses = static_cast<int>(w2.cols());
  const RowMatrixXf Y = oneHotEncoding(y, numClasses);

  const RowMatrixXf dJdz2 = 1.0f / static_cast<float>(m) * (probs - Y);
  dJdw2 = a1.transpose() * dJdz2;
  dJdb2 = dJdz2.colwise().sum();
  const RowMatrixXf dJda1 = dJdz2 * w2.transpose();
  const RowMatrixXf dJdz1 = dJda1.cwiseProduct((z1.array() > 0.f).cast<float>().matrix());
  dJdw1 = x.transpose() * dJdz1;
  dJdb1 = dJdz1.colwise().sum();
}

void nn::adam(Eigen::MatrixXf &theta, Eigen::MatrixXf &m,
                     Eigen::MatrixXf &v, const Eigen::MatrixXf &g) {
  m = beta1 * m + (1.f - beta1) * g;
  Eigen::MatrixXf mAdj = m / (1.f - std::pow(beta1, static_cast<float>(iter)));
  v = beta2 * v + (1.f - beta2) * g.cwiseProduct(g);
  Eigen::MatrixXf vAdj = v / (1.f - std::pow(beta2, static_cast<float>(iter)));
  theta.array() -= lr * mAdj.array() / (vAdj.array().sqrt() + eps);
}

void nn::step() {
  iter++;
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
  out.xTrain = trainRows.rightCols(784).array() / 255.f;
  out.xTest = testRows.rightCols(784).array() / 255.f;
  return out;
}

int accuracy(const RowMatrixXf &logits, const Eigen::VectorXf &y) {
  Eigen::VectorXi pred(logits.rows());
  for (Eigen::Index i = 0; i < logits.rows(); ++i)
    logits.row(i).maxCoeff(&pred(i));

  int correct = 0;
  for (Eigen::Index i = 0; i < pred.size(); ++i) {
    if (pred(i) == static_cast<int>(y(i)))
      ++correct;
  }
  return static_cast<int>(100.0 * static_cast<double>(correct) /
                           static_cast<double>(logits.rows()));
}

std::pair<RowMatrixXf, Eigen::VectorXf> dataLoader(const RowMatrixXf &x,
                                                  const Eigen::VectorXf &y,
                                                  int batchSize) {
  static std::mt19937 gen{std::random_device{}()};
  const Eigen::Index n = x.rows();
  if (n <= 0 || batchSize <= 0)
    return {RowMatrixXf(0, x.cols()), Eigen::VectorXf(0)};
  std::uniform_int_distribution<Eigen::Index> dist(0, n - 1);

  RowMatrixXf xBatch(batchSize, x.cols());
  Eigen::VectorXf yBatch(batchSize);
  for (int i = 0; i < batchSize; ++i) {
    const Eigen::Index idx = dist(gen);
    xBatch.row(i) = x.row(idx);
    yBatch(i) = y(idx);
  }
  return {xBatch, yBatch};
}



int main(int argc, char **argv) {
  const std::string csv = (argc > 1) ? argv[1] : "../data/data.csv";

  auto data = readData(csv);
  auto split = prepareTrainTest(std::move(data), 42);

  std::cout << "train: " << split.xTrain.rows() << "  test: " << split.xTest.rows()
            << "  features: " << split.xTrain.cols() << '\n';

  int epochs = 10000;
  int batchSize = 128;
  nn model(784, 64, 10);
  for (int epoch = 0; epoch < epochs; epoch++) {
    const auto [xTrainBatch, yTrainBatch] = dataLoader(split.xTrain, split.yTrain, batchSize);

    (void)model.forward(xTrainBatch);
    model.backward(xTrainBatch, yTrainBatch);
    model.step();

    const int trainAcc = accuracy(model.forward(xTrainBatch), yTrainBatch);
    const int testAcc = accuracy(model.forward(split.xTest), split.yTest);
    std::cout << "epoch " << epoch << "  train acc " << trainAcc << "%  test acc "
              << testAcc << "%\n";
  }

  return 0;
}
