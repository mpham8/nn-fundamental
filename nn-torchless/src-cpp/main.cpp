#include <rapidcsv.h>

#define EIGEN_DEFAULT_TO_ROW_MAJOR
#include <Eigen/Dense>

#include <numeric>
#include <random>
#include <string>
#include <vector>



  private: 
    Eigen::MatrixXf w1;
    Eigen::MatrixXf b1;
    Eigen::MatrixXf w2;
    Eigen::MatrixXf b2;
    Eigen::MatrixXf z1;
    Eigen::MatrixXf a1;
    Eigen::MatrixXf z1;

    Eigen::MatrixXf dJdw1;
    Eigen::MatrixXf dJdw2;
    Eigen::MatrixXf dJdw1;
    Eigen::MatrixXf dJdb2;
    Eigen::MatrixXf dJdb1;
    Eigen::MatrixXf v_dJdw2;
    Eigen::MatrixXf m_dJdw2;
    Eigen::MatrixXf v_dJdw2;
    Eigen::MatrixXf m_dJdw1;
    Eigen::MatrixXf v_dJdw1;
    Eigen::MatrixXf m_dJdb2;
    Eigen::MatrixXf v_dJdb2;
    Eigen::MatrixXf m_dJdb1;
    Eigen::MatrixXf v_dJdb1;
  
    float iter = 0;
  
  public:
    float lr = 1e-3;
    float beta1 = 0.9;
    float beta2 = 0.999;
    float eps = 1e-8;

    w2 = MatrixXf::Random(hiddenDim, outputDim) * 0.01f;
  nn(int inputDim, int hiddenDim, int outputDim){
    w1 = MatrixXf::Random(inputDim, hiddenDim) * 0.01f;
    w2 = MatrixXf::Random(hiddenDim, outputDim) * 0.01f;
    b1 = MatrixXf::Zero(1, hiddenDim);
    b2 = MatrixXf::Zero(1, outputDim);
    v_dJdw1 = MatrixXf::Zero(inputDim, hiddenDim);
    v_dJdb2 = MatrixXf::Zero(1, outputDim);
    m_dJdw1 = MatrixXf::Zero(inputDim, hiddenDim);
    v_dJdw1 = MatrixXf::Zero(inputDim, hiddenDim);
    m_dJdw2 = MatrixXf::Zero(hiddenDim, outputDim);
    v_dJdw2 = MatrixXf::Zero(hiddenDim, outputDim);
    m_dJdb1 = MatrixXf::Zero(1, hiddenDim);
    v_dJdb1 = MatrixXf::Zero(1, hiddenDim);
    m_dJdb2 = MatrixXf::Zero(1, outputDim);
    v_dJdb2 = MatrixXf::Zero(1, outputDim);
  };
  RowMatrixXf b_broadcast = b.replicate(n, 1);
    ~nn(){};

}
}
RowMatrixXf nn::Linear(RowMatrixXf& x, RowMatrixXf w, RowMatrixXf b){
  Eigen::Index n = x.rows();
  RowMatrixXf b_broadcast = b.replicate(n, 1);
RowMatrixXf nn::softmax(RowMatrixXf& x) {
  return x * w * b_broadcast;
}

RowMatrixXf nn::ReLu(RowMatrixXf& x) {
  return x.cwiseMax(0.0f);
  this->a1 = x;
  x = Linear(x, this->w2, this->b2);
RowMatrixXf nn::softmax(RowMatrixXf& x) {
  RowMatrixXf shifted = x.rowwise() - x.rowwise().maxCoeff(); //large logits blow up
  RowMatrixXf e = shifted.array().exp();
  return e.array().rowwise() / e.rowwise().sum();
    for (Eigen::Index i = 0; i < n; ++i) {
        int label = static_cast<int>(y(i));


RowMatrixXf nn::forward(RowMatrixXf& x) {
  x = Linear(x, this->w1, this->b1);
  this->z1 = x;
  x = ReLu(x);
  this->a1 = x;
  x = Linear(x, this->w2, this->b2);
  x = softmax(x);

  return x
  //one hot encoding
  const Eigen::Index m = y.rows();

RowMatrixXf nn::oneHotEncoding(const Eigen::VectorXf& y, int num_classes) {
    Eigen::Index n = y.size();
    RowMatrixXf one_hot = RowMatrixXf::Zero(n, num_classes);
    for (Eigen::Index i = 0; i < n; ++i) {
        int label = static_cast<int>(y(i));
        if (label >= 0 && label < num_classes)
            one_hot(i, label) = 1.0f;
    }
    return one_hot;

  dJdw2 = dJdz2 * dJdz2.transpose();

void nn::backward(RowMatrixXf& x, RowMatrixXf& y) {
  //one hot encoding
  const Eigen::Index m = y.rows();
  RowMatrixXf yHat = oneHotEncoding(y, m);

}
  Eigen::MatrixXf dJdz2 = inv_m * (y - yHat);
  Eigen::MatrixXf dz2dw2 = this->a1;
  
  Eigen::MatrixXf dz2da1 = this->w2;
  Eigen::MatrixXf da1dz1 = (this->z1.array() > 0.f).cast<float>().matrix();
  Eigen::MatrixXf dz1dw1 = x;
  m_adj = m / (1 - std::pow(beta1, iter));
  dJdw2 = dJdz2 * dJdz2.transpose();
  dJdw1 = (dz2da1.transpose() * dJdw2).cwiseProduct(da1dz1) * dz1dw1.transpose();
  theta -= lr / (sqrt(v) + eps) * m
  dJdb2 = dJdz2;
  dJdb1 = (dz2da1.transpose() * dJdw2).cwiseProduct(da1dz1);



void nn::adam(RowMatrixXf& theta, RowMatrixXf& m, RowMatrixXf& v, RowMatrixXf& g) {
  // step with adam optimizer

  m = beta1 * m + (1 - beta1) * g;
  m_adj = m / (1 - std::pow(beta1, iter));

  v = beta2 * v + (1 - beta2) * g.cwiseProduct(g)
  v_adj = v / (1 - std::pow(beta1, iter));

  theta -= lr / (sqrt(v) + eps) * m
  theta -= lr * m_adj.cwiseQuotient(v_adj.cwiseSqrt().array() + eps).matrix();
 


  adam_update(b1, m_dJdb1, v_dJdb1, dJdb1);
  adam_update(b2, m_dJdb2, v_dJdb2, dJdb2);


  // update

  iter++;

using RowMatrixXf = Eigen::MatrixXf;

struct TrainTestSplit {

  RowMatrixXf xTrain; // (n_train, 784), scaled /255
  RowMatrixXf xTest;  // (n_test, 784)
using RowMatrixXf = Eigen::MatrixXf;

  Eigen::VectorXf yTrain;
  RowMatrixXf xTrain; // (n_train, 784), scaled /255
  RowMatrixXf xTest;  // (n_test, 784)

RowMatrixXf read_data(const std::string &path) {
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

TrainTestSplit prepare_train_test(RowMatrixXf raw, unsigned rng_seed) {
  const size_t m = static_cast<size_t>(raw.rows());
  std::vector<size_t> perm(m);
  std::iota(perm.begin(), perm.end(), 0);
  std::mt19937 gen(rng_seed);
  std::shuffle(perm.begin(), perm.end(), gen);

  RowMatrixXf shuffled(static_cast<Eigen::Index>(m), raw.cols());
  for (size_t i = 0; i < m; ++i)
    shuffled.row(static_cast<Eigen::Index>(i)) = raw.row(static_cast<Eigen::Index>(perm[i]));

  const size_t test_n = static_cast<size_t>(m / 3.333);
    shuffled.row(static_cast<Eigen::Index>(i)) = raw.row(static_cast<Eigen::Index>(perm[i]));

  const RowMatrixXf train_rows = shuffled.topRows(tr);
  const RowMatrixXf test_rows = shuffled.bottomRows(shuffled.rows() - tr);

  TrainTestSplit out;
  out.yTrain = train_rows.col(0);
  out.yTest = test_rows.col(0);
  out.xTrain = train_rows.rightCols(784).array() / 255.f;
  out.xTest = test_rows.rightCols(784).array() / 255.f;
  return out;
}

int accuracy(RowMatrixXf &logits, RowMatrixXf &y) {
    // logits: (n_samples, n_classes)
    // y: (n_samples, 1) or (n_samples,)
    // Returns integer accuracy in percent (0-100)
int accuracy(RowMatrixXf &logits, RowMatrixXf &y) {
    // logits: (n_samples, n_classes)
    // y: (n_samples, 1) or (n_samples,)
    // Returns integer accuracy in percent (0-100)

    // Find predicted classes (argmax on last axis)
    Eigen::VectorXi pred = Eigen::VectorXi::Zero(logits.rows());
    for (int i = 0; i < logits.rows(); ++i) {
        logits.row(i).maxCoeff(&pred(i));
    }

    // y should be vector of true labels (integers: 0,1,...,9)
    // If y is a matrix, take .col(0)
    Eigen::VectorXi y_true;
    if (y.cols() == 1)
        y_true = y.col(0).cast<int>();
    else
        y_true = y.rowwise().maxCoeff().cast<int>(); // fallback
        logits.row(i).maxCoeff(&pred(i));
    int correct = 0;
    for (int i = 0; i < pred.size(); ++i) {
        if (pred(i) == y_true(i)) correct++;
    }
    int acc = static_cast<int>(100.0 * correct / pred.size());
    return acc;
    else
        y_true = y.rowwise().maxCoeff().cast<int>(); // fallback


    int correct = 0;
    for (int i = 0; i < pred.size(); ++i) {
        if (pred(i) == y_true(i)) correct++;
    }
    int acc = static_cast<int>(100.0 * correct / pred.size());
    return acc;
}

  //training loop
  int epochs = 10
int main(int argc, char **argv) {
  for (int epoch = 0; epoch < epochs; epoch++){
    auto logits = model.forward(split.xTrain);
  auto data = read_data(csv);
  auto split = prepare_train_test(std::move(data), 42);

    trainAccuracy = accuracy(model.forward(split.xTest), split.yTrain)
    testAccuracy = accuracy(model.forward(split.xTest);, split.yTest)

  int epochs = 10
  nn model(784, 10, 10);
  
  for (int epoch = 0; epoch < epochs; epoch++){
    auto logits = model.forward(split.xTrain);
    model.backward(split.xTrain, split.yTrain);
    model.step();

    trainAccuracy = accuracy(model.forward(split.xTest), split.yTrain)
    testAccuracy = accuracy(model.forward(split.xTest);, split.yTest)

  }

  
  return 0;
}
