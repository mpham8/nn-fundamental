#include <rapidcsv.h>

#include <iostream>
#include <random>
#include <string>
#include <vector>



struct TrainTestSplit {
  std::vector<std::vector<float>> x_train;
  std::vector<std::vector<float>> x_test;
  std::vector<float> y_train;
  std::vector<float> y_test;
};

std::vector<std::vector<float>> read_data(const std::string &path) {
  rapidcsv::Document doc(path, rapidcsv::LabelParams(0, -1));
  const size_t n = doc.GetRowCount();
  std::vector<std::vector<float>> data;
  data.reserve(n);
  for (size_t i = 0; i < n; ++i)
    data.push_back(doc.GetRow<float>(i));
  return data;
}

TrainTestSplit prepare_train_test(std::vector<std::vector<float>> data, unsigned rng_seed) {
  std::mt19937 gen(rng_seed);
  std::shuffle(data.begin(), data.end(), gen);

  const size_t m = data.size();
  const size_t test_n = static_cast<size_t>(m / 3.333);
  const size_t train_n = m - test_n;

  TrainTestSplit out;
  out.x_train.reserve(train_n);
  out.y_train.reserve(train_n);
  out.x_test.reserve(test_n);
  out.y_test.reserve(test_n);

  for (size_t i = 0; i < train_n; ++i) {
    const auto &row = data[i];
    out.y_train.push_back(row[0]);
    std::vector<float> x(row.begin() + 1, row.end());
    for (float &v : x)
      v /= 255.f;
    out.x_train.push_back(std::move(x));
  }
  for (size_t i = train_n; i < m; ++i) {
    const auto &row = data[i];
    out.y_test.push_back(row[0]);
    std::vector<float> x(row.begin() + 1, row.end());
    for (float &v : x)
      v /= 255.f;
    out.x_test.push_back(std::move(x));
  }
  return out;
}





int main(int argc, char **argv) {
  const std::string csv = (argc > 1) ? argv[1] : "../data/data.csv";

  auto data = read_data(csv);
  auto split = prepare_train_test(std::move(data), 42);

  std::cout << "train: " << split.x_train.size() << "  test: " << split.x_test.size()
            << "  features: " << split.x_train[0].size() << '\n';
  return 0;
}
