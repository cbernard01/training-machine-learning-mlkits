require("@tensorflow/tfjs-node");
const tf = require("@tensorflow/tfjs");
const loadCSV = require("./load-csv");

const LinearRegression = require("./linear-regression");

let {features, labels, testFeatures, testLabels} = loadCSV("cars.csv", {
  shuffle: true,
  splitTest: 50,
  dataColumns: ["horsepower", "weight", "displacement"],
  labelColumns: ["mpg"]
});

const regression = new LinearRegression(features, labels, testFeatures, testLabels, {
  learningRate: 0.1,
  iterations: 3,
  batchSize: 10
});

regression.train();
regression.test();
regression.predict([["120", "2","380"]]).print();
