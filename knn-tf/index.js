require("@tensorflow/tfjs-node");
const tf = require("@tensorflow/tfjs");
const loadCSV = require("./load-csv");

function knn(features, labels, predictionPoint, k) {
  const {mean, variance} = tf.moments(features, 0);
  const scaledPrediction = predictionPoint.sub(mean).div(variance.pow(0.5));

  return features
    .sub(mean)
    .div(variance.pow(0.5))
    .sub(scaledPrediction)
    .pow(2)
    .sum(1)
    .pow(0.5)
    .expandDims(1)
    .concat(labels, 1)
    .unstack()
    .sort((a, b) => a.arraySync()[0] > b.arraySync()[0] ? 1 : -1)
    .slice(0, k)
    .reduce((acc, obj) => {
      return acc + obj.arraySync()[1];
    }, 0) / k;
}

let {features, labels, testFeatures, testLabels} = loadCSV("kc_house_data.csv", {
  shuffle: true,
  splitTest: 10,
  dataColumns: ["lat", "long", "sqft_living", "sqft_lot"],
  labelColumns: ["price"]
});

features = tf.tensor(features);
labels = tf.tensor(labels);

testFeatures.forEach((testFeature, index) => {
  const result = knn(features, labels, tf.tensor(testFeature), 10);
  const errVar = (((testLabels[index][0] - result) / testLabels[index][0]) * 100).toFixed(2);
  console.log(`Guess: ${result}\t \tActual: ${testLabels[index][0]}\t \tOff: ${errVar}%`);
})
