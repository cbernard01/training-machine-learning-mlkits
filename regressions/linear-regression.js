const tf = require("@tensorflow/tfjs");

class LinearRegression {
  constructor(features, labels, options) {
    this.features = features;
    this.labels = labels;
    this.options = Object.assign(
      {learningRate: 0.1, iterations: 1000},
      options
    );

    this.m = 0;
    this.b = 0;
  }

  gradientDescent() {
    const currentGuessesForMPG = this.features.map(row => (this.m * row[0]) + this.b);
    const mSlope = (_.sum(currentGuessesForMPG.map((guess, i) => -1 * this.features[i][0] * (this.labels[i][0] - guess))) * 2) / this.features.length;
    const bSlope = (_.sum(currentGuessesForMPG.map((guess, i) => guess - this.labels[i][0])) * 2) / this.features.length;

    this.m = this.m - mSlope * this.options.learningRate;
    this.b = this.b - bSlope * this.options.learningRate;
  }

  train() {
    for (let i = 0; i < this.options.iterations; i++) {
      this.gradientDescent();
      console.log("Updated M:", this.m, "Updated B:", this.b);
    }
  }
}

module.exports = LinearRegression;
