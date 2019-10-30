const express = require('express');
const router = express.Router();

const X_train = require('../csv/X_train.json')
const Y_train = require('../csv/Y_train.json')

let fs = require('fs')

/* GET users listing. */
router.get('/', function (req, res, next) {

  // calculateOutput();

  // backpropagationOnline();
  backpropagationOffline();

  res.send('This is the algorithm');
});

// resultados:
// 1.
// step = 0.0001;
// n2 = 3;
// vueltas = + 500 000
// res = 438

// 2
// step = 0.003;
// n2 = 3;
// vueltas = + 500 000
// res = 422

// 3 BEST SO FARRRRRRRR    GANADORRRRRR
// step = 0.003;
// n2 = 6;
// vueltas = + 500 000
// res = 85% ACC

xIn = X_train;
yOut = Y_train;

// Samples
N = 300000;
// N = 1500;

// Step size toward gradient - The learning speed
step = 0.001;

// Input layer neurons
n1 = 5;

// Hidden layer neurons
n2 = 6;

// Output layer neurons
n3 = 1;

// Forward input-hidden layer weights
w12 = Array(n2).fill(0).map(line => Array(n1).fill(Math.random()));

// Backward hidden-input layer weights
dw12 = Array(n2).fill(0).map(line => Array(n1).fill(0));

// Forward hidden layer thresholds
thrsh2 = Array(n2).fill(Math.random());

// Backward hidden layer thresholds
dthrsh2 = Array(n2).fill(0);

// Forward hidden-output layer weights
w23 = Array(n3).fill(0).map(line => Array(n2).fill(Math.random()));

// Backward output-hidden layer weights
dw23 = Array(n3).fill(0).map(line => Array(n2).fill(0));

// Forward output layer thresholds
thrsh3 = Array(n3).fill(Math.random());

// Backward output layer thresholds
dthrsh3 = Array(n3).fill(0);

// Input layer outputs
out1 = Array(n1).fill(0);

// Hidden layer outputs
out2 = Array(n2).fill(0);

// Output layer outputs
out3 = Array(n3).fill(0);

// Expected outputs
expects = Array(n3).fill(0);

const sigmoid = x => 1 / (1 + Math.exp(-x));

const sigmoidDeriv = x => x * (1 - x);

// Calculates the error between the obtained results and the expected results
const errorFunction = (index) => {
  let err = 0;
  for (let o = 0; o < n3; o++) {
    expects[o] = yOut[index];
  }

  for (let o = 0; o < n3; o++) {
    err += Math.pow(out3[0] - expects[0], 2);
  }

  return err;
};

// Suffles the sequence of samples
const shuffleSequence = () => {
  for (let index = 0; index < 700; index++) {
    let a = Math.round(Math.random() * 2000),
      b = Math.round(Math.random() * 2000),
      tempX = xIn[a];
      xIn[a] = xIn[b];
      xIn[b] = tempX;

      tempY = yOut[a];
      yOut[a] = yOut[b];
      yOut[b] = tempY;
  }
};

// Runs the forward phase of Backpropagation for a specific sample
const forwardPhase = input => {
  // The input layer outputs are equal to their inputs
  for (let i = 0; i < n1; i++) {
    out1[i] = input[i];
  }

  // Propagate the signal from the input layer to the output layer
  // Input-Hidden propagation
  for (let h = 0; h < n2; h++) {
    let sum = 0;
    for (let i = 0; i < n1; i++) {
      sum += w12[h][i] * out1[i];
    }
    out2[h] = sigmoid(sum + thrsh2[h]);
  }

  // Hidden-Ouput propagation
  for (let o = 0; o < n3; o++) {
    let sum = 0;
    for (let h = 0; h < n2; h++) {
      sum += w23[o][h] * out2[h];
    }
    out3[o] = sigmoid(sum + thrsh3[o]);
  }
};

// Runs the backward phase of Backpropagation for a specific sample
const backwardPhase = () => {
  // Propagate the signal from the output layer to the input layer
  // Calculate backward thresholds of output layer
  for (let o = 0; o < n3; o++) {
    dthrsh3[o] += 2 * (out3[o] - expects[o]) * sigmoidDeriv(out3[o]);
  }

  // Output-hidden backward propagation
  for (let o = 0; o < n3; o++) {
    for (let h = 0; h < n2; h++) {
      dw23[o][h] += 2 * (out3[o] - expects[o]) * sigmoidDeriv(out3[o]) * out2[h];
    }
  }

  // Calculate backward thresholds of hidden layer
  for (let h = 0; h < n2; h++) {
    for (let o = 0; o < n3; o++) {
      dthrsh2[h] += 2 * (out3[o] - expects[o]) * sigmoidDeriv(out3[o]) * w23[o][h] * sigmoidDeriv(out2[h]);
    }
  }

  // Hidden-input backward propagation
  for (let h = 0; h < n2; h++) {
    for (let i = 0; i < n1; i++) {
      for (let o = 0; o < n3; o++) {
        dw12[h][i] += 2 * (out3[o] - expects[o]) * sigmoidDeriv(out3[o]) * w23[o][h] * sigmoidDeriv(out2[h]) * out1[i];
      }
    }
  }
};

// Runs the weight update phase of Backpropagation algorithm for forward propagation
const updateForwardWeights = () => {
  // Update forward weights and thresholds
  // Update forward input-hidden weights
  for (let i = 0; i < n1; i++) {
    for (let h = 0; h < n2; h++) {
      w12[h][i] -= step * dw12[h][i];
    }
  }

  // Update forward hidden-output weights
  for (let o = 0; o < n3; o++) {
    for (let h = 0; h < n2; h++) {
      w23[o][h] -= step * dw23[o][h];
    }
  }

  // Update forward hidden layer thresholds
  for (let h = 0; h < n2; h++) {
    thrsh2[h] -= step * dthrsh2[h];
  }

  // Update forward output layer thresholds
  for (let o = 0; o < n3; o++) {
    thrsh3[o] -= step * dthrsh3[o];
  }
};

// Runs the weight update phase of Backpropagation algorithm for backward propagation
const updateBackwardWeights = () => {
  // Clear backwward weights and thresholds
  // Clear backward input-hidden weights
  for (let i = 0; i < n1; i++) {
    for (let h = 0; h < n2; h++) {
      dw12[h][i] = 0;
    }
  }

  // Clear backward hidden-output weights
  for (let o = 0; o < n3; o++) {
    for (let h = 0; h < n2; h++) {
      dw23[o][h] = 0;
    }
  }

  // Clear backward hidden layer thresholds
  for (let h = 0; h < n2; h++) {
    dthrsh2[h] = 0;
  }

  // Clear backward output layer thresholds
  for (let o = 0; o < n3; o++) {
    dthrsh3[o] = 0;
  }
};

// Runs the backpropagation algorithm in offline mode
const backpropagationOffline = () => {
  let error, logStream = fs.createWriteStream('backOffline.txt', { flags: 'a' });

  let count = 0;
  let correct = 0;
  let missed = 0;
  let acc = 0;

  do {
    // Reset the global error
    error = 0;

    // Shuffle the sequence
    // shuffleSequence();

    correct = 0;
    missed = 0;
    acc = 0;

    // logStream.write(`Turn n: ${count}\n`);

    for (let index = 0; index < 1500; index++) {
      const input = xIn[index];
      const output = yOut[index];

      // Propagate the signal forward
      forwardPhase(input);

      if (output === 1 && out3[0] > 0.8 || output === 0 && out3[0] < 0.2) {
        correct++;
      } else {
        missed++;
      }

      // Establish the output error
      error += errorFunction(index);
      // Propagate the signal backward
      backwardPhase();
    };

    // Update the weights and thresholds
    updateForwardWeights();
    updateBackwardWeights();

    acc = ((correct / 1500) * 100).toFixed(2);

    // console.log(`missed: ${missed} || correct: ${correct} || Vuelta nro: ${count}`);
    console.log(`accuracy: ${acc}% || Turn n.: ${count} || missed: ${missed} || correct: ${correct}`);

    count++;

    // Log the learning error
    // logStream.write(`Error: ${error}\n`);

  } while (acc < 87);

  logStream.write(`!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n`);
  logStream.write(`Error: ${error}\n`);
  logStream.write(`Out1: ${out1}\n`);
  logStream.write(`Out2: ${out2}\n`);
  logStream.write(`Out3: ${out3}\n`);
  logStream.write(`!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n`);

  console.log(`!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!`);
  console.log(`!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!`);
  console.log(`!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!`);
  console.log(`!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!`);
  console.log(`!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!`);
  console.log(`!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!`);
  console.log(`!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!`);
  console.log(`!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!`);
  console.log(`!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!`);
  console.log(`!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!`);
  console.log(`!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!`);
  console.log(`!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!`);
  console.log(`!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!`);
  console.log(`!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!`);
  console.log(`!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!`);
  console.log(`!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!`);
  console.log(`!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!`);
  console.log(`!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!`);
  console.log(`!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!`);
  console.log(`!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!`);
  console.log(`!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!`);
  console.log(`!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!`);
  console.log(`!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!`);
  console.log(`!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!`);
  console.log(`!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!`);
  console.log(`!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!`);

};







module.exports = router;

