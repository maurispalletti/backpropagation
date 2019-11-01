const express = require('express');
const router = express.Router();

const fs = require('fs')

// External library used to export csv as required
const ExportToCsv = require('export-to-csv').ExportToCsv;

// we decided to transform csv to json just to
// have a faster execution in js
const X_train = require('../csv/X_train.json')
const X_test = require('../csv/X_test.json')
const Y_train = require('../csv/Y_train.json')

router.get('/', (req, res, next) => {

  // To run backpropagation algorithm
  backpropagation();

  // To test weights with trained data
  // testWithTrainData();

  // To get output after network training
  // calculateOutput();

  res.send('Backpropagation algorithm executed!');
});

// CODE TO SEND STARTING HERE:

// Our code is separed in several functions to be more readable

// Input as array
xIn = X_train;

// Expected output as array
yOut = Y_train;

// learning speed
step = 0.003;

// Input layer neurons
n1 = 5;
// Hidden layer neurons
n2 = 6;
// Output layer neurons
n3 = 1;

// Forward input-hidden layer weights (initialized with random values)
w12 = Array(n2).fill(0).map(line => Array(n1).fill(Math.random()));

// Backward hidden-input layer weights 
dw12 = Array(n2).fill(0).map(line => Array(n1).fill(0));

// Forward hidden layer thresholds (initialized with random values)
thrsh2 = Array(n2).fill(Math.random());

// Backward hidden layer thresholds
dthrsh2 = Array(n2).fill(0);

// Forward hidden-output layer weights (initialized with random values)
w23 = Array(n3).fill(0).map(line => Array(n2).fill(Math.random()));

// Backward output-hidden layer weights
dw23 = Array(n3).fill(0).map(line => Array(n2).fill(0));

// Forward output layer thresholds (initialized with random values)
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

// Sigmoid function
const sigmoid = x => 1 / (1 + Math.exp(-x));

// Sigmoid derivated
const sigmoidDeriv = x => x * (1 - x);

// Calculates the error between the obtained results and the expected results
const errorFunction = index => {
  let err = 0;
  for (let o = 0; o < n3; o++) {
    expects[o] = yOut[index];
  }
  for (let o = 0; o < n3; o++) {
    err += Math.pow(out3[0] - expects[0], 2);
  }
  return err;
};

// Runs the forward phase of Backpropagation for an input
const forwardPhase = input => {
  // The input layer outputs are equal to their inputs
  for (let i = 0; i < n1; i++) {
    out1[i] = input[i];
  }
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

// Runs the backward phase of Backpropagation for an input
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

// Runs the backpropagation algorithm
const backpropagation = () => {
  let error, logStream = fs.createWriteStream('backpropagationResults.txt', { flags: 'a' });

  let count = 0;
  let correct = 0;
  let missed = 0;
  let accuracy = 0;

  // Repeat do while cicle until accuracy > 0.87
  do {
    // Reset the global error and other variables for each 
    // training iteration
    error = 0;
    correct = 0;
    missed = 0;
    accuracy = 0;

    // First 1500 items as input, leaving 500 more to test later
    for (let index = 0; index < 1500; index++) {
      // For each iteration, get input and expected output to calculate
      const input = xIn[index];
      const output = yOut[index];

      // Propagate the signal forward
      forwardPhase(input);

      // As explained in the description, we determinate this way if
      // obteined output is acceptable or not
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

    // Calculate accuracy for each iteration
    accuracy = ((correct / 1500) * 100).toFixed(2);

    // Print data in console after each iteration
    console.log(`accuracy: ${accuracy}% || Turn n.: ${count} || missed: ${missed} || correct: ${correct}`);

    count++;
    // Break condition is to have accuracy higher than 87%.
  } while (accuracy < 89);

  // Once finished with expected accuracy, print results in file
  logStream.write(`!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n`);
  logStream.write(`Error: ${error}\n`);
  logStream.write(`w12: ${JSON.stringify(w12)}\n`);
  logStream.write(`w23: ${JSON.stringify(w23)}\n`);
  logStream.write(`thrsh2: ${thrsh2}\n`);
  logStream.write(`thrsh3: ${thrsh3}\n`);
  logStream.write(`!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n`);

  console.log(`!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!`);
  console.log(`Accuracy reached! Network trained.`);
  console.log(`!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!`);
};

//////////////////////////////////////////////////////////////////////////

// Runs the forward phase of Backpropagation after network is trained.
const forwardPhaseResults = input => {
  let nOut1 = [];
  let nOut2 = [];
  let nOut3 = [];


  // Using best obtained weights in previous run with > 0.87 accuracy
  const nW12 = [[-2.428763904673985, -2.380828415703536, -2.5908327858379874, -2.329533152678648, 7.637655167930898], [0.39576049339366276, 0.4684634022976463, 0.47696001840541735, 0.45754411838501313, 0.4854333136148282], [-0.7994014270030201, -0.8782241475508317, -0.8507243843381835, -0.7288110402649293, -3.0464093284675515], [-2.2536004844311237, -2.1132765708062187, -2.1106981201156776, -2.2960330340458217, 10.185571686885647], [0.9045022186205917, 0.9101001483070812, 0.9070852792888664, 0.9068081253688645, 0.9074715790889041], [0.9567327289426886, 0.9609550486157794, 0.9583283756043494, 0.9583608719789461, 0.9587577046565169]];
  const nW23 = [[5.121999473069013, 1.141890079358862, -11.099412392303128, -5.39283617547238, 0.9939959883058369, 0.9893638694177811]];

  const nThrsh2 = [5.55325385137848, 0.4041942083062008, 25.073821076114864, -0.16307796277881692, 0.5032399358297514, 0.5052512711203313];
  const nThrsh3 = [0.89000081049251];

  // Calculate output with given weights:
  for (let i = 0; i < n1; i++) {
    nOut1[i] = input[i];
  }
  // Input-Hidden propagation
  for (let h = 0; h < n2; h++) {
    let sum = 0;
    for (let i = 0; i < n1; i++) {
      sum += nW12[h][i] * nOut1[i];
    }
    nOut2[h] = sigmoid(sum + nThrsh2[h]);
  }
  // Hidden-Ouput propagation
  for (let o = 0; o < n3; o++) {
    let sum = 0;
    for (let h = 0; h < n2; h++) {
      sum += nW23[o][h] * nOut2[h];
    }
    nOut3[o] = sigmoid(sum + nThrsh3[o]);
  }

  return nOut3[0];
};

//////////////////////////////////////////////////////////////////////////

// Function to test weights and algorithm with 
// 500 inputs saved from train
const testWithTrainData = () => {
  let correct = 0;
  let missed = 0;
  let accuracy = 0;

  let newOutput = -1;

  // Last 500 inputs saved to test here
  for (let index = 1500; index < 2000; index++) {
    const input = xIn[index];
    const expectedOutput = yOut[index];

    // Propagate the signal forward
    let output = forwardPhaseResults(input);

    // In this case, we decided to be more permisive and divide
    // between values higher and lower than 0.5 to choose to be a
    // 1 or 0 output
    if (output > 0.5) {
      newOutput = 1;
    } else {
      newOutput = 0;
    }

    // Print output in console
    console.log(`output: ${output} || modified: ${newOutput} || expected: ${expectedOutput}`);

    if (newOutput === expectedOutput) {
      correct++;
    } else {
      missed++;
    }
  };

  // Calculate and print accuracy for this set of data
  accuracy = ((correct / 500) * 100).toFixed(2);

  // After finishing iterations, print obteined results
  console.log(`!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!`);
  console.log(`accuracy: ${accuracy}% || missed: ${missed} || correct: ${correct}`);
  console.log(`!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!`);
};

//////////////////////////////////////////////////////////////////////////

// Runs the backpropagation algorithm
const calculateOutput = () => {
  // Data to be printed in csv
  let dataToCSV = [];
  let newOutput = -1;

  // Length equivalent to input test length
  for (let index = 0; index < 2000; index++) {
    const input = X_test[index];

    // Propagate the signal forward
    let output = forwardPhaseResults(input);

    // In this case, we decided to be more permisive and divide
    // between values higher and lower than 0.5 to choose to be a
    // 1 or 0 output
    if (output > 0.5) {
      newOutput = 1;
    } else {
      newOutput = 0;
    }

    // Printing data in console
    console.log(`output: ${output} || modified: ${newOutput}`);

    // Adding result from each iteration to csv to be printed
    dataToCSV.push({ newOutput });
  };

  const options = {
    fieldSeparator: ',',
    quoteStrings: '"',
    decimalSeparator: '.',
    showLabels: true,
    showTitle: true,
    useTextFile: false,
    useBom: true,
    useKeysAsHeaders: true,
  };

  const exportToCsv = new ExportToCsv(options);
  const csvData = exportToCsv.generateCsv(dataToCSV, true)

  // Writing result in csv file
  fs.writeFileSync('Y_test.csv', csvData)

  console.log(`Execution ended. Data saved in Y_test.csv file.`);
};

module.exports = router;

