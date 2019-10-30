// const getCuadraticError = (errors, gen) => {
//   let errorSumatory;

//   errors.forEach(error => {
//     errorSumatory += (error^2)
//   });

//   return (1/2 * gen) * errorSumatory;
// }


// // SEGUNDA VERSION - ADALINE
// const calculateOutput = () => {
//   const amount = 1500;
//   const acceptedError = 0.2;
  
//   const learningRate = 0.03;

//   // initial weights
//   let w = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1];
//   let dw = [0, 0, 0, 0, 0, 0];

//   // number of cicles
//   let gen = 1;
  
//   let cuadraticError = 0;

//   do {

//     let errorsList = [];

//     for (let index = 0; index < amount; index++) {

//       // get input elements from input array 
//       const element = X_train[index];
      
//       // get output element from output array
//       const { 1: expectedOutput } = Y_train[index]

//       // extract data from each object
//       const aa = element[Object.keys(element)[0]];
//       const bb = element[Object.keys(element)[1]];
//       const cc = element[Object.keys(element)[2]];
//       const dd = element[Object.keys(element)[3]];
//       const ee = element[Object.keys(element)[4]];

//       const output = aa * w[0] + bb * w[1] + cc * w[2] + dd * w[3] + ee * w[4] + w[5];

//       const error = expectedOutput - output;

//       errorsList.push(error);

//       let learningRateError = learningRate * error;

//       dw[0] = learningRateError * w[0]
//       dw[1] = learningRateError * w[1]
//       dw[2] = learningRateError * w[2]
//       dw[3] = learningRateError * w[3]
//       dw[4] = learningRateError * w[4]
//       dw[5] = learningRateError * w[5]

//       // update weights
//       w[0] += dw[0]
//       w[1] += dw[1]
//       w[2] += dw[2]
//       w[3] += dw[3]
//       w[4] += dw[4]
//       w[5] += dw[5]
//     }

//     console.log(`!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!`)
//     console.log(`Vuelta nro: ${gen}`)
//     console.log(w)
//     console.log(`!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!`)

//     gen++;

//     cuadraticError = getCuadraticError(errorsList, gen)

//   } while (cuadraticError <= acceptedError);

//   console.log(`!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!`)
//   console.log(`Vuelta finalizada: ${gen}`)
//   console.log(`Error cuadratico: ${cuadraticError}`)
//   console.log(w)
//   console.log(`!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!`)
// }



// //////////////////////////////////////////////////////////////////////////////////////////////////////////////



// // SEGUNDA VERSION - PERCEPTRON 

// const calculateOutputPerceptron = () => {
//   const amount = 1500;

//   let w = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1];

//   const eta = 0.1;

//   let gen = 1;
//   let converged;

//   do {
//     converged = true;

//     for (let index = 0; index < amount; index++) {

//       const element = X_train[index];
//       const { 1: elementOutput } = Y_train[index]

//       // extract data from each object
//       const aa = element[Object.keys(element)[0]];
//       const bb = element[Object.keys(element)[1]];
//       const cc = element[Object.keys(element)[2]];
//       const dd = element[Object.keys(element)[3]];
//       const ee = element[Object.keys(element)[4]];

//       const sum = aa * w[0] + bb * w[1] + cc * w[2] + dd * w[3] + ee * w[4] + w[5];
//       const out = sum > 0 ? 1 : 0;
//       const delta = elementOutput - out;

//       if (delta != 0) {
//         converged = false;

//         w[0] += eta * delta * aa;
//         w[1] += eta * delta * bb;
//         w[2] += eta * delta * cc;
//         w[3] += eta * delta * dd;
//         w[4] += eta * delta * ee;
//         w[5] += eta * delta;
//       }

//     }

//     console.log(`!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!`)
//     console.log(`Vuelta nro: ${gen}`)
//     console.log(w)
//     console.log(`!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!`)

//     gen++;

//   } while (!converged);

//   console.log(`!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!`)
//   console.log(`Vuelta finalizada: ${gen}`)
//   console.log(w)
//   console.log(`!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!`)
// }


// //////////////////////////////////////////////////////////////////////////////////////////////////////////////


// // SEGUNDA VERSION

// const calculateOutput = () => {
//   const amount = 1500;

//   let w = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1];
//   let dw;
//   const eta = 0.05;
//   const epsilon = 1.01;
//   let gen = 1;
//   let error;

//   do {
//     error = 0;
//     dw = [0, 0, 0, 0, 0, 0];

//     for (let index = 0; index < amount; index++) {

//       const element = X_train[index];
//       const { 1: elementOutput } = Y_train[index]

//       // extract data from each object
//       const aa = element[Object.keys(element)[0]];
//       const bb = element[Object.keys(element)[1]];
//       const cc = element[Object.keys(element)[2]];
//       const dd = element[Object.keys(element)[3]];
//       const ee = element[Object.keys(element)[4]];

//       const out = aa * w[0] + bb * w[1] + cc * w[2] + dd * w[3] + ee * w[4];
//       const delta = elementOutput - out;

//       error += delta * delta;

//       dw[0] += eta * delta * aa
//       dw[1] += eta * delta * bb
//       dw[2] += eta * delta * cc
//       dw[3] += eta * delta * dd
//       dw[4] += eta * delta * ee
//       dw[5] += eta * delta
//     }

//     console.log(`!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!`)
//     console.log(`Vuelta nro: ${gen}`)
//     console.log(w)
//     console.log(`!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!`)

//     w[0] += dw[0];
//     w[1] += dw[1];
//     w[2] += dw[2];
//     w[3] += dw[3];
//     w[4] += dw[4];
//     w[5] += dw[5];

//     gen++;

//   } while (error >= epsilon);

//   console.log(`!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!`)
//   console.log(`Vuelta finalizada: ${gen}`)
//   console.log(w)
//   console.log(`!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!`)
// }



// //////////////////////////////////////////////////////////////////////////////////////////////////////////////



// PRIMERA VERSION - PERCEPTRON

// const calculateOutput = () => {

//   // let acceptationCalc = 0;

//   const amount = 1500;

//   const weight = {
//     a1: 0.1,
//     a2: 0.1,
//     a3: 0.1,
//     a4: 0.1,
//     a5: 0.1,
//   }

//   // while (acceptationCalc < (0, 8 * amount)) {
//     // let acceptationCalc = 0;

//     for (let index = 0; index < amount; index++) {

//       const element = X_train[index];

//       console.log(`!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!`);
//       console.log(index);
//       console.log(element);

//       // extract data from each object
//       const aa = element[Object.keys(element)[0]];
//       const bb = element[Object.keys(element)[1]];
//       const cc = element[Object.keys(element)[2]];
//       const dd = element[Object.keys(element)[3]];
//       const ee = element[Object.keys(element)[4]];

//       // const { 1: result } = Y_train[index]
//       // console.log(`Salida: ${JSON.stringify(result)}`)

//     }
//   // }
// }
