// training
const classifier = knnClassifier.create();

const trainButton = document.querySelector(".train-button");
const saveModelButton = document.querySelector(".save-model");
const quoteDetection = document.querySelector('.quote-detection');
const trainFeedback = document.querySelector('.train-feedback');
const neutralizeButton = document.querySelector('.neutralize');
const instagramLink = document.querySelector('.instagram-link');

const TRAINING_STEPS_MAX = 20; // change to 10

let trainingInterval;
let isTraining = false;
let quoteIndex = 0;
let trainingStep = 0;

const quoteMapping = {};

const quotes = ['aaa','bbb','ccc'];

const loadWeights = (src) => {
  const dataset = localStorage.getItem("myData") // the stringified weights
  let tensorObj = JSON.parse(dataset)
  Object.keys(tensorObj).forEach((key) => {
    tensorObj[key] = tf.tensor(tensorObj[key], [tensorObj[key].length / 1000, 1000])
  })
  classifier.setClassifierDataset(tensorObj);
}

function download(content, fileName, contentType) {
  var a = document.createElement("a");
  var file = new Blob([content], {type: contentType});
  a.href = URL.createObjectURL(file);
  a.download = fileName;
  a.click();
}

const saveWeights = () => {
  let dataset = classifier.getClassifierDataset()
  var datasetObj = {}
  Object.keys(dataset).forEach((key) => {
    let data = dataset[key].dataSync();
    datasetObj[key] = Array.from(data); 
  });
  let jsonStr = JSON.stringify(datasetObj)
  download(jsonStr, 'pipoKNjson.txt', 'text/plain');

  console.log(jsonStr)
    //can be change to other source
    // localStorage.setItem("myData", jsonStr);
}

async function app() {
  net = await mobilenet.load();
  const webcam = await tf.data.webcam(webcamElement);

  initCamera();
  // train
  const addExample = async (classId) => {
    const img = await webcam.capture();
    const activation = net.infer(img, true); // before : img insyead of webcamElement
    classifier.addExample(activation, classId);
    img.dispose();
  };

  const keepTraining = () => {
    trainingInterval = setInterval(() => {
      if (trainingStep < TRAINING_STEPS_MAX) {
        trainingStep += 1;
        isTraining = true;
        trainStep(trainingStep);
      }
      if (trainingStep === TRAINING_STEPS_MAX && isTraining) {
        quoteMapping[quoteIndex] = quotes[quoteIndex];
        quoteIndex += 1
        isTraining = false;
        console.log(quoteMapping)

        trainFeedback.style.top = 100 + "%";
      }
    }, 800);
  };
  
  const stopTraining = () => {
    clearInterval(trainingInterval);
    isTraining = false;
    trainFeedback.style.top = 100 + "%";
    trainingStep = 0;
  };

  const neutralize = () => {
    addExample(quotes.length + 1);
    quoteMapping[quotes.length + 1] = "";
  }
  trainButton.addEventListener("touchstart", keepTraining);
  trainButton.addEventListener("touchend", stopTraining);
  // trainButton.addEventListener("mousedown", keepTraining);
  // trainButton.addEventListener("mouseup", stopTraining);
  
  neutralizeButton.addEventListener('click', neutralize);
  saveModelButton.addEventListener("click", saveWeights);


  const trainStep = (n) => {
    trainFeedback.style.top = 100 - (n+1) * (100 / TRAINING_STEPS_MAX) + "%";
    console.log(n)
    addExample(quoteIndex);
  };

  let ones = [];

  while (true) {
    if (classifier.getNumClasses() > 0) {
      const img = await webcam.capture();
      const activation = net.infer(img, "conv_preds");
      const result = await classifier.predictClass(activation);
      const classes = Object.values(quoteMapping);

      if (classes[result.label] && result.confidences[result.label] === 1) {
        ones.push(1);
        if (ones.length === 50) {
          quoteDetection.innerHTML = classes[result.label];
          quoteDetection.classList.add('show');
          instagramLink.classList.add('show');
        }
      } else {
        ones = [];
        quoteDetection.classList.remove('show');
      }
      img.dispose();
    }
    await tf.nextFrame();
  }
}

app();
