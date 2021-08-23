const webcamElement = document.getElementById("webcam");

const initCamera = () => {
  if (webcamElement) {
    navigator.mediaDevices
      .enumerateDevices()
      .then(function (devices) {
        devices.forEach(function (device) {
          // console.log(
          //   device.kind + ": " + device.label + " id = " + device.groupId
          // );
        });
      })
      .catch(function (e) {
        console.log(e.name + ": " + e.message);
      });
  }
  const constraints = {
    video: {
      facingMode: "environment",
    },
    audio: false,
  };

  navigator.mediaDevices.getUserMedia(constraints).then(function (stream) {
    webcamElement.srcObject = stream;
  });
};

initCamera();
