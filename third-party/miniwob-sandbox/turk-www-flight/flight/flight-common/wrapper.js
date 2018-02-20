// ################################################
// Additional methods for flight domain

core.flightChildWindow = function () {
  return document.getElementById('wrap').contentWindow;
}

window.addEventListener('load', function () {
  var correctLocation = null;

  document.getElementById('wrap').addEventListener('load', function (event) {
    var currentLocation = this.contentWindow.location.href;
    if (correctLocation === null) {
      correctLocation = currentLocation;
    } else if (correctLocation != currentLocation) {
      if (core.EP_TIMER !== null) {
        document.getElementById('reward-reason').innerHTML = (
          '<b>BAD:</b> Navigated away to a different web page');
      }
      core.endEpisode(-1, false, 'BAD: Navigated away to a different web page');
      return;
    }
    if (recorder.isRecording) {
      // Tell the child to register event listeners
      core.flightChildWindow().$miniwob.recorder.setup();
    }
  });

});

// ################################################
// Dataset and validation

// The child page should call this method on form submit
core.flightSubmit = function (data, recorder_data) {
  var reward = core.validateForm(data);
  var reason = document.getElementById('reward-reason').textContent;
  core.endEpisode(reward, false, reason, recorder_data);
}

/**
  Overrides genProblem.
  Set core.currentQuestion as a sampled problem. Also starts the task.
  Does not return anything.
  */
var genProblem = function () {
  core.currentQuestion = core.sampleQuestion();
  var instruction = core.currentQuestion.instruction;
  document.getElementById('query').textContent = JSON.stringify(instruction);
  var queryPretty = [];
  Object.keys(instruction).forEach(function (key) {
    queryPretty.push('<tr><th>' + key + '</th><td>' + instruction[key] + '</td>');
  });
  document.getElementById('query-pretty').innerHTML = (
      '<div class=mode>Mode: ' + WOB_DATA_MODE + '</div>' +
      '<table>' + queryPretty.join('') + '</table>');
  document.getElementById('reward-reason').innerHTML = '';
  WOB_TASK_READY = false;   // The child page must set this to true
  document.getElementById('wrap').src = 'index.html';
}

/**
  Return an object that looks like this:
  {
    "instruction": {"key1", "value1", ...}
    "request": {"key1", "value1", ...}
  }
*/
core.sampleQuestion = function () {
  if (WOB_DATA_MODE == 'train' || WOB_DATA_MODE == 'default')
    return core.sample(DATA_TRAIN);
  else if (WOB_DATA_MODE == 'test')
    return core.sample(DATA_TEST);
  else
    throw 'Incorrect WOB_DATA_MODE';
}

// List of required fields
// The reward is -1 if any of these fields is not filled
core.requiredFields = [];

/**
  Validate the form and return the reward.
  data format: list of [tag, type, name, value]
*/
core.validateForm = function(data) {
  // Convert to a dict
  var dataDict = {};
  data.forEach(function (datum) {
    dataDict[datum[2]] = datum[3];
  });
  // Compute accuracy 
  var target = core.currentQuestion.request;
  var score = 0., n = 0., wrongFields = [];
  for (var key in target) {
    n++;
    var expected = target[key], predicted = dataDict[key],
        check = (expected == predicted);
    console.log([check, key, expected, predicted]);
    if (!check) {
      wrongFields.push(
          '<b>' + key + '</b><br>Correct: ' + expected + '<br>Entered: ' + predicted);
    }
    score += check;
  }
  // Validate the required fields
  if (!core.validateRequiredFields(dataDict)) return -1;
  // Display reasons
  if (score == n) {
    document.getElementById('reward-reason').innerHTML = '<b>GOOD</b>';
  } else {
    document.getElementById('reward-reason').innerHTML = (
        '<b>PARTIAL:</b> Incorrect fields:<br>' + wrongFields.join('<br>'));
  }
  return score / n;
}

core.validateRequiredFields = function(dataDict) {
  for (var i = 0; i < core.requiredFields.length; i++) {
    var key = core.requiredFields[i];
    if (!(dataDict[key] || '').length) {
      console.log(['missing required field', key]);
      document.getElementById('reward-reason').innerHTML = (
        '<b>BAD:</b> Missing required field ' + key);
      return false;
    }
  }
  return true;
}

// ################################################
// Function overrides (delegate to the iframe)

core.getDOMInfo = function () {
  return core.flightChildWindow().$miniwob.getDOMInfo();
}

core.elementClick = function (ref) {
  return core.flightChildWindow().$miniwob.elementClick(ref);
}
