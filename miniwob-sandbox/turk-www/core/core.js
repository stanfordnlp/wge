var core = {}

// various common utilities

// seedrandom.min.js -- https://github.com/davidbau/seedrandom
// Usage: Math.seedrandom('hello.'); -- Set the seed
// Usage: Math.seedrandom(); -- Automatically set a random seed
!function(a,b){function c(c,j,k){var n=[];j=1==j?{entropy:!0}:j||{};var s=g(f(j.entropy?[c,i(a)]:null==c?h():c,3),n),t=new d(n),u=function(){for(var a=t.g(m),b=p,c=0;a<q;)a=(a+c)*l,b*=l,c=t.g(1);for(;a>=r;)a/=2,b/=2,c>>>=1;return(a+c)/b};return u.int32=function(){return 0|t.g(4)},u.quick=function(){return t.g(4)/4294967296},u.double=u,g(i(t.S),a),(j.pass||k||function(a,c,d,f){return f&&(f.S&&e(f,t),a.state=function(){return e(t,{})}),d?(b[o]=a,c):a})(u,s,"global"in j?j.global:this==b,j.state)}function d(a){var b,c=a.length,d=this,e=0,f=d.i=d.j=0,g=d.S=[];for(c||(a=[c++]);e<l;)g[e]=e++;for(e=0;e<l;e++)g[e]=g[f=s&f+a[e%c]+(b=g[e])],g[f]=b;(d.g=function(a){for(var b,c=0,e=d.i,f=d.j,g=d.S;a--;)b=g[e=s&e+1],c=c*l+g[s&(g[e]=g[f=s&f+b])+(g[f]=b)];return d.i=e,d.j=f,c})(l)}function e(a,b){return b.i=a.i,b.j=a.j,b.S=a.S.slice(),b}function f(a,b){var c,d=[],e=typeof a;if(b&&"object"==e)for(c in a)try{d.push(f(a[c],b-1))}catch(a){}return d.length?d:"string"==e?a:a+"\0"}function g(a,b){for(var c,d=a+"",e=0;e<d.length;)b[s&e]=s&(c^=19*b[s&e])+d.charCodeAt(e++);return i(b)}function h(){try{var b;return j&&(b=j.randomBytes)?b=b(l):(b=new Uint8Array(l),(k.crypto||k.msCrypto).getRandomValues(b)),i(b)}catch(b){var c=k.navigator,d=c&&c.plugins;return[+new Date,k,d,k.screen,i(a)]}}function i(a){return String.fromCharCode.apply(0,a)}var j,k=this,l=256,m=6,n=52,o="random",p=b.pow(l,m),q=b.pow(2,n),r=2*q,s=l-1;if(b["seed"+o]=c,g(b.random(),a),"object"==typeof module&&module.exports){module.exports=c;try{j=require("crypto")}catch(a){}}else"function"==typeof define&&define.amd&&define(function(){return c})}([],Math);

core.randi = function(min, max) {
  return Math.floor(Math.random()*(max-min)+min);
}

core.randf = function(min, max) {
  return Math.random()*(max-min)+min;
}

core.sample = function(lst) {
  var ix = core.randi(0,lst.length);
  return lst[ix];
}

// https://stackoverflow.com/questions/2450954/how-to-randomize-shuffle-a-javascript-array
core.shuffle = function(array) {
  var currentIndex = array.length, temporaryValue, randomIndex;

  // While there remain elements to shuffle...
  while (0 !== currentIndex) {

    // Pick a remaining element...
    randomIndex = Math.floor(Math.random() * currentIndex);
    currentIndex -= 1;

    // And swap it with the current element.
    temporaryValue = array[currentIndex];
    array[currentIndex] = array[randomIndex];
    array[randomIndex] = temporaryValue;
  }

  return array;
}

// utilities for timing episodes
var WOB_REWARD_GLOBAL = 0; // what was reward in previous iteration?
var WOB_RAW_REWARD_GLOBAL = 0; // reward without time penalty
var WOB_DATA_MODE = 'default';
var WOB_DONE_GLOBAL = false; // a done indicator
var WOB_EPISODE_ID = 0; // number of episodes done so far
core.EPISODE_MAX_TIME = 10000; // in ms. Set default time to 10s.
core.EPISODES_NEEDED = 5;

// https://stackoverflow.com/questions/3169786/clear-text-selection-with-javascript
// this piece of code clears the selection in a new episode, if a user happened
// to select some part of text. We don't want this to persist across episodes
var clearUserSelection = function() {
  if (window.getSelection) {
    if (window.getSelection().empty) {  // Chrome
      window.getSelection().empty();
    } else if (window.getSelection().removeAllRanges) {  // Firefox
      window.getSelection().removeAllRanges();
    }
  } else if (document.selection) {  // IE?
    document.selection.empty();
  }
}

core.EP_TIMER = null; // stores timer id
core.CD_TIMER = null; // stores timer ID for displaying rewards
core.ept0; // stores system time when episode begins (so we can time it)
core.cover_div = null; // cover div for synchronization
core.submitClicked = false;

core.startEpisode = function() {
  core.createDisplay();
  if (core.cover_div == null) {
    // Initialize things
    core.cover_div = document.createElement('div');
    core.cover_div.setAttribute('id', 'sync-task-cover');
    core.cover_div.innerHTML = 'START';
    core.cover_div.onclick = function () {
      core.startEpisodeReal();
    };
    document.body.appendChild(core.cover_div);
    // Prevent form submission until the quota is met
    [].forEach.call(document.getElementsByClassName('episodes-needed'),
      function (x) {x.innerHTML = core.EPISODES_NEEDED;});
    document.getElementById('mturk_form').onsubmit = function (event) {
      if (+core.wob_scores < core.EPISODES_NEEDED || !core.submitClicked) {
        return false;
      }
    };
    window.addEventListener('keydown', function (event) {
      if (event.keyCode == 13) {
        event.preventDefault();
        return false;
      }
    });
    // Preview mode?
    core.checkPreviewMode();
  }
  core.cover_div.style.display = 'block';
}

core.startEpisodeReal = function () {
  resetRefCode();
  genProblem();
  WOB_DONE_GLOBAL = false;
  WOB_REWARD_GLOBAL = 0;
  WOB_RAW_REWARD_GLOBAL = 0;
  clearUserSelection();
  core.cover_div.style.display = 'none';
  core.ept0 = new Date().getTime();
  core.countdownTimer(core.EPISODE_MAX_TIME);
  // start an end of episode timer
  if(core.EP_TIMER !== null) { clearTimeout(core.EP_TIMER); } // reset timer if needed
  core.EP_TIMER = setTimeout(function(){
    core.endEpisode(-1); // time ran out
  }, core.EPISODE_MAX_TIME);
}

core.endEpisode = function(reward, time_proportional) {
  // stop timer and set to null, so that only one event gets rewarded
  // for any given episode.
  if(core.EP_TIMER !== null) {
    clearTimeout(core.EP_TIMER);
    core.EP_TIMER = null;
  } else {
    // if timer is null, don't reward anything and exit out.
    return;
  }

  WOB_RAW_REWARD_GLOBAL = reward;

  // adjust reward based on time, so acting early is encouraged
  var ept1 = new Date().getTime(); // get system time
  if(typeof time_proportional === 'undefined') { time_proportional = false; }
  if(time_proportional) {
    var dt = ept1 - core.ept0; // difference in ms since start of ep
    reward = reward * Math.max(0, 1.0 - dt/core.EPISODE_MAX_TIME);
  }

  WOB_REWARD_GLOBAL = reward; // add to global, to be accessed from Python
  WOB_DONE_GLOBAL = true;
  WOB_EPISODE_ID++;
  console.log('reward: ' + WOB_REWARD_GLOBAL + ' (raw: ' + WOB_RAW_REWARD_GLOBAL + ')');
  core.updateDisplay(WOB_RAW_REWARD_GLOBAL == 1);
  core.clearTimer();
  core.startEpisode();
}

// returns parameters passed in the url.
// e.g. ?topic=123&name=query+string in the url would return
// QueryString["topic"];    // 123
// QueryString["name"];     // query string
// QueryString["nothere"];  // undefined (object)
core.QueryString = (function(a) {
  if (a == "") return {};
  var b = {};
  for (var i = 0; i < a.length; ++i)
  {
    var p=a[i].split('=', 2);
    if (p.length == 1)
      b[p[0]] = "";
    else
      b[p[0]] = decodeURIComponent(p[1].replace(/\+/g, " "));
  }
  return b;
})(window.location.search.substr(1).split('&'));

core.getTaskName = function () {
  var url = window.location.pathname;
  return url.substr(url.lastIndexOf('/') + 1).replace(/\.html/, '');
}

core.getOpt = function(d, k, def) {
  var v = d[k]
  return typeof v === 'undefined' ? def : v;
}

core.DISPLAY_HTML = `
  <div class="info">
    <label>Time left:</label>
    <span id='timer-countdown'>-</span>
  </div>
  <div class="info">
    <label>Last task:</label>
    <span id='reward-last'>-</span>
  </div>
  <div class="info">
    <label>Successes:</label>
    <span id='reward-total'>0</span> / <span class='episodes-needed'>5</span>
  </div>
  <div class="info" id="preview-mode-wrapper">
    <strong>Preview mode.</strong>
    You must ACCEPT the HIT before you can submit the results. 
  </div>
`;

core.INSTRUCTION_HTML = `
  <h1>Instructions</h1>
  <p>Please perform the specified user interface tasks within the time limit.</p>
  <p>You can submit the HIT after successfully completing <span class='episodes-needed'>5</span> tasks.</p>
  <p>Note: The tasks were designed for <strong>Google Chrome on desktop</strong> and will not work correctly on other browsers.</p>
`;

core.MTURK_FORM_HTML = `
<input name="assignmentId" id="assignmentId" type="hidden">
<input name="task" id="mturkTaskName" type="hidden">
`;

// create element via JS; appending the HTML template
// directly to the body will cause jQuery UI elements
// to freak out.
core.createDisplay = function(){
  var display = document.getElementById('reward-display');
  if(display === null) {
    // Reward display
    var newDiv = document.createElement('div');
    newDiv.setAttribute('id','reward-display');
    newDiv.innerHTML = core.DISPLAY_HTML;
    document.body.appendChild(newDiv);
    // Instruction
    var newDiv = document.createElement('div');
    newDiv.setAttribute('id','instructions');
    newDiv.innerHTML = core.INSTRUCTION_HTML;
    document.body.appendChild(newDiv);
    // Turk form
    var newForm = document.createElement('form');
    newForm.setAttribute('id','mturk_form');
    newForm.setAttribute('method','POST');
    newForm.innerHTML = core.MTURK_FORM_HTML;
    document.body.appendChild(newForm);
    if ((core.QueryString.turkSubmitTo || '').indexOf('workersandbox') !== -1) {
      // Sandbox mode
      newForm.setAttribute('action', "https://workersandbox.mturk.com/mturk/externalSubmit");
    } else if (core.QueryString.debug === 'true') {
      // Debug mode
      newForm.setAttribute('action', "javascript:alert('debug!')");
    } else {
      // Real mode
      newForm.setAttribute('action', "https://www.mturk.com/mturk/externalSubmit");
    }
    document.getElementById('assignmentId').value = core.QueryString.assignmentId || 'ASSIGNMENT_ID_NOT_AVAILABLE';
    document.getElementById('mturkTaskName').value = core.getTaskName();
  }
  core.updateDisplay();
}

core.updateDisplay = function(reward){
  core.wob_latest = core.wob_latest || '-';
  core.wob_scores = core.wob_scores || 0;
  if (typeof reward !== 'undefined') {
    core.wob_latest = +reward;
    core.wob_scores += +reward;
  }

  if(core.wob_latest !== '-'){
    var latestText = (core.wob_latest == 1 ? 'success' : 'failure');
    var latestColor = (core.wob_latest == 1 ? 'green' : 'red');
    document.getElementById('reward-last').setAttribute('style', 'color: ' + latestColor);
    document.getElementById('reward-last').innerHTML = latestText;
  }

  var total = core.wob_scores;
  var totalColor = (total >= core.EPISODES_NEEDED ? 'green' : 'red');
  document.getElementById('reward-total').setAttribute('style', 'color: ' + totalColor);
  document.getElementById('reward-total').innerHTML = total;
}

core.countdownTimer = function(et){
  core.clearTimer();
  var episodeTime = et/1000;
  var currentTime = et/1000;
  var intervalTime = 1000;
  // update the timer immediately to display the total episode
  // time on start, eg. "10 / 10s"
  updateTimer();
  // set an interval so that the timer text will be updated
  // based on the intervalTime (ie. every 1sec)
  core.CD_TIMER = setInterval(updateTimer, intervalTime);

  function updateTimer(){
    var cd = document.getElementById('timer-countdown');
    if (currentTime <= 0){
      cd.setAttribute('style', 'color: red');
      cd.innerHTML = '0 / ' + episodeTime + 's';
      window.clearInterval(core.CD_TIMER);
      return;
    } else {
      var frac = currentTime / episodeTime;
      if(frac > 0.75) { var col = 'green'; }
      else if(frac > 0.5) { var col = 'orange'; }
      else if(frac > 0.25) { var col = 'brown'; }
      else { var col = 'red'; }
      cd.setAttribute('style', 'color:' + col);
      cd.innerHTML = currentTime + ' / ' + episodeTime + 'sec';
      currentTime-=intervalTime/1000;
    }
  }
};

core.clearTimer = function(){
  window.clearInterval(core.CD_TIMER);
  var cd = document.getElementById('timer-countdown');
  cd.setAttribute('style', 'color: black');
  cd.innerHTML = '-';
}

// ################################
// Custom getter

core.getUtterance = function () {
  var query = document.getElementById('query');
  return query.textContent.replace(/\s+/g, ' ').trim();
}

var previousDOMInfo = {};
var nextRefCode = 1, nextTextRefCode = -1;
function resetRefCode() {
  nextRefCode = 1;
  nextTextRefCode = -1;
}

/* Returns a nested object (dict) with all visible DOM element information.

   Special handling for Text nodes:
   - Text nodes with only whitespaces are discarded.
   - If the Text node is the only child, discard that Text node
     and reassign its text to the parent Element.
   - If the Text node is not the only child, it is broken into
     pseudo-Elements with tag "t".
*/
function getDOMInfo() {
  previousDOMInfo = {}

  function getDOMInfoOfElement(element) {
    if (element.id === 'reward-display'
        || element.id === 'sync-task-cover'
        || element.id === 'instructions'
        || element.id === 'query') return;
    var rect = element.getBoundingClientRect();
    if (rect.width == 0 || rect.height == 0) return;
    var answer = {
      tag: element.tagName,
      left: rect.left, top: rect.top,
      width: rect.width, height: rect.height,
      children: [],
      id: element.id,
      classes: element.className,
    };
    // Assign ref code
    if (element.dataset.wob_ref !== undefined
        && element.dataset.wob_eps === 'e' + WOB_EPISODE_ID) {
      answer.ref = +element.dataset.wob_ref;
    } else {
      element.dataset.wob_ref = answer.ref = nextRefCode++;
      element.dataset.wob_eps = 'e' + WOB_EPISODE_ID;
    }
    // Record styles
    var computedStyle = window.getComputedStyle(element);
    answer.bgColor = computedStyle.backgroundColor;
    answer.fgColor = computedStyle.color;
    // Indicate if the element is being focused on
    if (document.activeElement === element) {
      answer.focused = true;
    }
    // Indicate if the element is tampered with in this episode
    if (element.dataset.tampered !== undefined
        && element.dataset.tampered == 'e' + WOB_EPISODE_ID) {
      answer.tampered = true;
    }
    // For recording demonstrations: Record the target
    if (element.dataset.recording_target) {
      answer.recordingTarget = true;
    }
    // For <input>, also add input type and value
    if (element instanceof HTMLInputElement) {
      var inputType = element.type;
      answer.tag += '_' + inputType;
      if (inputType === 'checkbox' || inputType === 'radio') {
        answer.value = element.checked;
      } else {
        answer.value = element.value;
      }
    } else if (element instanceof HTMLTextAreaElement) {
      answer.value = element.value;
    }
    previousDOMInfo[answer.ref] = element;
    // Read the children
    var filteredChildNodes = [], textOnly = true;
    element.childNodes.forEach(function (child) {
      if (child instanceof Text) {
        if (!/^\s*$/.test(child.data)) {
          filteredChildNodes.push(child);
        }
      } else if (child instanceof Element) {
        filteredChildNodes.push(child);
        textOnly = false;
      }
    });
    if (textOnly) {
      answer.text = filteredChildNodes.map(function (x) {
        return x.data.trim();
      }).join(' ');
    } else {
      filteredChildNodes.forEach(function (child) {
        if (child instanceof Text) {
          addDOMInfosOfTextNode(child, answer.children);
        } else {
          child = getDOMInfoOfElement(child);
          if (child !== undefined)
            answer.children.push(child);
        }
      });
    }
    return answer;
  }

  function addDOMInfosOfTextNode(textNode, collection) {
    // Break the text node into multiple nodes
    // Each node only occupies a single rectangle boundary
    var range = document.createRange();
    range.selectNodeContents(textNode);
    var absolute_start = range.startOffset, absolute_end = range.endOffset;
    var start = absolute_start;
    var itr = 0;
    while (start < absolute_end) {
      // Binary search on the next end point
      var end_lower_bound = start + 1,
          end_upper_bound = absolute_end,
          l = range.getClientRects().length,
          end = Math.floor((end_lower_bound * (l-1) + end_upper_bound) / l);
      while (end_lower_bound <= end_upper_bound) {
        range.setEnd(textNode, end);
        if (range.getClientRects().length == 1) {
          end_lower_bound = end + 1;
          end = Math.min(end_lower_bound + 5, Math.floor((end_lower_bound + end_upper_bound) / 2));
        } else {
          end_upper_bound = end - 1;
          end = Math.max(end_upper_bound - 5, Math.floor((end_lower_bound + end_upper_bound) / 2));
        }
        if (itr++ > 1000) throwTextNodeError('Text node computation stuck in an infinite loop');
      }
      range.setEnd(textNode, end);
      var rects = range.getClientRects();
      if (rects.length !== 1) throwTextNodeError('Text node computation incorrect');
      var rect = rects[0], text = textNode.data.substring(start, end).trim();
      if (rect.width > 0 && rect.height > 0 && text) {
        var answer = {
          tag: "t",
          left: rect.left, top: rect.top,
          width: rect.width, height: rect.height,
          ref: nextTextRefCode--,
          children: [],
          text: text,
        };
        collection.push(answer);
      }
      start = end;
      range.setEnd(textNode, absolute_end);
      range.setStart(textNode, start);
      if (itr++ > 1000) throwTextNodeError('Text node computation stuck in an infinite loop');
    }
  }

  function throwTextNodeError(message) {
    alert(message);
    throw message;
  }

  return getDOMInfoOfElement(document.body);
}

 
/* Debug: return a mapping from ref to its DOMInfo */
function flattenDOMInfo(rootDomInfo, flattened) {
  if (flattened == undefined) flattened = {};
  flattened[rootDomInfo.ref] = rootDomInfo;
  rootDomInfo.children.forEach(function (x) { flattenDOMInfo(x, flattened); });
  return flattened;
}

// ################################################
// Record demonstrations

/* POST submit format

* utterance
* states: array of objects with the following keys:
  - time: time elapsed
  - dom: DOM structure
  - action: action performed at that moment
* reward

*/

var recorder = {};

// Add event listeners
recorder.LISTENERS = [
  'click',
  'dblclick',
  'mousedown',
  'mouseup',
  'keypress',
  'keydown',
  'keyup',
  'scroll',
];
recorder.setup = function () {
  if (recorder.isSetup) return;
  recorder.LISTENERS.forEach(function (name) {
    document.addEventListener(name, recorder['on' + name], true);
    document.addEventListener(name, recorder['on' + name], false);
  });
  recorder.isSetup = true;
}

// Start recording the episode
recorder.startRecording = function () {
  recorder.data = {};
  recorder.data.taskName = core.getTaskName();
  var utterance = core.getUtterance();
  if (typeof utterance === 'string') {
    recorder.data.utterance = utterance;
  } else {
    recorder.data.utterance = utterance.utterance;
    recorder.data.fields = utterance.fields;
  }
  recorder.data.states = [];
  recorder.isRecording = true;
  recorder.addState(null, null);
}

// Add a state to the recording data
recorder.addState = function (event, action) {
  if (!recorder.isRecording) return;
  if (event && action)
    action.timing = event.eventPhase;
  //console.log('Adding state', action);
  var state = {
    'time': new Date().getTime() - core.ept0,
    'action': action,
  };
  if (event)
    event.target.dataset.recording_target = true;
  state.dom = getDOMInfo();
  if (event)
    delete event.target.dataset.recording_target;
  recorder.data.states.push(state);
}

// Actions
recorder.ondblclick = function (event) {
  if (event.target === core.cover_div
      || event.pageX >= 160 || event.pageY >= 210)
    return;
  recorder.addState(event, {
    'type': 'dblclick',
    'x': event.pageX,
    'y': event.pageY,
  });
}
recorder.onclick = function (event) {
  if (event.target === core.cover_div
      || event.pageX >= 160 || event.pageY >= 210)
    return;
  recorder.addState(event, {
    'type': 'click',
    'x': event.pageX,
    'y': event.pageY,
  });
}
recorder.onmousedown = function (event) {
  if (event.target === core.cover_div
      || event.pageX >= 160 || event.pageY >= 210)
    return;
  recorder.addState(event, {
    'type': 'mousedown',
    'x': event.pageX,
    'y': event.pageY,
  });
}
recorder.onmouseup = function (event) {
  if (event.target === core.cover_div
      || event.pageX >= 160 || event.pageY >= 210)
    return;
  recorder.addState(event, {
    'type': 'mouseup',
    'x': event.pageX,
    'y': event.pageY,
  });
}

recorder.onkeypress = function (event) {
  recorder.addState(event, {
    'type': 'keypress',
    'keyCode': event.keyCode,
    'charCode': event.charCode,
  });
}
recorder.onkeydown = function (event) {
  recorder.addState(event, {
    'type': 'keydown',
    'keyCode': event.keyCode,
    'charCode': event.charCode,
  });
}
recorder.onkeyup = function (event) {
  recorder.addState(event, {
    'type': 'keyup',
    'keyCode': event.keyCode,
    'charCode': event.charCode,
  });
}

recorder.onscroll = function (event) {
  // Scroll is super redundant; only keep the first one
  if (recorder.data.states.length) {
    var lastState = recorder.data.states[recorder.data.states.length - 1];
    if (lastState.action && lastState.action.type === 'scroll')
      return;
      //recorder.data.states.pop();     // <-- use this for keeping the last one
  }
  recorder.addState(event, {
    'type': 'scroll',
  });
}

recorder.recordedEpisodes = 0;

// End recording the episode
recorder.endRecording = function () {
  recorder.data.reward = WOB_REWARD_GLOBAL;
  recorder.data.rawReward = WOB_RAW_REWARD_GLOBAL;
  // Add data to an input
  recorder.isRecording = false;
  var data = recorder.data;
  recorder.data = {};   // Prevent future addition
  //console.log(data);
  data = recorder.compress(data);
  var dumped = document.createElement("input");
  dumped.setAttribute("type", "hidden");
  dumped.setAttribute("name", "d" + WOB_EPISODE_ID);
  dumped.setAttribute("value", data);
  document.getElementById('mturk_form').appendChild(dumped);
  if (WOB_RAW_REWARD_GLOBAL == 1) recorder.recordedEpisodes++;
  // Make it ready for the next episode
  core.cover_div.classList.remove('cover-transparent');
  if (recorder.recordedEpisodes >= core.EPISODES_NEEDED) {
    core.cover_div.classList.add('cover-submit');
    core.cover_div.innerHTML = 'SUBMIT';
    core.cover_div.onclick = function () {
      document.getElementById('mturk_form').submit();
    }
  }
}

recorder.compress = function (data) {
  data = JSON.stringify(data);
  data = pako.deflate(data, {to:'string'})
  data = btoa(data);
  return data;
}

// ################################
// Wrappers

// Wrap startEpisodeReal
core.startEpisodeReal = (function(startEpisodeReal) {
  return function () {
    if (core.cover_div.classList.contains('cover-transparent')) return;
    recorder.setup();
    startEpisodeReal();
    recorder.startRecording();
  }
})(core.startEpisodeReal);

// Wrap endEpisode
core.endEpisode = (function(endEpisode) {
  return function (reward, time_proportional) {
    if (core.EP_TIMER === null) return;
    core.cover_div.classList.add('cover-transparent');
    endEpisode(reward, time_proportional);
    // Delay to allow the last action to be recorded
    setTimeout(recorder.endRecording, 500);
  }
})(core.endEpisode);

// ################################
// Initial setup for MTurk

core.checkPreviewMode = function () {
  if (core.QueryString.assignmentId != "ASSIGNMENT_ID_NOT_AVAILABLE") return;
  document.getElementById('preview-mode-wrapper').style.display = 'block';
};

(function () {
  // Pako for compression
  var script = document.createElement('script');
  script.type = 'text/javascript';
  script.src = 'https://cdnjs.cloudflare.com/ajax/libs/pako/1.0.6/pako.min.js';
  document.head.appendChild(script);
})();
