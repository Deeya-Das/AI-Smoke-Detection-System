const imageInput = document.getElementById('imageInput');
const fileName = document.getElementById('fileName');
const previewImage = document.getElementById('previewImage');
const previewEmpty = document.getElementById('previewEmpty');

const resultBanner = document.getElementById('resultBanner');
const bannerIcon = document.getElementById('bannerIcon');
const bannerText = document.getElementById('bannerText');
const confidenceValue = document.getElementById('confidenceValue');
const detectionTime = document.getElementById('detectionTime');
const alarmStatus = document.getElementById('alarmStatus');
const stopAlarmBtn = document.getElementById('stopAlarmBtn');

const detectBtn = document.getElementById('detectBtn');
const clearBtn = document.getElementById('clearBtn');
const exitBtn = document.getElementById('exitBtn');

const alarmAudio = document.getElementById('alarmAudio');
const headerClock = document.getElementById('headerClock');

let selectedFile = null;

const BANNER_ICONS = {
  waiting: '<circle cx="12" cy="12" r="9"></circle><path d="M12 7v6l4 2"></path>',
  safe: '<path d="M8 12.5 11 15.5 16 9"></path><circle cx="12" cy="12" r="9"></circle>',
  danger: '<path d="M12 4 21 20H3L12 4Z"></path><path d="M12 10v4"></path><path d="M12 17h.01"></path>'
};

function formatClock(date) {
  return date.toLocaleTimeString('en-IN', { hour12: false });
}

function startClock() {
  const tick = () => { headerClock.textContent = formatClock(new Date()); };
  tick();
  setInterval(tick, 1000);
}

function handleImageSelect(event) {
  const file = event.target.files[0];
  if (!file) return;

  selectedFile = file;
  fileName.textContent = file.name;

  const reader = new FileReader();
  reader.onload = () => {
    previewImage.src = reader.result;
    previewImage.classList.remove('hidden');
    previewEmpty.classList.add('hidden');
  };
  reader.readAsDataURL(file);

  detectBtn.disabled = false;
}

function setBanner(state, text) {
  resultBanner.classList.remove('waiting', 'safe', 'danger');
  resultBanner.classList.add(state);
  bannerIcon.innerHTML = BANNER_ICONS[state];
  bannerText.textContent = text;
}

function setAlarm(isOn) {
  if (isOn) {
    alarmStatus.textContent = 'ON';
    alarmStatus.classList.remove('alarm-off');
    alarmStatus.classList.add('alarm-on');
    stopAlarmBtn.classList.remove('hidden');
    alarmAudio.currentTime = 0;
    alarmAudio.play().catch(() => {
      console.warn('Playback blocked until the page receives a user interaction.');
    });
  } else {
    alarmStatus.textContent = 'OFF';
    alarmStatus.classList.remove('alarm-on');
    alarmStatus.classList.add('alarm-off');
    stopAlarmBtn.classList.add('hidden');
    alarmAudio.pause();
    alarmAudio.currentTime = 0;
  }
}

function renderResult(data) {
  const status = data.status;
  const confidence = Number(data.confidence).toFixed(2);

  confidenceValue.textContent = `${confidence}%`;
  detectionTime.textContent = formatClock(new Date());

  if (status === 'Fire' || status === 'Smoke') {
    setBanner('danger', 'FIRE / SMOKE DETECTED');
    setAlarm(true);
  } else {
    setBanner('safe', 'SYSTEM SAFE');
    setAlarm(false);
  }
}

async function detectFire() {
  if (!selectedFile) return;

  detectBtn.disabled = true;
  detectBtn.textContent = 'Analyzing...';

  const formData = new FormData();
  formData.append('file', selectedFile);

  try {
    const response = await fetch('/upload', {
      method: 'POST',
      body: formData
    });

    if (!response.ok) {
      throw new Error(`Server responded with status ${response.status}`);
    }

    const data = await response.json();
    renderResult(data);
  } catch (error) {
    console.error('Detection request failed:', error);
    setBanner('waiting', 'Detection Failed - Check Server');
  } finally {
    detectBtn.disabled = false;
    detectBtn.textContent = 'Detect Fire';
  }
}

function clearDashboard() {
  selectedFile = null;
  imageInput.value = '';
  fileName.textContent = 'No file chosen';

  previewImage.classList.add('hidden');
  previewImage.src = '';
  previewEmpty.classList.remove('hidden');

  confidenceValue.textContent = '--';
  detectionTime.textContent = '--';

  setBanner('waiting', 'Waiting for Detection');
  setAlarm(false);
  detectBtn.disabled = true;
}

function exitApp() {
  setAlarm(false);
  const closed = window.close();
  if (window.top === window.self) {
    alert('You can now close this browser tab.');
  }
}

imageInput.addEventListener('change', handleImageSelect);
detectBtn.addEventListener('click', detectFire);
clearBtn.addEventListener('click', clearDashboard);
exitBtn.addEventListener('click', exitApp);
stopAlarmBtn.addEventListener('click', () => setAlarm(false));

startClock();
