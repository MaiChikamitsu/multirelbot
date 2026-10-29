'use strict';
const video = document.getElementById('video');
const play = document.getElementById('play');
const theme = document.getElementById('theme');
const status = document.getElementById('status');
const repeat = document.getElementById('repeat');
const restart = document.getElementById('restart');
const volumeBoost = document.getElementById('volume-boost');
const volumeValue = document.getElementById('volume-value');
let audioContext, gainNode;
function prepareAudio() {
  if (!audioContext) {
    const AudioEngine = window.AudioContext || window.webkitAudioContext;
    if (!AudioEngine) return;
    audioContext = new AudioEngine();
    gainNode = audioContext.createGain();
    const source = audioContext.createMediaElementSource(video);
    source.connect(gainNode);
    gainNode.connect(audioContext.destination);
    gainNode.gain.value = Number(volumeBoost.value);
  }
  // Called directly from a user gesture so browsers can enable audio.
  if (audioContext.state === 'suspended') audioContext.resume().catch(() => {
    status.textContent = '音声を有効にするには、もう一度再生を押してください。';
  });
}
volumeBoost.oninput = () => {
  volumeValue.textContent = `${Number(volumeBoost.value).toFixed(1)}倍`;
  prepareAudio();
  if (gainNode) gainNode.gain.setTargetAtTime(Number(volumeBoost.value), audioContext.currentTime, 0.02);
};
let videos = [], config, index = 0, stopped = false, frame;
const stamp = n => `${Math.floor(n / 60)}:${String(Math.floor(n % 60)).padStart(2, '0')}`;
function startTime() { return index === 0 ? config.start : config.stops[index - 1]; }
function render() {
  document.getElementById('segment').textContent = `インデックス ${index + 1} / ${config.stops.length} （${stamp(startTime())}〜${stamp(config.stops[index])}）`;
  play.disabled = stopped && index === config.stops.length - 1;
  play.textContent = stopped ? '次の区間を再生' : video.paused ? '再生' : '一時停止';
}
function checkStop() {
  if (!config || stopped) return;
  if (video.currentTime >= config.stops[index]) {
    video.pause();
    stopped = true;
    video.currentTime = config.stops[index];
    status.textContent = index === config.stops.length - 1 ? '最後のインデックスの再生が終了しました。' : `インデックス ${index + 1} の終わりで停止しました。`;
    render();
  }
}
function tick() {
  checkStop();
  if (!video.paused) frame = requestAnimationFrame(tick);
}
function resetSegment() {
  video.pause();
  stopped = false;
  video.currentTime = startTime();
  status.textContent = '再生すると、この区間の終わりで自動停止します。';
  render();
}
function load() {
  video.pause(); cancelAnimationFrame(frame);
  config = videos.find(v => v.id === theme.value); index = 0; stopped = false;
  play.disabled = repeat.disabled = restart.disabled = true;
  video.src = `/media/${encodeURIComponent(config.id)}`;
  video.load();
}
video.addEventListener('loadedmetadata', () => { resetSegment(); repeat.disabled = restart.disabled = false; });
video.addEventListener('timeupdate', checkStop);
video.addEventListener('play', () => { cancelAnimationFrame(frame); render(); tick(); });
video.addEventListener('pause', render);
video.addEventListener('error', () => { status.textContent = '動画を読み込めませんでした。'; play.disabled = true; });
play.onclick = async () => {
  if (!video.paused) { video.pause(); return; }
  if (stopped) {
    if (index === config.stops.length - 1) return;
    index++; stopped = false; video.currentTime = startTime();
  }
  status.textContent = '';
  prepareAudio();
  try { await video.play(); } catch { status.textContent = '再生できませんでした。もう一度再生を押してください。'; }
  render();
};
repeat.onclick = resetSegment;
restart.onclick = () => { index = 0; resetSegment(); };
theme.onchange = load;
document.addEventListener('visibilitychange', () => { if (document.hidden) video.pause(); });
fetch('/api/videos').then(r => { if (!r.ok) throw new Error(); return r.json(); }).then(items => {
  videos = items;
  if (!items.length) throw new Error();
  for (const item of items) { const option = document.createElement('option'); option.value = item.id; option.textContent = item.theme; theme.append(option); }
  load();
}).catch(() => { status.textContent = '登録済みの動画設定を読み込めませんでした。'; });
