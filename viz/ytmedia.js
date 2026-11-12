// A YouTube player that behaves enough like a <video> element for the Trailer View's clock:
// currentTime (get / set), paused, duration, muted, play(), pause(), and the events
// play, pause, timeupdate, seeked, loadedmetadata, error. The public site plays trailers from
// YouTube; the page's sync logic stays the same as with a local file.
//
// YouTube quirks it hides:
// - before the first play, seekTo() or cueing at an offset leaves the player stuck "unstarted", so
//   until then the wanted time is only remembered; play starts from 0 and jumps there once it runs
// - seekTo() on an ended video starts playback, so that seek re-pauses ("hold")
// - getCurrentTime() lags right after a seek, so the target time is returned until it catches up
class YTMedia extends EventTarget {
  constructor(el) {
    super();
    this._paused = true; this._target = null; this._hold = false; this._t = 0; this._id = null;
    this._started = false; this._startAt = null;
    this.ready = YTMedia.api().then(() => new Promise(res => {
      this.p = new YT.Player(el, {
        width: "100%", height: "100%",
        playerVars: { controls: 0, disablekb: 1, rel: 0, modestbranding: 1, playsinline: 1, iv_load_policy: 3, fs: 0 },
        events: {
          onReady: () => res(),
          onStateChange: e => this._state(e.data),
          onError: () => this._emit("error"),
        },
      });
    }));
  }

  static api() {
    if (!YTMedia._api) YTMedia._api = new Promise(res => {
      window.onYouTubeIframeAPIReady = res;
      const s = document.createElement("script");
      s.src = "https://www.youtube.com/iframe_api";
      document.head.appendChild(s);
    });
    return YTMedia._api;
  }

  _emit(type) { this.dispatchEvent(new Event(type)); }

  async load(videoId, t0 = 0) {
    await this.ready;
    this._paused = true; this._hold = false; this._t = t0; this._id = videoId;
    this._started = false; this._target = this._startAt = t0 || null;
    this.p.cueVideoById({ videoId });
  }

  get currentTime() {
    if (!this.p || !this.p.getCurrentTime || !this._started) return this._target ?? this._t;
    const now = this.p.getCurrentTime();
    if (this._target !== null) {
      if (Math.abs(now - this._target) < 0.1) this._target = null;
      else return this._target;
    }
    return now;
  }
  set currentTime(t) {
    this._t = t; this._target = t;
    if (!this.p || !this.p.seekTo) return;
    if (!this._started) {                    // not playing yet: remember it, jump there on play
      this._startAt = t;
      setTimeout(() => this._emit("seeked"), 0);
      return;
    }
    if (this._paused && this.p.getPlayerState() !== 2) this._hold = true;   // seekTo from "ended" would play
    this.p.seekTo(t, true);
    this._waitSeeked(t);
  }
  _waitSeeked(t, n = 0) {
    const now = this.p.getCurrentTime ? this.p.getCurrentTime() : t;
    if (Math.abs(now - t) < 0.1 || n > 40) { this._emit("seeked"); return; }
    setTimeout(() => this._waitSeeked(t, n + 1), 50);
  }

  get paused() { return this._paused; }
  get duration() { const d = this.p && this.p.getDuration ? this.p.getDuration() : 0; return d || NaN; }
  get muted() { return this.p && this.p.isMuted ? this.p.isMuted() : false; }
  set muted(v) { if (this.p) (v ? this.p.mute() : this.p.unMute()); }
  get videoId() { return this._id; }

  play() {
    this._hold = false;
    if (this.p) this.p.playVideo();
    return Promise.resolve();
  }
  pause() { this._hold = false; if (this.p) this.p.pauseVideo(); }

  // -1 unstarted, 0 ended, 1 playing, 2 paused, 3 buffering, 5 cued
  _state(s) {
    if (s === 1) {
      this._started = true;
      if (this._startAt !== null) { const t = this._startAt; this._startAt = null; this.p.seekTo(t, true); }
      if (this._hold) { this.p.pauseVideo(); return; }
      if (this._paused) { this._paused = false; this._emit("play"); this._tick(); }
    } else if (s === 2 || s === 0) {
      this._hold = false;
      if (!this._paused) { this._paused = true; this._emit("pause"); }
      this._emit("timeupdate");
    } else if (s === 5) {
      this._emit("loadedmetadata");
    }
  }
  _tick() {
    if (this._paused) return;
    this._emit("timeupdate");
    setTimeout(() => this._tick(), 250);
  }
}
