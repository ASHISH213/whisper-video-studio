(function () {
  "use strict";
  var $ = function (s, r) { return (r || document).querySelector(s); };
  var $$ = function (s, r) { return Array.prototype.slice.call((r || document).querySelectorAll(s)); };
  var reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  /* Theme toggle */
  $("#theme").addEventListener("click", function () {
    var root = document.documentElement;
    var current = root.getAttribute("data-theme");
    if (!current) current = window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
    var next = current === "dark" ? "light" : "dark";
    root.setAttribute("data-theme", next);
    try { localStorage.setItem("wvs-theme", next); } catch (e) {}
  });

  /* Timestamp rail: highlight the section in view */
  var railLinks = $$("[data-rail]");
  var ids = railLinks.map(function (a) { return a.getAttribute("data-rail"); });
  function setRail(id) {
    railLinks.forEach(function (a) {
      if (a.getAttribute("data-rail") === id) a.setAttribute("aria-current", "true");
      else a.removeAttribute("aria-current");
    });
  }
  if ("IntersectionObserver" in window) {
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (e) { if (e.isIntersecting) setRail(e.target.id); });
    }, { rootMargin: "-35% 0px -55% 0px" });
    ids.forEach(function (id) { var el = document.getElementById(id); if (el) io.observe(el); });
  }
  setRail("top");

  /* Hero demo: waveform + transcript lines */
  var wave = $("#wave");
  var bars = [];
  var COUNT = Math.max(32, Math.min(96, Math.floor(wave.clientWidth / 7)));
  for (var i = 0; i < COUNT; i++) {
    var b = document.createElement("i");
    var h = 18 + Math.abs(Math.sin(i * 0.37) * 38 + Math.sin(i * 1.3) * 18) + (i * 7 % 11);
    b.style.setProperty("--h", Math.min(h, 100) + "%");
    wave.appendChild(b);
    bars.push(b);
  }
  var lines = $$("#lines li");
  var clock = $("#demoClock");
  var barEl = $("#demoBar");
  var statusEl = $("#demoStatus");
  var TOTAL = 34; // seconds of demo audio
  var timer = null;

  function pad(n) { return (n < 10 ? "0" : "") + n; }
  function render(t) {
    var pct = Math.min(t / TOTAL, 1);
    barEl.style.width = (pct * 100) + "%";
    var on = Math.round(pct * bars.length);
    bars.forEach(function (b, idx) { b.classList.toggle("on", idx < on); });
    clock.textContent = "00:" + pad(Math.floor(t / 60)) + ":" + pad(Math.floor(t % 60));
    var latest = -1;
    lines.forEach(function (li, idx) {
      var show = t >= Number(li.dataset.t);
      li.classList.toggle("show", show);
      if (show) latest = idx;
    });
    lines.forEach(function (li, idx) { li.classList.toggle("now", idx === latest && t < TOTAL); });
  }
  function run() {
    clearInterval(timer);
    var t = 0;
    statusEl.textContent = "Transcribing";
    render(0);
    timer = setInterval(function () {
      t += 0.25;
      render(t);
      if (t >= TOTAL) {
        clearInterval(timer);
        statusEl.textContent = "Done: .md .srt .txt saved";
      }
    }, 250 / 3);
  }
  if (reduce) {
    render(TOTAL);
    statusEl.textContent = "Done: .md .srt .txt saved";
  } else {
    var started = false;
    var demo = $("#demo");
    if ("IntersectionObserver" in window) {
      new IntersectionObserver(function (es, ob) {
        if (es[0].isIntersecting && !started) { started = true; run(); ob.disconnect(); }
      }, { threshold: 0.4 }).observe(demo);
    } else { run(); }
  }
  $("#replay").addEventListener("click", function () {
    if (reduce) { render(0); setTimeout(function () { render(TOTAL); }, 400); return; }
    run();
  });

  /* Model picker */
  var picks = {
    balanced: { model: "distil-large-v3", note: "distil-large-v3 gives high accuracy at high speed on a 3 GB budget. It's a good default." },
    accuracy: { model: "large-v3", note: "large-v3 is the most accurate and handles technical terms best. It needs about 6 GB of VRAM and runs slower." },
    speed: { model: "base", note: "base is the quickest and fine for rough drafts. Expect more mistakes on jargon." },
    memory: { model: "small", note: "small fits in about 1.5 GB of VRAM and still runs fast." }
  };
  var rows = $$("#modelTable tbody tr");
  function pick(key) {
    var p = picks[key];
    rows.forEach(function (r) { r.classList.toggle("rec", r.dataset.model === p.model); });
    $("#pickNote").textContent = p.note;
  }
  var modelBtns = $$("[data-pick]");
  modelBtns.forEach(function (btn) {
    btn.addEventListener("click", function () {
      modelBtns.forEach(function (b) { b.setAttribute("aria-checked", b === btn ? "true" : "false"); });
      pick(btn.dataset.pick);
    });
  });
  pick("balanced");

  /* Generic tab behaviour with arrow keys */
  function tabs(listEl, onSelect) {
    var btns = $$("[role=tab]", listEl);
    function select(i, focus) {
      btns.forEach(function (b, idx) {
        b.setAttribute("aria-selected", idx === i ? "true" : "false");
        b.tabIndex = idx === i ? 0 : -1;
      });
      if (focus) btns[i].focus();
      onSelect(i);
    }
    btns.forEach(function (b, i) {
      b.addEventListener("click", function () { select(i); });
      b.addEventListener("keydown", function (e) {
        var n = btns.length, j = null;
        if (e.key === "ArrowRight") j = (i + 1) % n;
        else if (e.key === "ArrowLeft") j = (i - 1 + n) % n;
        else if (e.key === "Home") j = 0;
        else if (e.key === "End") j = n - 1;
        if (j !== null) { e.preventDefault(); select(j, true); }
      });
    });
  }

  /* Screenshots */
  var shots = [
    { src: "screenshots/05_step5_live_dashboard.png", alt: "Live transcription dashboard showing GPU status, progress and speed", cap: "Watch progress, speed and time remaining while a batch runs." },
    { src: "screenshots/02_step2_media_selection.png", alt: "Media scanner listing the video and audio files found in a folder", cap: "See every file found in your folder and choose what to transcribe." },
    { src: "screenshots/03_step3_model_config.png", alt: "Model configuration screen with model and language options", cap: "Pick a Whisper model and settings that fit your GPU." },
    { src: "screenshots/04_step4_dry_run_review.png", alt: "Dry-run review listing planned work before transcription starts", cap: "Review the plan before anything runs. Nothing is processed in a dry run." },
    { src: "screenshots/06_transcripts_viewer.png", alt: "Transcript viewer showing a finished transcript with timestamps", cap: "Read, search and copy finished transcripts and notes." }
  ];
  var img = $("#shotImg"), cap = $("#shotCap"), panel = $("#shotPanel");
  tabs($("#shotTabs"), function (i) {
    img.src = shots[i].src; img.alt = shots[i].alt; cap.textContent = shots[i].cap;
    panel.setAttribute("aria-labelledby", "tab-" + i);
  });

  /* Lightbox */
  var lb = $("#lightbox"), lbImg = $("#lbImg");
  $("#shotOpen").addEventListener("click", function () {
    lbImg.src = img.src; lbImg.alt = img.alt;
    if (lb.showModal) lb.showModal(); else window.open(img.src, "_blank");
  });
  lb.addEventListener("click", function (e) { if (e.target === lb || e.target === lbImg) lb.close(); });

  /* Run tabs */
  var panels = [$("#runColab"), $("#runLocal")];
  tabs($("#runTabs"), function (i) {
    panels.forEach(function (p, idx) { p.hidden = idx !== i; });
  });

  /* OS toggle + copy */
  var cmds = {
    win: "git clone https://github.com/ASHISH213/whisper-video-studio.git\ncd whisper-video-studio\npython -m venv .venv\n.venv\\Scripts\\activate\npip install -r requirements.txt\npython colab_studio.py",
    nix: "git clone https://github.com/ASHISH213/whisper-video-studio.git\ncd whisper-video-studio\npython3 -m venv .venv\nsource .venv/bin/activate\npip install -r requirements.txt\npython colab_studio.py"
  };
  var osBtns = $$("[data-os]");
  osBtns.forEach(function (btn) {
    btn.addEventListener("click", function () {
      osBtns.forEach(function (b) { b.setAttribute("aria-checked", b === btn ? "true" : "false"); });
      $("#localCode").textContent = cmds[btn.dataset.os];
    });
  });
  var copyBtn = $("#copyBtn");
  copyBtn.addEventListener("click", function () {
    var text = $("#localCode").textContent;
    function done(ok) {
      copyBtn.textContent = ok ? "Copied" : "Press Ctrl+C to copy";
      setTimeout(function () { copyBtn.textContent = "Copy"; }, 1800);
    }
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(text).then(function () { done(true); }, function () { done(false); });
    } else { done(false); }
  });
})();
