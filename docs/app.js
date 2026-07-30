(function () {
  "use strict";

  var API_BASE = "https://swing-option-viewer-528005343431.europe-west4.run.app";
  var CONFIG_KEYS = ["T", "alpha_preset", "kappa", "sigma", "r", "K", "n_max", "n_min", "q_max"];
  var NUMERIC_KEYS = ["T", "kappa", "sigma", "r", "K", "n_max", "n_min", "q_max"];
  var SLIDER_KEYS = ["n_max", "n_min", "kappa", "sigma", "r", "K"];

  var STAT_LABELS = {
    price: "Option price",
    pct_forced_states: "Forced-exercise states",
    forced_onset_mean_t: "Forced-exercise onset (mean t, yrs)",
    boundary_alpha_corr: "Boundary–α correlation",
    pct_bang_bang: "Bang-bang states",
    stationary_sd: "Stationary s.d.",
    moneyness: "Moneyness"
  };

  var presets = null;
  var config = null;
  var currentResult = null;
  var statusTimer = null;

  function $(id) {
    return document.getElementById(id);
  }

  function clamp(v, lo, hi) {
    return Math.min(hi, Math.max(lo, v));
  }

  function fmtNum(v, decimals) {
    if (v === null || v === undefined) return "–";
    var num = Number(v);
    if (Number.isNaN(num)) return "–";
    return num.toLocaleString("en-US", { maximumFractionDigits: decimals === undefined ? 4 : decimals, minimumFractionDigits: 0 });
  }

  function humanizeKey(key) {
    return key.replace(/_/g, " ").replace(/\b\w/g, function (c) { return c.toUpperCase(); });
  }

  // ---- rendering ----

  function renderSurface(result) {
    var trace = {
      type: "surface",
      x: result.volume_grid,
      y: result.time_grid,
      z: result.boundary,
      colorscale: "Viridis",
      colorbar: { title: { text: "S* ($)" } },
      hovertemplate: "C=%{x:.2f}<br>t=%{y:.3f}<br>S*=%{z:.3f}<extra></extra>"
    };
    var layout = {
      autosize: true,
      margin: { l: 0, r: 0, t: 10, b: 0 },
      scene: {
        xaxis: { title: { text: "Cumulative volume" } },
        yaxis: { title: { text: "Time (years)" } },
        zaxis: { title: { text: "Exercise threshold ($)" } },
        // The forced-exercise gap sits at low cumulative volume and late time
        // (too few exercises used with too little horizon left). Look in from
        // the high-volume / early-time side so that far corner stays visible
        // instead of hiding behind the front of the surface.
        camera: { eye: { x: 1.6, y: -1.6, z: 0.8 } }
      },
      font: { family: "IBM Plex Mono, ui-monospace, monospace", size: 11 }
    };
    Plotly.react("surface-plot", [trace], layout, { displaylogo: false, responsive: true });
  }

  function renderSlice(result, idx) {
    var nCols = result.volume_grid.length;
    idx = clamp(idx | 0, 0, nCols - 1);

    var sValues = result.boundary.map(function (row) { return row[idx]; });
    var traceS = {
      x: result.time_grid, y: sValues, mode: "lines", name: "S*(t)",
      line: { color: "#2b5246", width: 2 }, connectgaps: false
    };
    var traceAlpha = {
      x: result.time_grid, y: result.alpha, mode: "lines", name: "α(t)",
      line: { color: "#b8860b", width: 1.5, dash: "dot" }
    };
    var layout = {
      margin: { l: 55, r: 15, t: 10, b: 40 },
      xaxis: { title: { text: "Time (years)" } },
      yaxis: { title: { text: "Price ($)" } },
      legend: { orientation: "h", y: 1.12 },
      font: { family: "IBM Plex Mono, ui-monospace, monospace", size: 11 }
    };
    Plotly.react("slice-plot", [traceS, traceAlpha], layout, { displaylogo: false, responsive: true });
    $("slice-value").textContent = "C = " + fmtNum(result.volume_grid[idx], 2);
  }

  function renderStats(stats) {
    var tbody = document.querySelector("#stats-table tbody");
    tbody.innerHTML = "";
    Object.keys(stats || {}).forEach(function (key) {
      if (key === "STUB") return;
      var val = stats[key];
      var tr = document.createElement("tr");
      var tdLabel = document.createElement("td");
      tdLabel.textContent = STAT_LABELS[key] || humanizeKey(key);
      var tdVal = document.createElement("td");
      tdVal.textContent = formatStatValue(key, val);
      tr.appendChild(tdLabel);
      tr.appendChild(tdVal);
      tbody.appendChild(tr);
    });
    $("stub-banner").hidden = !(stats && stats.STUB);
  }

  function formatStatValue(key, val) {
    if (val === null || val === undefined) return "–";
    if (typeof val === "boolean") return val ? "Yes" : "No";
    if (typeof val === "number") {
      if (key.indexOf("pct_") === 0) return fmtNum(val, 2) + "%";
      return fmtNum(val, key === "price" ? 2 : 4);
    }
    return String(val);
  }

  function renderMeta(meta) {
    var block = $("resolution-note-block");
    var text = $("resolution-note-text");
    if (meta && meta.resolution_note) {
      var extra = typeof meta.solve_seconds === "number" ? " Solved in " + meta.solve_seconds.toFixed(2) + "s." : "";
      text.textContent = meta.resolution_note + extra;
      block.hidden = false;
    } else {
      block.hidden = true;
    }
  }

  function renderResult(result) {
    currentResult = result;
    renderSurface(result);

    var nCols = result.volume_grid.length;
    var idx = Math.floor((nCols - 1) / 2);
    var sliceInput = $("input-slice");
    sliceInput.min = 0;
    sliceInput.max = Math.max(0, nCols - 1);
    sliceInput.step = 1;
    sliceInput.value = idx;
    renderSlice(result, idx);

    renderStats(result.stats || {});
    renderMeta(result.meta);
  }

  // ---- bounds / derived values ----

  function boundsFor(key, cfg) {
    var N = Math.round(cfg.T * presets.dates_per_year);
    var preset = presets.alpha_presets[cfg.alpha_preset];
    switch (key) {
      case "n_max":
        return { min: 1, max: Math.min(presets.n_max_cap, N), step: 1 };
      case "n_min":
        return { min: 0, max: cfg.n_max, step: 1 };
      case "kappa":
        return presets.ranges.kappa;
      case "sigma":
        return { min: presets.ranges.sigma.min, max: preset.sigma_max, step: presets.ranges.sigma.step };
      case "r":
        return presets.ranges.r;
      case "K": {
        var kMin = presets.ranges.K.min > 0 ? presets.ranges.K.min : presets.ranges.K.step;
        return { min: kMin, max: preset.K_max, step: presets.ranges.K.step };
      }
    }
  }

  function clampConfig(raw) {
    var cfg = {};
    CONFIG_KEYS.forEach(function (k) { cfg[k] = raw[k]; });

    if (presets.T_choices.indexOf(cfg.T) === -1) cfg.T = presets.calibrated.T;
    if (!presets.alpha_presets[cfg.alpha_preset]) cfg.alpha_preset = presets.calibrated.alpha_preset;

    var N = Math.round(cfg.T * presets.dates_per_year);
    cfg.n_max = Math.round(clamp(cfg.n_max, 1, Math.min(presets.n_max_cap, N)));
    cfg.n_min = Math.round(clamp(cfg.n_min, 0, cfg.n_max));

    cfg.kappa = clamp(cfg.kappa, presets.ranges.kappa.min, presets.ranges.kappa.max);
    var preset = presets.alpha_presets[cfg.alpha_preset];
    cfg.sigma = clamp(cfg.sigma, presets.ranges.sigma.min, preset.sigma_max);
    cfg.r = clamp(cfg.r, presets.ranges.r.min, presets.ranges.r.max);
    var kMin = presets.ranges.K.min > 0 ? presets.ranges.K.min : presets.ranges.K.step;
    cfg.K = clamp(cfg.K, kMin, preset.K_max);
    if (!(cfg.q_max > 0)) cfg.q_max = presets.calibrated.q_max;

    return cfg;
  }

  function updateAllBoundsAndValues() {
    $("input-T").value = String(config.T);
    $("input-alpha_preset").value = config.alpha_preset;
    $("input-N").value = String(Math.round(config.T * presets.dates_per_year));
    $("input-q_max").value = config.q_max;

    SLIDER_KEYS.forEach(function (key) {
      var b = boundsFor(key, config);
      var range = $("input-" + key);
      var num = $("input-" + key + "-num");
      range.min = b.min; range.max = b.max; range.step = b.step;
      num.min = b.min; num.max = b.max; num.step = b.step;
      range.value = config[key];
      num.value = config[key];
    });

    var preset = presets.alpha_presets[config.alpha_preset];
    $("illustrative-note").hidden = !preset.illustrative;
  }

  function recomputeDerived() {
    if (!presets) return;
    var preset = presets.alpha_presets[config.alpha_preset];
    if (!preset) return;
    var level = preset.level;
    var moneyness = config.K / level;
    var sd = config.sigma / Math.sqrt(2 * config.kappa);
    $("derived-moneyness").textContent = fmtNum(moneyness, 4);
    $("derived-sd").textContent = fmtNum(sd, 4);
    $("derived-sd-pct").textContent = fmtNum((sd / level) * 100, 2) + "%";
  }

  // ---- initial (pre-/presets) seeding from the bundled result ----

  function seedDisabledControls(cfg) {
    var tSel = $("input-T");
    tSel.innerHTML = "";
    var optT = document.createElement("option");
    optT.value = String(cfg.T);
    optT.textContent = cfg.T + " yr";
    tSel.appendChild(optT);
    tSel.value = String(cfg.T);

    var aSel = $("input-alpha_preset");
    aSel.innerHTML = "";
    var optA = document.createElement("option");
    optA.value = cfg.alpha_preset;
    optA.textContent = humanizeKey(cfg.alpha_preset);
    aSel.appendChild(optA);
    aSel.value = cfg.alpha_preset;

    $("input-N").value = String(Math.round(cfg.T * 252));
    $("input-q_max").value = cfg.q_max;

    SLIDER_KEYS.forEach(function (key) {
      var range = $("input-" + key);
      var num = $("input-" + key + "-num");
      range.value = cfg[key];
      num.value = cfg[key];
    });

    var stats = currentResult && currentResult.stats;
    if (stats) {
      $("derived-moneyness").textContent = fmtNum(stats.moneyness, 4);
      $("derived-sd").textContent = fmtNum(stats.stationary_sd, 4);
      if (stats.moneyness) {
        var level = cfg.K / stats.moneyness;
        $("derived-sd-pct").textContent = fmtNum((stats.stationary_sd / level) * 100, 2) + "%";
      }
    }

    setControlsEnabled(false);
  }

  function setControlsEnabled(enabled) {
    ["input-T", "input-alpha_preset", "input-q_max", "btn-calculate"].forEach(function (id) {
      $(id).disabled = !enabled;
    });
    SLIDER_KEYS.forEach(function (key) {
      $("input-" + key).disabled = !enabled;
      $("input-" + key + "-num").disabled = !enabled;
    });
  }

  function buildControlsFromPresets() {
    var tSel = $("input-T");
    tSel.innerHTML = "";
    presets.T_choices.forEach(function (t) {
      var opt = document.createElement("option");
      opt.value = String(t);
      opt.textContent = t + " yr";
      tSel.appendChild(opt);
    });

    var aSel = $("input-alpha_preset");
    aSel.innerHTML = "";
    Object.keys(presets.alpha_presets).forEach(function (key) {
      var opt = document.createElement("option");
      opt.value = key;
      opt.textContent = presets.alpha_presets[key].label;
      aSel.appendChild(opt);
    });

    config = clampConfig(config);
    updateAllBoundsAndValues();
    recomputeDerived();
    setControlsEnabled(true);
  }

  // ---- validation ----

  function validateConfig(cfg) {
    var errs = [];
    if (!presets) return errs;

    if (presets.T_choices.indexOf(cfg.T) === -1) {
      errs.push("T must be one of " + presets.T_choices.join(", "));
    }
    var N = Math.round(cfg.T * presets.dates_per_year);
    var nMaxCap = Math.min(presets.n_max_cap, N);
    if (!(cfg.n_max >= 1 && cfg.n_max <= nMaxCap)) {
      errs.push("n_max must be between 1 and " + nMaxCap);
    }
    if (!(cfg.n_min >= 0 && cfg.n_min <= cfg.n_max)) {
      errs.push("n_min must be between 0 and n_max (" + cfg.n_max + ")");
    }
    var preset = presets.alpha_presets[cfg.alpha_preset];
    if (!preset) {
      errs.push("Unknown commodity preset '" + cfg.alpha_preset + "'");
      return errs;
    }
    if (!(cfg.kappa >= presets.ranges.kappa.min && cfg.kappa <= presets.ranges.kappa.max)) {
      errs.push("κ must be between " + presets.ranges.kappa.min + " and " + presets.ranges.kappa.max);
    }
    if (!(cfg.sigma >= presets.ranges.sigma.min && cfg.sigma <= preset.sigma_max)) {
      errs.push("σ must be between " + presets.ranges.sigma.min + " and " + preset.sigma_max + " for this preset");
    }
    if (!(cfg.r >= presets.ranges.r.min && cfg.r <= presets.ranges.r.max)) {
      errs.push("r must be between " + presets.ranges.r.min + " and " + presets.ranges.r.max);
    }
    var kMin = presets.ranges.K.min > 0 ? presets.ranges.K.min : presets.ranges.K.step;
    if (!(cfg.K > 0 && cfg.K <= preset.K_max)) {
      errs.push("K must be between " + kMin + " and " + preset.K_max + " for this preset");
    }
    if (!(cfg.q_max > 0)) {
      errs.push("q_max must be greater than 0");
    }
    return errs;
  }

  function formatServerErrors(detail) {
    if (Array.isArray(detail)) {
      return detail.map(function (d) {
        var loc = Array.isArray(d.loc) ? d.loc.filter(function (p) { return p !== "body"; }).join(".") : "";
        return (loc ? loc + ": " : "") + (d.msg || JSON.stringify(d));
      });
    }
    if (typeof detail === "string") return [detail];
    return ["The server rejected this configuration."];
  }

  // ---- solving ----

  function setBusy(isBusy) {
    $("btn-calculate").disabled = isBusy;
    $("status-msg").textContent = isBusy ? "Solving — this can take up to a minute on a cold container…" : "";
  }

  function doSolve() {
    var errs = validateConfig(config);
    if (errs.length) {
      showErrorList(errs);
      return;
    }
    hideError();
    setBusy(true);

    fetch(API_BASE + "/solve", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(config)
    })
      .then(function (res) {
        if (res.status === 422) {
          return res.json().catch(function () { return {}; }).then(function (body) {
            showErrorList(formatServerErrors(body.detail));
            return null;
          });
        }
        if (!res.ok) {
          throw new Error("Server error (" + res.status + ")");
        }
        return res.json();
      })
      .then(function (data) {
        if (data) renderResult(data);
      })
      .catch(function (err) {
        showRetryable("Could not reach the solver (" + err.message + ").", doSolve);
      })
      .finally(function () {
        setBusy(false);
      });
  }

  // ---- error / status UI ----

  function showErrorList(errs) {
    var box = $("error-msg");
    box.innerHTML = "";
    if (!errs || !errs.length) {
      box.hidden = true;
      return;
    }
    if (errs.length === 1) {
      box.textContent = errs[0];
    } else {
      var ul = document.createElement("ul");
      errs.forEach(function (e) {
        var li = document.createElement("li");
        li.textContent = e;
        ul.appendChild(li);
      });
      box.appendChild(ul);
    }
    box.hidden = false;
  }

  function hideError() {
    showErrorList([]);
  }

  function showRetryable(msg, retryFn) {
    var box = $("error-msg");
    box.innerHTML = "";
    var span = document.createElement("span");
    span.textContent = msg + " ";
    var btn = document.createElement("button");
    btn.type = "button";
    btn.textContent = "Retry";
    btn.addEventListener("click", function () {
      hideError();
      retryFn();
    });
    box.appendChild(span);
    box.appendChild(btn);
    box.hidden = false;
  }

  function setStatus(msg) {
    $("status-msg").textContent = msg;
    if (statusTimer) clearTimeout(statusTimer);
    statusTimer = setTimeout(function () {
      if ($("status-msg").textContent === msg) $("status-msg").textContent = "";
    }, 4000);
  }

  // ---- permalink ----

  function configToQueryString(cfg) {
    var params = new URLSearchParams();
    CONFIG_KEYS.forEach(function (k) { params.set(k, cfg[k]); });
    return params.toString();
  }

  function fallbackCopyToClipboard(text) {
    var ta = document.createElement("textarea");
    ta.value = text;
    ta.style.position = "fixed";
    ta.style.opacity = "0";
    document.body.appendChild(ta);
    ta.focus();
    ta.select();
    try { document.execCommand("copy"); } catch (e) { /* best effort */ }
    document.body.removeChild(ta);
  }

  function copyLink() {
    var qs = configToQueryString(config);
    var url = window.location.origin + window.location.pathname + "?" + qs;
    try { window.history.replaceState(null, "", "?" + qs); } catch (e) { /* file:// or sandboxed */ }

    if (navigator.clipboard && window.isSecureContext) {
      navigator.clipboard.writeText(url).then(
        function () { setStatus("Link copied to clipboard."); },
        function () { fallbackCopyToClipboard(url); setStatus("Link copied to clipboard."); }
      );
    } else {
      fallbackCopyToClipboard(url);
      setStatus("Link copied to clipboard.");
    }
  }

  function applyQueryString() {
    var qs = window.location.search;
    if (!qs) return;
    var params = new URLSearchParams(qs);
    var raw = {};
    CONFIG_KEYS.forEach(function (k) { raw[k] = config[k]; });

    var any = false;
    NUMERIC_KEYS.forEach(function (k) {
      if (params.has(k)) {
        var v = parseFloat(params.get(k));
        if (!Number.isNaN(v)) { raw[k] = v; any = true; }
      }
    });
    if (params.has("alpha_preset")) {
      var p = params.get("alpha_preset");
      if (presets.alpha_presets[p]) { raw.alpha_preset = p; any = true; }
    }
    if (!any) return;

    config = clampConfig(raw);
    updateAllBoundsAndValues();
    recomputeDerived();
    doSolve();
  }

  // ---- CSV export ----

  function downloadCsv() {
    if (!currentResult) return;
    var boundary = currentResult.boundary;
    var timeGrid = currentResult.time_grid;
    var volumeGrid = currentResult.volume_grid;
    var cfg = currentResult.config || config;

    var lines = [];
    lines.push(["t \\ C"].concat(volumeGrid).join(","));
    boundary.forEach(function (row, i) {
      var cells = [timeGrid[i]].concat(row.map(function (v) { return v === null || v === undefined ? "" : v; }));
      lines.push(cells.join(","));
    });
    var csv = lines.join("\r\n");

    var blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
    var url = URL.createObjectURL(blob);
    var name = "boundary_" + cfg.alpha_preset + "_T" + cfg.T + "_kappa" + cfg.kappa + "_sigma" + cfg.sigma + "_K" + cfg.K + ".csv";
    name = name.replace(/\s+/g, "");

    var a = document.createElement("a");
    a.href = url;
    a.download = name;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }

  // ---- wiring ----

  function bindEvents() {
    $("btn-calculate").addEventListener("click", doSolve);
    $("btn-copy-link").addEventListener("click", copyLink);
    $("btn-download-csv").addEventListener("click", downloadCsv);

    $("input-T").addEventListener("change", function () {
      if (!presets) return;
      config.T = parseFloat($("input-T").value);
      config = clampConfig(config);
      updateAllBoundsAndValues();
      recomputeDerived();
    });

    $("input-alpha_preset").addEventListener("change", function () {
      if (!presets) return;
      config.alpha_preset = $("input-alpha_preset").value;
      config = clampConfig(config);
      updateAllBoundsAndValues();
      recomputeDerived();
    });

    $("input-q_max").addEventListener("change", function () {
      if (!presets) return;
      var v = parseFloat($("input-q_max").value);
      if (Number.isNaN(v) || !(v > 0)) v = config.q_max;
      config.q_max = v;
      $("input-q_max").value = v;
    });

    SLIDER_KEYS.forEach(function (key) {
      var range = $("input-" + key);
      var num = $("input-" + key + "-num");
      function handle(source) {
        if (!presets) return;
        var v = parseFloat(source.value);
        if (Number.isNaN(v)) v = config[key];
        config[key] = v;
        config = clampConfig(config);
        updateAllBoundsAndValues();
        recomputeDerived();
      }
      range.addEventListener("input", function () { handle(range); });
      num.addEventListener("change", function () { handle(num); });
    });

    $("input-slice").addEventListener("input", function () {
      if (!currentResult) return;
      var idx = parseInt($("input-slice").value, 10);
      renderSlice(currentResult, idx);
    });
  }

  // ---- boot ----

  function fireHealthPing() {
    fetch(API_BASE + "/health").catch(function () { /* best-effort wake, ignore failures */ });
  }

  function loadPresetsAndMaybeApplyQuery() {
    fetch(API_BASE + "/presets")
      .then(function (res) {
        if (!res.ok) throw new Error("HTTP " + res.status);
        return res.json();
      })
      .then(function (data) {
        presets = data;
        buildControlsFromPresets();
        applyQueryString();
      })
      .catch(function (err) {
        showErrorList([
          "Could not load parameter ranges from the server (" + err.message + "). " +
          "The chart above still works; controls stay disabled until this succeeds. Try reloading the page."
        ]);
      });
  }

  function init() {
    var bundled = window.__BUNDLED_PRESET__;
    config = {};
    CONFIG_KEYS.forEach(function (k) { config[k] = bundled.config[k]; });

    renderResult(bundled);
    seedDisabledControls(config);
    bindEvents();

    fireHealthPing();
    loadPresetsAndMaybeApplyQuery();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
