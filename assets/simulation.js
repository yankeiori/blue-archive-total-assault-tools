/**
 * クライアントサイド Monte Carlo シミュレーションエンジン
 * サーバー側 (Python/NumPy) の処理をブラウザ側 JavaScript に移植
 */
(function () {
  "use strict";

  var ns = (window.dash_clientside = window.dash_clientside || {});
  ns.sim = {};

  // =========================================================================
  // 定数
  // =========================================================================
  var N_SAMPLES = 100000;
  var N_CUTOFF_SAMPLES = 100000;

  // 減衰関数は cos.js (ns.cos) に一元管理。cos.js はファイル名昇順で
  // simulation.js より先にロードされるため参照できる (正本は
  // app/backend/simulation.py の DAMAGE_FUNC)。
  var decay = ns.cos.decay;
  var inverseDecay = ns.cos.inverseDecay;

  // =========================================================================
  // ヘルパー
  // =========================================================================
  function buildParams(values, ids) {
    var params = {};
    for (var i = 0; i < ids.length; i++) {
      var idx = ids[i].index;
      var param = ids[i].param;
      if (!params[idx]) params[idx] = {};
      params[idx][param] = values[i];
    }
    return params;
  }

  function extractHitParams(
    indices,
    params,
    globalCrit,
    globalEvade,
    damageMode,
    globalStab
  ) {
    var hp = {
      critLows: [],
      critHighs: [],
      normalLows: [],
      normalHighs: [],
      critRates: [],
      evadeRates: [],
      hpDeps: [],
    };
    for (var ii = 0; ii < indices.length; ii++) {
      var p = params[indices[ii]];
      if (!p) continue;
      var critMin = parseFloat(p.crit_min || 0);
      var critMax = parseFloat(p.crit_max || 0);
      var normalMin = parseFloat(p.normal_min || 0);
      var normalMax = parseFloat(p.normal_max || 0);
      var hits = parseInt(p.hits || 1);
      // 敵の数を Hit 数に掛けて総ヒット数とする (未入力/不正は 1)。
      var enemies = parseInt(p.enemies || 1);
      if (!(enemies >= 1)) enemies = 1;
      var totalHits = hits * enemies;
      var cr =
        (p.crit_rate != null
          ? parseFloat(p.crit_rate)
          : parseFloat(globalCrit || 0)) / 100;
      var er =
        (p.evade_rate != null
          ? parseFloat(p.evade_rate)
          : parseFloat(globalEvade || 0)) / 100;
      // 安定値は全体設定 (サイドバー) から取得する。
      var stab = globalStab != null && globalStab !== "" ? parseFloat(globalStab) : null;

      // 安定値考慮の減衰前下限・上限 (cos.js に一元化)。最大がキャップ張り付き時は
      // 最小から逆算する。
      var cb = ns.cos.rawBounds(critMin, critMax, stab, damageMode);
      var nb = ns.cos.rawBounds(normalMin, normalMax, stab, damageMode);
      var rcl = cb[0], rch = cb[1], rnl = nb[0], rnh = nb[1];

      var dep = ns.cos.hpDepFlag(p);
      for (var h = 0; h < totalHits; h++) {
        hp.critLows.push(rcl);
        hp.critHighs.push(rch);
        hp.normalLows.push(rnl);
        hp.normalHighs.push(rnh);
        hp.critRates.push(cr);
        hp.evadeRates.push(er);
        hp.hpDeps.push(dep);
      }
    }
    return hp;
  }

  // =========================================================================
  // Monte Carlo シミュレーション
  // =========================================================================
  function simulate(hp, nSamples) {
    var nHits = hp.critLows.length;
    if (nHits === 0) return new Float64Array(nSamples);
    var totals = new Float64Array(nSamples);
    for (var s = 0; s < nSamples; s++) {
      var total = 0;
      for (var h = 0; h < nHits; h++) {
        if (Math.random() < hp.evadeRates[h]) continue;
        var raw;
        if (Math.random() < hp.critRates[h]) {
          raw =
            hp.critLows[h] +
            Math.random() * (hp.critHighs[h] - hp.critLows[h]);
        } else {
          raw =
            hp.normalLows[h] +
            Math.random() * (hp.normalHighs[h] - hp.normalLows[h]);
        }
        total += decay(raw);
      }
      totals[s] = total;
    }
    return totals;
  }

  // =========================================================================
  // ヒストグラム (事前ビニング)
  // =========================================================================
  function computeHistogram(samples, nbins) {
    var min = Infinity,
      max = -Infinity;
    var mean = 0;
    for (var i = 0; i < samples.length; i++) {
      if (samples[i] < min) min = samples[i];
      if (samples[i] > max) max = samples[i];
      mean += samples[i];
    }
    mean /= samples.length;
    if (min === max) max = min + 1;

    var binWidth = (max - min) / nbins;
    var counts = new Array(nbins);
    for (var i = 0; i < nbins; i++) counts[i] = 0;
    for (var i = 0; i < samples.length; i++) {
      var bin = Math.floor((samples[i] - min) / binWidth);
      if (bin >= nbins) bin = nbins - 1;
      counts[bin]++;
    }
    var centers = new Array(nbins);
    for (var i = 0; i < nbins; i++) {
      centers[i] = min + i * binWidth + binWidth / 2;
    }
    return { x: centers, y: counts, binWidth: binWidth, mean: mean };
  }

  // =========================================================================
  // CDF テーブル (足切り用)。{grid, cdf, min, max} を ns.cos が補間する。
  //   COS 経路: ns.cos.cdfTable()。MC 経路: ns.cos.tableFromSamples()。
  // =========================================================================
  var exceedanceProb = ns.cos.exceedanceProb;
  var valueAtExceedance = ns.cos.valueAtExceedance;

  // 指定カード群の合計ダメージ MC をソート済みで返す (足切り MC 経路用)。
  function simulateSorted(hp, nSamples) {
    var samples = simulate(hp, nSamples);
    var arr = Array.prototype.slice.call(samples);
    arr.sort(function (a, b) { return a - b; });
    return arr;
  }

  // 足切り用 CDF テーブルを構築 (method='cos' は解析、'mc' は MC サンプル)。
  function buildCutoffTable(indices, params, globalCrit, globalEvade, damageMode, method, globalStab) {
    if (method === "mc") {
      var hp = extractHitParams(indices, params, globalCrit, globalEvade, damageMode, globalStab);
      return ns.cos.tableFromSamples(simulateSorted(hp, N_CUTOFF_SAMPLES));
    }
    // 足切りは和モデル (HP非依存)。COS で CDF テーブルを直接構築。
    return ns.cos.cdfTable({
      indices: indices, params: params, globalCrit: globalCrit,
      globalEvade: globalEvade, damageMode: damageMode, hpMode: "off",
      globalStability: globalStab,
    });
  }

  // HP依存 (積/混在モデル) の累積ダメージ MC。
  // HP依存ヒット: H_{n+1} = H_n - (βH_n + R0)·decay(raw)、通常ヒット: H_n - decay(raw)。
  // 全ヒット HP依存なら従来の積モデル MC と同一。
  function simulateMixed(hp, hpP, nSamples) {
    var nHits = hp.critLows.length;
    var H = parseFloat(hpP.H), H1 = parseFloat(hpP.H1), R0 = parseFloat(hpP.R0), R1 = parseFloat(hpP.R1);
    var beta = (R1 - R0) / H;
    var out = new Float64Array(nSamples);
    for (var s = 0; s < nSamples; s++) {
      var Hn = H1;
      for (var h = 0; h < nHits; h++) {
        var x = 0;
        if (Math.random() >= hp.evadeRates[h]) {
          var raw;
          if (Math.random() < hp.critRates[h]) raw = hp.critLows[h] + Math.random() * (hp.critHighs[h] - hp.critLows[h]);
          else raw = hp.normalLows[h] + Math.random() * (hp.normalHighs[h] - hp.normalLows[h]);
          x = decay(raw);
        }
        if (hp.hpDeps[h]) Hn -= (beta * Hn + R0) * x;
        else Hn -= x;
      }
      out[s] = H1 - Hn;
    }
    return out;
  }

  // =========================================================================
  // 対数スライダー変換
  // =========================================================================
  function logSliderToPct(val) {
    return Math.pow(10, val / 100 - 2);
  }

  function pctToLogSlider(pct) {
    if (pct <= 0) return 0;
    var val = (Math.log10(pct) + 2) * 100;
    return Math.max(0, Math.min(400, val));
  }

  // =========================================================================
  // 数値フォーマット
  // =========================================================================
  function fmt(n) {
    return Math.round(n).toLocaleString();
  }

  // =========================================================================
  // triggered_id パース (pattern-matching callback 用)
  // =========================================================================
  function getTriggeredId() {
    var ctx = window.dash_clientside.callback_context;
    if (!ctx.triggered.length) return null;
    var propId = ctx.triggered[0].prop_id;
    var idStr = propId.substring(0, propId.lastIndexOf("."));
    try {
      return JSON.parse(idStr);
    } catch (e) {
      return idStr;
    }
  }

  function getTriggeredSimpleId() {
    var ctx = window.dash_clientside.callback_context;
    if (!ctx.triggered.length) return null;
    var propId = ctx.triggered[0].prop_id;
    return propId.substring(0, propId.lastIndexOf("."));
  }

  // =========================================================================
  // コールバック: ドラッグ順同期
  // =========================================================================
  ns.sim.syncDragOrder = function (dragOrder, indices) {
    if (!dragOrder) return indices;
    try {
      return JSON.parse(dragOrder);
    } catch (e) {
      return indices;
    }
  };

  // =========================================================================
  // コールバック: 世代カウンタ
  // =========================================================================
  ns.sim.incrementGeneration = function () {
    // 入力 (card-indices / params / 一括率 / damage-mode / calc-method 等) のいずれかが
    // 変わると足切りキャッシュ世代を更新する。currentGen は常に最後の State。
    var a = arguments;
    return (a[a.length - 1] || 0) + 1;
  };

  // HP依存パラメータ入力欄の表示切替 (hp-mode → div#hp-params の style)
  ns.sim.toggleHpParams = function (hpMode) {
    return hpMode === "on"
      ? { display: "block", marginTop: "4px" }
      : { display: "none", marginTop: "4px" };
  };

  // =========================================================================
  // コールバック: 一括適用
  // =========================================================================
  ns.sim.applyGlobal = function (
    nClicks,
    globalCrit,
    globalEvade,
    currentCritValues
  ) {
    var n = currentCritValues.length;
    var crits = [];
    var evades = [];
    for (var i = 0; i < n; i++) {
      crits.push(globalCrit);
      evades.push(globalEvade);
    }
    return [crits, evades];
  };

  // =========================================================================
  // コールバック: シミュレーション実行
  // =========================================================================
  // 期待値と目標の縦線が近いか (ラベルが被るか) の判定。
  // xRange があれば表示幅比で、無ければ値の相対差で判定する。
  function markersOverlap(mean, target, xRange) {
    if (!(target > 0)) return false;
    var span = xRange ? xRange[1] - xRange[0]
                      : Math.max(Math.abs(mean), Math.abs(target));
    return span > 0 && Math.abs(mean - target) < 0.22 * span;
  }

  // ダメージ分布図の縦線 (期待値・目標) shape/annotation を作る。
  // 縦線が近いときは目標ラベルを 1 行ぶん上へずらして重なりを避ける。
  function markerShapes(mean, target, xRange) {
    var shapes = [{ type: "line", x0: mean, x1: mean, y0: 0, y1: 1, yref: "paper",
                    line: { dash: "dash", color: "red" } }];
    var annotations = [{ x: mean, y: 1, yref: "paper", text: "期待値: " + fmt(mean),
                         showarrow: false, yanchor: "bottom", font: { color: "red" } }];
    if (target > 0) {
      var lift = markersOverlap(mean, target, xRange) ? 16 : 0;
      shapes.push({ type: "line", x0: target, x1: target, y0: 0, y1: 1, yref: "paper",
                    line: { color: "green" } });
      annotations.push({ x: target, y: 1, yref: "paper", text: "目標: " + fmt(target),
                         showarrow: false, yanchor: "bottom", yshift: lift,
                         font: { color: "green" } });
    }
    return { shapes: shapes, annotations: annotations };
  }

  // 累積分布関数 (CDF) の図を作る。xs 昇順, cdf は [0,1] の単調非減少。
  // 目標が指定されていれば縦線 + その点の通過確率 (1-CDF) を注記する。
  function cdfFigure(xs, cdf, mean, target, title, xLabel, xRange) {
    // それ以上になる確率 P(D≥x) = 1 - CDF を表示する (反転)。
    var y = new Array(cdf.length);
    for (var i = 0; i < cdf.length; i++) y[i] = (1 - cdf[i]) * 100;
    var shapes = [{ type: "line", x0: mean, x1: mean, y0: 0, y1: 1, yref: "paper",
                    line: { dash: "dash", color: "red" } }];
    var annotations = [{ x: mean, y: 1, yref: "paper", text: "期待値: " + fmt(mean),
                         showarrow: false, yanchor: "bottom", font: { color: "red" } }];
    if (target > 0) {
      // target の CDF を線形補間し、通過確率 = 100 - CDF(%) を求める。
      var cdfAt = 0;
      if (target <= xs[0]) cdfAt = cdf[0];
      else if (target >= xs[xs.length - 1]) cdfAt = cdf[cdf.length - 1];
      else {
        for (var j = 1; j < xs.length; j++) {
          if (xs[j] >= target) {
            var denom = xs[j] - xs[j - 1];
            var t = denom > 0 ? (target - xs[j - 1]) / denom : 0;
            cdfAt = cdf[j - 1] + t * (cdf[j] - cdf[j - 1]);
            break;
          }
        }
      }
      var lift = markersOverlap(mean, target, xRange) ? 16 : 0;
      shapes.push({ type: "line", x0: target, x1: target, y0: 0, y1: 1, yref: "paper",
                    line: { color: "green" } });
      annotations.push({ x: target, y: 1, yref: "paper",
                         text: "目標: " + fmt(target) + " (通過 " + ((1 - cdfAt) * 100).toFixed(2) + "%)",
                         showarrow: false, yanchor: "bottom", yshift: lift,
                         font: { color: "green" } });
    }
    return {
      data: [{ x: Array.prototype.slice.call(xs), y: y, type: "scatter", mode: "lines",
               line: { color: "#7b1fa2" }, name: "P(D≥x)" }],
      layout: {
        title: title,
        xaxis: { title: { text: xLabel }, range: xRange || undefined },
        yaxis: { title: { text: "それ以上になる確率 (%)" }, range: [0, 100] },
        shapes: shapes, annotations: annotations,
      },
    };
  }

  // 分布の有効域 (中央 lowP〜highP の質量を覆う x 範囲) を返す。
  // weights は正規化不要 (pdf 値でも頻度でも可)。両端に少しパディングを付ける。
  function massRange(xs, weights, lowP, highP) {
    var total = 0;
    for (var i = 0; i < weights.length; i++) total += weights[i];
    if (!(total > 0) || xs.length === 0) return null;
    var loT = total * lowP;
    var hiT = total * highP;
    var cum = 0;
    var lo = xs[0];
    var hi = xs[xs.length - 1];
    var gotLo = false;
    for (var j = 0; j < xs.length; j++) {
      cum += weights[j];
      if (!gotLo && cum >= loT) {
        lo = xs[j];
        gotLo = true;
      }
      if (cum >= hiT) {
        hi = xs[j];
        break;
      }
    }
    var span = hi - lo;
    var pad = span > 0 ? span * 0.03 : Math.max(1, Math.abs(lo) * 0.05);
    return [lo - pad, hi + pad];
  }

  ns.sim.runSimulation = function (
    nClicks,
    values,
    ids,
    sortedIndices,
    cardIndices,
    globalCrit,
    globalEvade,
    globalStab,
    targetDamage,
    damageMode,
    method,
    hpMode,
    hpH,
    hpH1,
    hpR0,
    hpR1
  ) {
    if (!nClicks) throw window.dash_clientside.PreventUpdate;

    var indices =
      sortedIndices && sortedIndices.length ? sortedIndices : cardIndices;
    indices = indices.filter(function (x) {
      return typeof x === "number";
    });

    var params = buildParams(values, ids);
    var target = parseFloat(targetDamage || 0);
    var hp = { H: hpH, H1: hpH1, R0: hpR0, R1: hpR1 };
    var anyNormal = hpMode === "on" && indices.some(function (i) {
      return params[i] && !ns.cos.hpDepFlag(params[i]);
    });
    var title = hpMode === "on"
      ? (anyNormal ? "累積ダメージ分布 (HP依存+通常混在)" : "累積ダメージ分布 (HP依存)")
      : "合計ダメージ分布";

    // -------- COS 法 (準厳密) --------
    if (method !== "mc") {
      var dist = ns.cos.distribution(
        {
          indices: indices, params: params, globalCrit: globalCrit,
          globalEvade: globalEvade, damageMode: damageMode,
          hpMode: hpMode, hp: hp, globalStability: globalStab,
        },
        600
      );
      if (!dist) throw window.dash_clientside.PreventUpdate;
      var xRangeCos = massRange(dist.x, dist.pdf, 0.001, 0.999);
      var mk = markerShapes(dist.mean, target, xRangeCos);
      var passText = "";
      if (target > 0) {
        // 通過確率 = P(D >= target) = 1 - CDF(target) を grid 上で補間
        var tbl = { grid: dist.x, cdf: dist.cdf };
        var ex = ns.cos.exceedanceProb(tbl, target);
        passText = "目標ダメージ " + fmt(target) + " の通過確率: " + ex.toFixed(2) + "% (COS法)";
      }
      var xLabelCos = hpMode === "on" ? "累積ダメージ" : "合計ダメージ";
      var figureCos = {
        data: [{ x: dist.x, y: dist.pdf, type: "scatter", mode: "lines",
                 fill: "tozeroy", name: "ダメージ密度 (COS法)" }],
        layout: {
          title: title + " — COS 法",
          xaxis: {
            title: { text: xLabelCos },
            range: xRangeCos || undefined,
          },
          yaxis: { title: { text: "確率密度" } },
          shapes: mk.shapes, annotations: mk.annotations,
        },
      };
      var cdfFigCos = cdfFigure(dist.x, dist.cdf, dist.mean, target,
                                "それ以上になる確率 P(D≥x) — COS 法", xLabelCos, xRangeCos);
      var tblCos = { grid: Array.prototype.slice.call(dist.x), cdf: Array.prototype.slice.call(dist.cdf) };
      return [figureCos, passText, cdfFigCos, tblCos, target > 0 ? target : null];
    }

    // -------- モンテカルロ --------
    var hitParams = extractHitParams(indices, params, globalCrit, globalEvade, damageMode, globalStab);
    var totalDamage =
      hpMode === "on" ? simulateMixed(hitParams, hp, N_SAMPLES) : simulate(hitParams, N_SAMPLES);

    var hist = computeHistogram(totalDamage, 200);
    var xRangeMc = massRange(hist.x, hist.y, 0.001, 0.999);
    var mkm = markerShapes(hist.mean, target, xRangeMc);
    var passTextMc = "";
    if (target > 0) {
      var passCount = 0;
      for (var i = 0; i < totalDamage.length; i++) if (totalDamage[i] >= target) passCount++;
      var passRate = (passCount / totalDamage.length) * 100;
      passTextMc =
        "目標ダメージ " + fmt(target) + " の通過確率: " + passRate.toFixed(2) + "% (MC)";
    }
    var xLabelMc = hpMode === "on" ? "累積ダメージ" : "合計ダメージ";
    var figure = {
      data: [{ x: hist.x, y: hist.y, type: "bar", width: hist.binWidth * 0.95, name: "ダメージ分布" }],
      layout: {
        title: title + " — モンテカルロ",
        xaxis: {
          title: { text: xLabelMc },
          range: xRangeMc || undefined,
        },
        yaxis: { title: { text: "頻度" } },
        bargap: 0.05,
        shapes: mkm.shapes, annotations: mkm.annotations,
      },
    };
    // ヒストグラム頻度の累積和から経験 CDF を構築 (x = 各ビンの右端)。
    var totalCount = totalDamage.length;
    var cdfXs = new Array(hist.x.length);
    var cdfYs = new Array(hist.x.length);
    var run = 0;
    for (var c = 0; c < hist.x.length; c++) {
      run += hist.y[c];
      cdfXs[c] = hist.x[c] + hist.binWidth / 2;
      cdfYs[c] = run / totalCount;
    }
    var cdfFigMc = cdfFigure(cdfXs, cdfYs, hist.mean, target,
                             "それ以上になる確率 P(D≥x) — モンテカルロ", xLabelMc, xRangeMc);
    var tblMc = { grid: cdfXs, cdf: cdfYs };
    return [figure, passTextMc, cdfFigMc, tblMc, target > 0 ? target : null];
  };

  // =========================================================================
  // 通過確率 ⇄ ダメージ 変換 (CDF テーブルを逆引き)
  // =========================================================================
  // ダメージ → 通過確率 P(D≥x) を % 文字列で返す。
  ns.sim.damageToProb = function (damage, table) {
    if (!table || !table.grid || !table.grid.length) return "—";
    var d = parseFloat(damage);
    if (!isFinite(d)) return "—";
    return ns.cos.exceedanceProb(table, d).toFixed(2) + " %";
  };
  // 通過確率 (%) → 対応するダメージを返す。
  ns.sim.probToDamage = function (prob, table) {
    if (!table || !table.grid || !table.grid.length) return "—";
    var p = parseFloat(prob);
    if (!isFinite(p)) return "—";
    p = Math.min(100, Math.max(0, p));
    return ns.cos.valueAtExceedance(table, p).toLocaleString();
  };

  // =========================================================================
  // コールバック: 足切り計算
  // =========================================================================
  ns.sim.computeCutoff = function (
    nClicksList,
    sortedIndices,
    cardIndices,
    values,
    ids,
    globalCrit,
    globalEvade,
    globalStab,
    targetDamage,
    damageMode,
    method,
    prevDist,
    prevValues,
    generation,
    statusIds
  ) {
    var trigger = getTriggeredId();
    if (
      !trigger ||
      typeof trigger !== "object" ||
      !nClicksList.some(function (n) {
        return n;
      })
    ) {
      throw window.dash_clientside.PreventUpdate;
    }

    var cutoffIndex = trigger.index;
    var key = String(cutoffIndex);
    var order =
      sortedIndices && sortedIndices.length ? sortedIndices : cardIndices;

    var distStore = Object.assign({}, prevDist || {});
    var valuesStore = Object.assign({}, prevValues || {});

    var cached = distStore[key];
    var needRecompute = !cached || cached.generation !== generation;

    var upperTable, lowerTable;
    var statusAction;

    if (needRecompute) {
      var params = buildParams(values, ids);

      // order 内のカットオフ位置で上下に分割
      var marker = "cutoff_" + cutoffIndex;
      var pos = order.indexOf(marker);
      if (pos === -1) pos = order.length;
      var upperIndices = order.slice(0, pos).filter(function (x) {
        return typeof x === "number";
      });
      var lowerIndices = order.slice(pos + 1).filter(function (x) {
        return typeof x === "number";
      });

      // 足切りは和モデル (HP非依存) で上側・下側を独立に CDF テーブル化する。
      // method='cos' は解析、'mc' は MC サンプルから経験 CDF。
      upperTable = buildCutoffTable(upperIndices, params, globalCrit, globalEvade, damageMode, method, globalStab);
      lowerTable = buildCutoffTable(lowerIndices, params, globalCrit, globalEvade, damageMode, method, globalStab);

      distStore[key] = {
        upper_table: upperTable,
        lower_table: lowerTable,
        generation: generation,
      };
      statusAction = "再計算完了";
    } else {
      upperTable = cached.upper_table;
      lowerTable = cached.lower_table;
      statusAction = "キャッシュ利用";
    }

    var target = parseFloat(targetDamage || 0);
    var e2 = 50.0;
    var e1 = valueAtExceedance(upperTable, e2);
    var e3 = Math.max(target - e1, 0);
    var e4 = exceedanceProb(lowerTable, e3);

    valuesStore[key] = {
      e1: Math.round(e1),
      e2: Math.round(e2 * 100) / 100,
      e3: Math.round(e3),
      e4: Math.round(e4 * 100) / 100,
    };

    var upperRange = fmt(upperTable.min) + " ~ " + fmt(upperTable.max);
    var lowerRange = fmt(lowerTable.min) + " ~ " + fmt(lowerTable.max);
    var methodLabel = method === "mc" ? "MC" : "COS法";
    var statusMsg =
      statusAction +
      " (" + methodLabel + ", 足切りは和モデル)" +
      " | 上側: " +
      upperRange +
      " | 下側: " +
      lowerRange;

    var statusTexts = statusIds.map(function (sid) {
      return sid.index === cutoffIndex
        ? statusMsg
        : window.dash_clientside.no_update;
    });

    return [distStore, valuesStore, statusTexts];
  };

  // =========================================================================
  // コールバック: 足切りスライダー変更
  // =========================================================================
  ns.sim.onSliderChange = function (
    sliderVals,
    currentValues,
    dist,
    targetDamage,
    sliderIds
  ) {
    var trigger = getTriggeredId();
    if (!trigger || typeof trigger !== "object")
      throw window.dash_clientside.PreventUpdate;

    var cutoffKey = String(trigger.index);
    var elem = trigger.elem;

    if (!dist || !dist[cutoffKey]) throw window.dash_clientside.PreventUpdate;

    var idxInList = -1;
    for (var i = 0; i < sliderIds.length; i++) {
      if (
        sliderIds[i].elem === elem &&
        sliderIds[i].index === trigger.index
      ) {
        idxInList = i;
        break;
      }
    }
    if (idxInList === -1) throw window.dash_clientside.PreventUpdate;

    var val = sliderVals[idxInList];
    if (val === null || val === undefined)
      throw window.dash_clientside.PreventUpdate;
    val = parseFloat(val);

    var isPct = elem === "e2" || elem === "e4";
    var pctVal = isPct ? logSliderToPct(val) : val;

    // エコーバック防止
    if (currentValues) {
      var cardValues = currentValues[cutoffKey];
      if (cardValues) {
        var currentStored = parseFloat(
          cardValues[elem] !== undefined ? cardValues[elem] : Infinity
        );
        if (isPct) {
          var currentLog = pctToLogSlider(currentStored);
          if (Math.abs(val - currentLog) < 1.5)
            throw window.dash_clientside.PreventUpdate;
        } else {
          if (Math.abs(val - currentStored) < 0.5)
            throw window.dash_clientside.PreventUpdate;
        }
      }
    }

    var target = parseFloat(targetDamage || 0);
    var upper = dist[cutoffKey].upper_table;
    var lower = dist[cutoffKey].lower_table;

    var e1, e2, e3, e4;
    if (elem === "e1") {
      e1 = pctVal;
      e2 = exceedanceProb(upper, e1);
      e3 = Math.max(target - e1, 0);
      e4 = exceedanceProb(lower, e3);
    } else if (elem === "e2") {
      e2 = pctVal;
      e1 = valueAtExceedance(upper, e2);
      e3 = Math.max(target - e1, 0);
      e4 = exceedanceProb(lower, e3);
    } else if (elem === "e3") {
      e3 = pctVal;
      e1 = Math.max(target - e3, 0);
      e2 = exceedanceProb(upper, e1);
      e4 = exceedanceProb(lower, e3);
    } else if (elem === "e4") {
      e4 = pctVal;
      e3 = valueAtExceedance(lower, e4);
      e1 = Math.max(target - e3, 0);
      e2 = exceedanceProb(upper, e1);
    } else {
      throw window.dash_clientside.PreventUpdate;
    }

    var newValues = Object.assign({}, currentValues || {});
    newValues[cutoffKey] = {
      e1: Math.round(e1),
      e2: Math.round(e2 * 100) / 100,
      e3: Math.round(e3),
      e4: Math.round(e4 * 100) / 100,
    };
    return newValues;
  };

  // =========================================================================
  // コールバック: 足切りスライダー表示更新
  // =========================================================================
  ns.sim.updateCutoffDisplay = function (values, dist, sliderIds) {
    if (!values || !dist) throw window.dash_clientside.PreventUpdate;

    var sVals = [],
      sMins = [],
      sMaxs = [];
    for (var i = 0; i < sliderIds.length; i++) {
      var key = String(sliderIds[i].index);
      var elem = sliderIds[i].elem;
      var cardVals = values[key];
      var cardDist = dist[key];

      if (cardVals) {
        var raw = cardVals[elem] || 0;
        if (elem === "e2" || elem === "e4") {
          sVals.push(pctToLogSlider(parseFloat(raw)));
        } else {
          sVals.push(raw);
        }
      } else {
        sVals.push(window.dash_clientside.no_update);
      }

      if (cardDist) {
        if (elem === "e1") {
          sMins.push(cardDist.upper_table.min || 0);
          sMaxs.push(cardDist.upper_table.max || 10000000);
        } else if (elem === "e3") {
          sMins.push(cardDist.lower_table.min || 0);
          sMaxs.push(cardDist.lower_table.max || 10000000);
        } else {
          sMins.push(window.dash_clientside.no_update);
          sMaxs.push(window.dash_clientside.no_update);
        }
      } else {
        sMins.push(window.dash_clientside.no_update);
        sMaxs.push(window.dash_clientside.no_update);
      }
    }
    return [sVals, sMins, sMaxs];
  };

  // =========================================================================
  // コールバック: % Input 直接入力
  // =========================================================================
  ns.sim.onPctInputChange = function (
    inputVals,
    currentValues,
    dist,
    targetDamage,
    inputIds
  ) {
    var trigger = getTriggeredId();
    if (!trigger || typeof trigger !== "object")
      throw window.dash_clientside.PreventUpdate;

    var cutoffKey = String(trigger.index);
    var elem = trigger.elem;

    if (!dist || !dist[cutoffKey]) throw window.dash_clientside.PreventUpdate;

    var idxInList = -1;
    for (var i = 0; i < inputIds.length; i++) {
      if (
        inputIds[i].elem === elem &&
        inputIds[i].index === trigger.index
      ) {
        idxInList = i;
        break;
      }
    }
    if (idxInList === -1) throw window.dash_clientside.PreventUpdate;

    var val = inputVals[idxInList];
    if (val === null || val === undefined)
      throw window.dash_clientside.PreventUpdate;
    val = Math.max(0.01, Math.min(100, parseFloat(val)));

    // エコーバック防止
    if (currentValues) {
      var cardValues = currentValues[cutoffKey];
      if (cardValues) {
        var currentStored = parseFloat(
          cardValues[elem] !== undefined ? cardValues[elem] : Infinity
        );
        if (Math.abs(val - currentStored) < 0.005)
          throw window.dash_clientside.PreventUpdate;
      }
    }

    var target = parseFloat(targetDamage || 0);
    var upper = dist[cutoffKey].upper_table;
    var lower = dist[cutoffKey].lower_table;

    var e1, e2, e3, e4;
    if (elem === "e2") {
      e2 = val;
      e1 = valueAtExceedance(upper, e2);
      e3 = Math.max(target - e1, 0);
      e4 = exceedanceProb(lower, e3);
    } else if (elem === "e4") {
      e4 = val;
      e3 = valueAtExceedance(lower, e4);
      e1 = Math.max(target - e3, 0);
      e2 = exceedanceProb(upper, e1);
    } else {
      throw window.dash_clientside.PreventUpdate;
    }

    var newValues = Object.assign({}, currentValues || {});
    newValues[cutoffKey] = {
      e1: Math.round(e1),
      e2: Math.round(e2 * 100) / 100,
      e3: Math.round(e3),
      e4: Math.round(e4 * 100) / 100,
    };
    return newValues;
  };

  // =========================================================================
  // コールバック: Store → % Input 表示同期
  // =========================================================================
  ns.sim.updatePctInputDisplay = function (values, inputIds) {
    if (!values) throw window.dash_clientside.PreventUpdate;
    var out = [];
    for (var i = 0; i < inputIds.length; i++) {
      var key = String(inputIds[i].index);
      var elem = inputIds[i].elem;
      var cardVals = values[key];
      if (cardVals) {
        out.push(Math.round(parseFloat(cardVals[elem] || 0) * 100) / 100);
      } else {
        out.push(window.dash_clientside.no_update);
      }
    }
    return out;
  };

  // =========================================================================
  // コールバック: スライダー → % Input リアルタイム同期
  // =========================================================================
  ns.sim.sliderToPctSync = function (sliderValues, sliderIds, inputIds) {
    var out = [];
    for (var j = 0; j < inputIds.length; j++) {
      var found = false;
      for (var i = 0; i < sliderIds.length; i++) {
        if (
          sliderIds[i].elem === inputIds[j].elem &&
          sliderIds[i].index === inputIds[j].index
        ) {
          var v = sliderValues[i];
          if (v === null || v === undefined) {
            out.push(0.01);
          } else {
            var pct = Math.pow(10, v / 100 - 2);
            out.push(Math.round(pct * 100) / 100);
          }
          found = true;
          break;
        }
      }
      if (!found) out.push(window.dash_clientside.no_update);
    }
    return out;
  };

  // =========================================================================
  // コールバック: マニュアルモーダル開閉
  // =========================================================================
  ns.sim.toggleManualModal = function (openClicks, closeClicks) {
    var id = getTriggeredSimpleId();
    if (id === "open-manual-btn") return { display: "flex" };
    return { display: "none" };
  };

  // =========================================================================
  // コールバック: 画面スニップ (getDisplayMedia → 範囲選択 → data URL)
  // =========================================================================
  // getDisplayMedia で画面/ウィンドウを取り込み、フルスクリーンの
  // オーバーレイ上でドラッグ範囲選択した矩形を切り出して data URL を返す。
  // 返り値の Promise を Store(ocr-image-store) に流し、サーバー側 OCR を起動。
  ns.sim.snipScreen = function (nClicks) {
    if (!nClicks) throw window.dash_clientside.PreventUpdate;
    var noUpdate = window.dash_clientside.no_update;

    if (!navigator.mediaDevices || !navigator.mediaDevices.getDisplayMedia) {
      window.alert("このブラウザは画面キャプチャ(getDisplayMedia)に未対応です。画像アップロードをご利用ください。");
      return Promise.resolve(noUpdate);
    }

    return navigator.mediaDevices
      .getDisplayMedia({ video: { cursor: "never" }, audio: false })
      .then(function (stream) {
        var video = document.createElement("video");
        video.srcObject = stream;
        return video.play().then(function () {
          // 1 フレームを丸ごとキャンバスへ描画
          var full = document.createElement("canvas");
          full.width = video.videoWidth;
          full.height = video.videoHeight;
          full.getContext("2d").drawImage(video, 0, 0);
          stream.getTracks().forEach(function (t) {
            t.stop();
          });
          return cropInteractively(full);
        });
      })
      .catch(function (err) {
        // ユーザーがキャンセルした場合等は黙って終了
        if (err && err.name === "NotAllowedError") return noUpdate;
        window.alert("画面キャプチャに失敗しました: " + (err && err.message ? err.message : err));
        return noUpdate;
      });
  };

  // 取り込んだ画像をオーバーレイ表示し、ドラッグ矩形で切り出す。
  // 選択中は範囲外を暗転(スポットライト)+ サイズ表示し、ドラッグ終了後は
  // 確認バー(取り込む / 選び直す / キャンセル)を出してから確定する。
  function cropInteractively(sourceCanvas) {
    return new Promise(function (resolve) {
      var noUpdate = window.dash_clientside.no_update;

      var overlay = document.createElement("div");
      overlay.style.cssText =
        "position:fixed;inset:0;z-index:100000;cursor:crosshair;" +
        "display:flex;flex-direction:column;align-items:center;justify-content:center;";

      // 上部ツールバー (ヒント + 確認ボタン)
      var bar = document.createElement("div");
      bar.style.cssText =
        "position:fixed;top:0;left:0;right:0;z-index:100002;display:flex;" +
        "align-items:center;gap:12px;padding:8px 12px;background:rgba(0,0,0,0.78);" +
        "color:#fff;font:14px sans-serif;";
      var hint = document.createElement("span");
      hint.textContent = "取り込む範囲をドラッグで選択してください (Esc でキャンセル)";
      bar.appendChild(hint);

      function mkBtn(label, bg) {
        var b = document.createElement("button");
        b.textContent = label;
        b.style.cssText =
          "border:none;border-radius:4px;padding:6px 14px;cursor:pointer;" +
          "font:14px sans-serif;color:#fff;background:" + bg + ";display:none;";
        return b;
      }
      var okBtn = mkBtn("✅ この範囲で取り込む", "#4a90d9");
      var redoBtn = mkBtn("↺ 選び直す", "#888");
      var cancelBtn = mkBtn("✕ キャンセル", "#d63031");
      cancelBtn.style.display = "inline-block"; // キャンセルは常時表示
      cancelBtn.style.marginLeft = "auto";
      bar.appendChild(okBtn);
      bar.appendChild(redoBtn);
      bar.appendChild(cancelBtn);

      // 画面に収まるよう縮小表示。表示倍率を保持して切り出しに使う。
      var maxW = window.innerWidth * 0.96;
      var maxH = window.innerHeight * 0.86;
      var scale = Math.min(1, maxW / sourceCanvas.width, maxH / sourceCanvas.height);
      var dispCanvas = document.createElement("canvas");
      dispCanvas.width = sourceCanvas.width;
      dispCanvas.height = sourceCanvas.height;
      dispCanvas.getContext("2d").drawImage(sourceCanvas, 0, 0);
      dispCanvas.style.width = sourceCanvas.width * scale + "px";
      dispCanvas.style.height = sourceCanvas.height * scale + "px";
      dispCanvas.style.touchAction = "none";

      // 選択枠。box-shadow で範囲外を一括で暗転 (スポットライト)。
      var sel = document.createElement("div");
      sel.style.cssText =
        "position:fixed;z-index:100001;border:2px solid #4a90d9;" +
        "box-shadow:0 0 0 9999px rgba(0,0,0,0.6);display:none;pointer-events:none;";
      // サイズ表示ラベル
      var sizeLbl = document.createElement("div");
      sizeLbl.style.cssText =
        "position:fixed;z-index:100002;background:#4a90d9;color:#fff;" +
        "font:12px monospace;padding:2px 6px;border-radius:3px;display:none;pointer-events:none;";

      overlay.appendChild(dispCanvas);
      document.body.appendChild(overlay);
      document.body.appendChild(sel);
      document.body.appendChild(sizeLbl);
      document.body.appendChild(bar);

      var start = null;      // ドラッグ開始点 (client 座標)
      var lastRect = null;   // 確定待ちの選択矩形 (client 座標)

      function cleanup() {
        overlay.remove();
        sel.remove();
        sizeLbl.remove();
        bar.remove();
        document.removeEventListener("keydown", onKey);
        window.removeEventListener("pointermove", onMove);
        window.removeEventListener("pointerup", onUp);
      }
      function finishCancel() {
        cleanup();
        resolve(noUpdate);
      }
      function onKey(e) {
        if (e.key === "Escape") finishCancel();
        else if (e.key === "Enter" && lastRect) confirmCrop();
      }
      document.addEventListener("keydown", onKey);
      cancelBtn.addEventListener("click", finishCancel);

      function showConfirmUI(show) {
        okBtn.style.display = show ? "inline-block" : "none";
        redoBtn.style.display = show ? "inline-block" : "none";
        hint.textContent = show
          ? "この範囲でよろしいですか? (Enter で確定 / 選び直し可)"
          : "取り込む範囲をドラッグで選択してください (Esc でキャンセル)";
      }

      function updateSel(curX, curY) {
        var x = Math.min(start.x, curX);
        var y = Math.min(start.y, curY);
        var w = Math.abs(curX - start.x);
        var h = Math.abs(curY - start.y);
        sel.style.left = x + "px";
        sel.style.top = y + "px";
        sel.style.width = w + "px";
        sel.style.height = h + "px";
        // 元画像基準のピクセル数を表示
        sizeLbl.textContent =
          Math.round(w / scale) + " × " + Math.round(h / scale) + " px";
        sizeLbl.style.left = x + "px";
        sizeLbl.style.top = Math.max(0, y - 22) + "px";
        return { x: x, y: y, w: w, h: h };
      }

      function onMove(e) {
        if (!start) return;
        updateSel(e.clientX, e.clientY);
      }
      function onUp(e) {
        if (!start) return;
        lastRect = updateSel(e.clientX, e.clientY);
        start = null;
        if (lastRect.w < 8 || lastRect.h < 8) {
          // 誤クリック相当 → 選択リセット
          sel.style.display = "none";
          sizeLbl.style.display = "none";
          lastRect = null;
          showConfirmUI(false);
          return;
        }
        showConfirmUI(true); // 確定待ち
      }

      dispCanvas.addEventListener("pointerdown", function (e) {
        // 新規ドラッグ開始 → 確認UIを隠して選択し直し
        start = { x: e.clientX, y: e.clientY };
        lastRect = null;
        sel.style.display = "block";
        sizeLbl.style.display = "block";
        showConfirmUI(false);
        updateSel(e.clientX, e.clientY);
      });
      window.addEventListener("pointermove", onMove);
      window.addEventListener("pointerup", onUp);

      redoBtn.addEventListener("click", function () {
        sel.style.display = "none";
        sizeLbl.style.display = "none";
        lastRect = null;
        showConfirmUI(false);
      });

      function confirmCrop() {
        if (!lastRect) return;
        var rect = dispCanvas.getBoundingClientRect();
        // 表示座標 → 元画像座標へ逆変換
        var sx = (lastRect.x - rect.left) / scale;
        var sy = (lastRect.y - rect.top) / scale;
        var sw = lastRect.w / scale;
        var sh = lastRect.h / scale;
        cleanup();
        var out = document.createElement("canvas");
        out.width = Math.round(sw);
        out.height = Math.round(sh);
        out
          .getContext("2d")
          .drawImage(sourceCanvas, sx, sy, sw, sh, 0, 0, out.width, out.height);
        resolve(out.toDataURL("image/png"));
      }
      okBtn.addEventListener("click", confirmCrop);
    });
  }
})();
