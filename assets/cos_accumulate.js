/**
 * 蓄積 (チャージ) 型スキルのダメージ分布 — app/backend/accumulate.py の移植。
 *
 * 正本は Python 側 (app/backend/accumulate.py)。理論は docs/accumulate.md。
 *
 *     A_k = g_k(mult_k * min(C_k, α_k Σ_{i∈W_k} X_i)),   T = Σ_i X_i + Σ_k A_k
 *
 * 蓄積窓 W_k は互いに素と仮定する (docs/accumulate.md §3.1)。このとき窓の寄与
 * Z_k = S_k + A_k と窓外合計 V は相互独立なので、各成分を 1 次元で作って畳み込む。
 *
 * 全体を「原点 0・刻み h の等間隔セル質量」で統一し、
 *   1. Hit ごとの一様混合を厳密な重なり積分でセル質量にする (質量・平均とも保存)
 *   2. 同じカードの Hit は FFT を count 乗して一括
 *   3. 窓の基礎分布は実空間に戻して ψ を通して押し出し、再び FFT
 *   4. 全成分を周波数領域で掛けて 1 回だけ逆変換
 * とする。セル質量は常に非負・総和 1 なので Gibbs も負密度も出ない。
 *
 * cos.js の後に読み込まれる (ファイル名昇順) が、ns.cos の参照は関数内だけで行う。
 */
(function () {
  "use strict";

  var ns = (window.dash_clientside = window.dash_clientside || {});
  ns.accum = {};

  var N_CELLS = 1 << 17;      // 合成グリッドのセル数 (2 の冪)
  var N_CAP_NODES = 64;       // 上限分布の等質量ノード数
  var TRIM_EPS = 1e-15;

  // =========================================================================
  // FFT (反復 radix-2, in-place)。ツイドル表は長さごとにキャッシュする。
  // =========================================================================
  var twCache = {};

  function twiddle(n) {
    if (twCache[n]) return twCache[n];
    var half = n >> 1;
    var cos = new Float64Array(half), sin = new Float64Array(half);
    for (var k = 0; k < half; k++) {
      var a = (-2 * Math.PI * k) / n;
      cos[k] = Math.cos(a);
      sin[k] = Math.sin(a);
    }
    twCache[n] = { cos: cos, sin: sin };
    return twCache[n];
  }

  function fftInPlace(re, im, inverse) {
    var n = re.length, i, j, bit;
    for (i = 1, j = 0; i < n; i++) {
      for (bit = n >> 1; j & bit; bit >>= 1) j ^= bit;
      j ^= bit;
      if (i < j) {
        var tr = re[i]; re[i] = re[j]; re[j] = tr;
        var ti = im[i]; im[i] = im[j]; im[j] = ti;
      }
    }
    var tw = twiddle(n);
    for (var len = 2; len <= n; len <<= 1) {
      var half = len >> 1, stride = n / len;
      for (var base = 0; base < n; base += len) {
        for (var k = 0; k < half; k++) {
          var t = k * stride;
          var wr = tw.cos[t], wi = inverse ? -tw.sin[t] : tw.sin[t];
          var a = base + k, b = a + half;
          var vr = re[b] * wr - im[b] * wi;
          var vi = re[b] * wi + im[b] * wr;
          re[b] = re[a] - vr; im[b] = im[a] - vi;
          re[a] += vr; im[a] += vi;
        }
      }
    }
    if (inverse) {
      for (i = 0; i < n; i++) { re[i] /= n; im[i] /= n; }
    }
  }

  /** 実数セル質量 → 周波数表現 {re, im}。 */
  function forward(cells) {
    var n = cells.length;
    var re = new Float64Array(n), im = new Float64Array(n);
    re.set(cells);
    fftInPlace(re, im, false);
    return { re: re, im: im };
  }

  /** 周波数表現 → 実数セル質量 (負は 0 に丸める)。null なら原点の点質量。 */
  function inverse(cf, n) {
    var out = new Float64Array(n), i;
    if (!cf) { out[0] = 1.0; return out; }
    var re = new Float64Array(cf.re), im = new Float64Array(cf.im);
    fftInPlace(re, im, true);
    for (i = 0; i < n; i++) out[i] = re[i] > 0 ? re[i] : 0;
    return out;
  }

  function cMul(a, b) {
    var n = a.re.length;
    for (var i = 0; i < n; i++) {
      var r = a.re[i] * b.re[i] - a.im[i] * b.im[i];
      a.im[i] = a.re[i] * b.im[i] + a.im[i] * b.re[i];
      a.re[i] = r;
    }
    return a;
  }

  /** 各成分を k 乗する (極形式。|φ| <= 1 なので安定)。 */
  function cPow(a, k) {
    if (k === 1) return a;
    var n = a.re.length;
    for (var i = 0; i < n; i++) {
      var r = Math.hypot(a.re[i], a.im[i]);
      if (r === 0) { a.re[i] = 0; a.im[i] = 0; continue; }
      var th = Math.atan2(a.im[i], a.re[i]) * k;
      var rk = Math.pow(r, k);
      a.re[i] = rk * Math.cos(th);
      a.im[i] = rk * Math.sin(th);
    }
    return a;
  }

  // =========================================================================
  // セル質量
  // =========================================================================
  /** 質量を隣接 2 セルへ線形分配する (総質量と平均を保存)。 */
  function deposit(out, pos, mass, step) {
    var n = out.length;
    var x = pos / step;
    if (!(x > 0)) x = 0;
    if (x > n - 1) x = n - 1;
    var i = Math.floor(x);
    if (i > n - 2) i = n - 2;
    var t = x - i;
    out[i] += mass * (1 - t);
    out[i + 1] += mass * t;
  }

  /** 1 Hit の一様混合を厳密なセル質量へ (各セルの質量と条件付き平均を保つ)。 */
  function mixtureCells(mix, step, n) {
    var out = new Float64Array(n);
    for (var c = 0; c < mix.length; c++) {
      var u = mix[c];
      if (!(u.hi > u.lo)) { deposit(out, u.lo, u.weight, step); continue; }
      var i0 = Math.max(0, Math.floor(u.lo / step + 0.5));
      var i1 = Math.min(n - 1, Math.floor(u.hi / step + 0.5));
      var width = u.hi - u.lo;
      for (var i = i0; i <= i1; i++) {
        var eLo = Math.max(u.lo, (i - 0.5) * step);
        var eHi = Math.min(u.hi, (i + 0.5) * step);
        var seg = eHi - eLo;
        if (seg <= 0) continue;
        deposit(out, 0.5 * (eLo + eHi), (u.weight * seg) / width, step);
      }
    }
    return out;
  }

  /** セル質量を等質量バケットへまとめ、{v: 代表値, w: 質量} を返す。 */
  function quantize(cells, step, nNodes) {
    var idx = [], i;
    for (i = 0; i < cells.length; i++) if (cells[i] > TRIM_EPS) idx.push(i);
    if (idx.length <= nNodes) {
      return { v: idx.map(function (k) { return k * step; }),
               w: idx.map(function (k) { return cells[k]; }) };
    }
    var total = 0;
    for (i = 0; i < idx.length; i++) total += cells[idx[i]];
    var v = [], w = [], acc = 0, sum = 0, mom = 0, bucket = 1;
    for (i = 0; i < idx.length; i++) {
      var m = cells[idx[i]];
      acc += m; sum += m; mom += m * idx[i] * step;
      if (acc >= (total * bucket) / nNodes && bucket < nNodes) {
        if (sum > 0) { v.push(mom / sum); w.push(sum); }
        sum = 0; mom = 0; bucket++;
      }
    }
    if (sum > 0) { v.push(mom / sum); w.push(sum); }
    return { v: v, w: w };
  }

  // =========================================================================
  // 区分線形写像 ψ (docs/accumulate.md §1.1)
  // =========================================================================
  function burstAmount(s, cap, rate, burstDecay, mult) {
    var p = mult * Math.min(cap, rate * s);
    return burstDecay ? ns.cos.decay(p) : p;   // ns.cos.decay はスカラー
  }

  function psi(s, cap, rate, burstDecay, mult) {
    return s + burstAmount(s, cap, rate, burstDecay, mult);
  }

  // =========================================================================
  // 窓・成分の構成
  // =========================================================================
  function supportHi(groups, idxs) {
    var hi = 0;
    for (var t = 0; t < idxs.length; t++) {
      var g = groups[idxs[t]], m = 0;
      for (var c = 0; c < g.mix.length; c++) m = Math.max(m, g.mix[c].hi);
      hi += m * g.count;
    }
    return hi;
  }

  function allIdx(groups) {
    return groups.map(function (_g, i) { return i; });
  }

  /** カード index の配列 → グループ位置の配列。 */
  function toPositions(groups, cardIds) {
    var pos = [];
    for (var i = 0; i < groups.length; i++) {
      if ((cardIds || []).indexOf(groups[i].idx) >= 0) pos.push(i);
    }
    return pos;
  }

  /**
   * 蓄積スキル設定をグループ位置ベースへ正規化し、重なりを検証する。
   * 返り値 {pools, roles, error}。error があれば計算しない。
   */
  function normalize(groups, pools) {
    var live = [], roles = [], i, k;
    for (i = 0; i < groups.length; i++) roles.push({ kind: "free" });
    for (k = 0; k < (pools || []).length; k++) {
      var p = pools[k];
      var cards = toPositions(groups, p.cards);
      var rate = parseFloat(p.rate);
      var mult = parseFloat(p.burstMult);
      if (!cards.length || !(rate > 0) || !(mult > 0) || p.emit === false) continue;
      for (i = 0; i < cards.length; i++) {
        if (roles[cards[i]].kind !== "free") {
          return { error: "蓄積スキル「" + (p.name || k + 1) + "」の対象カードが" +
                          "他の蓄積スキルと重なっています。重なる蓄積は未対応です。" };
        }
      }
      live.push({ name: p.name, cards: cards, rate: rate, mult: mult,
                  capKind: p.capKind, capValue: parseFloat(p.capValue) || 0,
                  capCards: toPositions(groups, p.capCards),
                  capCoef: parseFloat(p.capCoef) || 0,
                  burstDecay: !!p.burstDecay });
      for (i = 0; i < cards.length; i++) roles[cards[i]] = { kind: "win", k: live.length - 1 };
    }
    for (k = 0; k < live.length; k++) {
      var src = live[k].capKind === "cards" ? live[k].capCards : [];
      if (live[k].capKind === "cards" && !src.length) {
        return { error: "蓄積スキル「" + (live[k].name || k + 1) +
                        "」の上限カードが指定されていません。" };
      }
      var anyIn = false, anyOut = false;
      for (i = 0; i < src.length; i++) {
        var r = roles[src[i]];
        if (r.kind === "cap") {
          return { error: "カードが複数の蓄積スキルの上限ロールになっています (未対応)。" };
        }
        if (r.kind === "win" && r.k === k) anyIn = true;
        else if (r.kind === "win") {
          return { error: "上限カードが別の蓄積スキルの対象になっています (未対応)。" };
        } else anyOut = true;
      }
      if (anyIn && anyOut) {
        return { error: "上限カードは蓄積対象の内・外どちらかに揃えてください。" };
      }
      live[k].inside = anyIn;
      live[k].src = src;
      for (i = 0; i < src.length; i++) roles[src[i]] = { kind: "cap", k: k };
    }
    return { pools: live, roles: roles };
  }

  function groupCF(groups, idxs, step, n) {
    if (!idxs.length) return null;
    var cf = null;
    for (var t = 0; t < idxs.length; t++) {
      var g = groups[idxs[t]];
      var f = cPow(forward(mixtureCells(g.mix, step, n)), g.count);
      cf = cf ? cMul(cf, f) : f;
    }
    return cf;
  }

  function capNodes(pool, srcCells, step, nNodes) {
    if (pool.capKind === "cards") {
      var q = quantize(srcCells, step, nNodes);
      return { v: q.v.map(function (x) { return pool.capCoef * x; }), x: q.v, w: q.w };
    }
    return { v: [pool.capValue], x: [0], w: [1] };
  }

  /** 窓の寄与 Z_k のセル質量と診断量。 */
  function windowCells(pool, baseCells, srcCells, step, n, nNodes) {
    var nodes = capNodes(pool, srcCells, step, nNodes);
    var out = new Float64Array(n);
    var sat = 0, poolMean = 0, over = 0, burst = 0, dmg = 0, capMean = 0;
    var keep = [], i;
    for (i = 0; i < baseCells.length; i++) if (baseCells[i] > TRIM_EPS) keep.push(i);
    for (i = 0; i < keep.length; i++) dmg += baseCells[keep[i]] * keep[i] * step;
    for (var j = 0; j < nodes.v.length; j++) {
      var c = nodes.v[j], x = nodes.x[j], w = nodes.w[j];
      capMean += w * c;
      if (pool.inside) dmg += 0;  // x は下で加える
      for (i = 0; i < keep.length; i++) {
        var s = keep[i] * step, m = baseCells[keep[i]] * w;
        var arg = pool.inside ? s + x : s;
        var z = psi(arg, c, pool.rate, pool.burstDecay, pool.mult);
        if (!pool.inside) z += x;
        deposit(out, z, m, step);
        var raw = pool.rate * arg;
        if (raw > c) sat += m;
        poolMean += m * Math.min(c, raw);
        over += m * Math.max(raw - c, 0);
        burst += m * burstAmount(arg, c, pool.rate, pool.burstDecay, pool.mult);
      }
      if (pool.inside) dmg += w * x;
    }
    return {
      cells: out,
      stats: { name: pool.name, rate: pool.rate, satProb: sat, poolMean: poolMean,
               overflowMean: over, burstMean: burst, capMean: capMean, damageMean: dmg },
    };
  }

  // =========================================================================
  // 公開 API
  // =========================================================================
  /**
   * 蓄積つき合計ダメージの分布を構築する。
   * opts: {indices, params, globalCrit, globalEvade, globalStability, damageMode, pools}
   * 返り値は ns.cos.buildDist と同じ形 {kind, mean, std, supportLo, supportHi, cdf, pdf}
   * に windowStats / error を足したもの。
   */
  ns.accum.buildDist = function (opts) {
    var groups = ns.cos.buildGroups(opts.indices, opts.params, opts.globalCrit,
                                    opts.globalEvade, opts.damageMode,
                                    opts.globalStability);
    if (!groups.length) return null;
    var norm = normalize(groups, opts.pools);
    if (norm.error) return { error: norm.error };
    var live = norm.pools, roles = norm.roles, i, k;

    var hiT = supportHi(groups, allIdx(groups));
    for (k = 0; k < live.length; k++) {
      var capHi = live[k].capKind === "cards"
        ? live[k].capCoef * supportHi(groups, live[k].src)
        : live[k].capValue;
      hiT += burstAmount(supportHi(groups, live[k].cards), capHi, live[k].rate,
                         live[k].burstDecay, live[k].mult);
    }
    var n = N_CELLS;
    var step = Math.max(hiT / (n - 2), 1e-9);

    var freeIdx = [];
    for (i = 0; i < groups.length; i++) if (roles[i].kind === "free") freeIdx.push(i);
    var cfTotal = groupCF(groups, freeIdx, step, n);

    var stats = [];
    for (k = 0; k < live.length; k++) {
      var baseIdx = live[k].cards.filter(function (p) {
        return roles[p].kind === "win";
      });
      var baseCells = inverse(groupCF(groups, baseIdx, step, n), n);
      var srcCells = live[k].src && live[k].src.length
        ? inverse(groupCF(groups, live[k].src, step, n), n) : null;
      var wc = windowCells(live[k], baseCells, srcCells, step, n, N_CAP_NODES);
      var cfZ = forward(wc.cells);
      cfTotal = cfTotal ? cMul(cfTotal, cfZ) : cfZ;
      stats.push(wc.stats);
    }

    var mass = inverse(cfTotal, n);
    var total = 0;
    for (i = 0; i < n; i++) total += mass[i];
    var i0 = 0, i1 = n - 1;
    while (i0 < n && mass[i0] <= TRIM_EPS * total) i0++;
    while (i1 > i0 && mass[i1] <= TRIM_EPS * total) i1--;
    var size = i1 - i0 + 1;
    var m = new Float64Array(size), cum = new Float64Array(size);
    var run = 0, mean = 0;
    for (i = 0; i < size; i++) {
      m[i] = mass[i0 + i] / total;
      run += m[i];
      cum[i] = run;
      mean += m[i] * (i0 + i) * step;
    }
    var varr = 0;
    for (i = 0; i < size; i++) {
      var d = (i0 + i) * step - mean;
      varr += m[i] * d * d;
    }
    var lo = i0 * step;

    function cdf(x) {
      if (x < lo - 0.5 * step) return 0;
      var t = (x - lo) / step + 0.5;   // セル右端基準
      if (t >= size) return 1;
      if (t <= 0) return 0;
      var a = Math.floor(t), fr = t - a;
      var c0 = a > 0 ? cum[a - 1] : 0;
      var c1 = a < size ? cum[a] : 1;
      return c0 + fr * (c1 - c0);
    }

    return {
      kind: "accum",
      mean: mean,
      std: Math.sqrt(Math.max(varr, 0)),
      supportLo: lo,
      supportHi: lo + step * (size - 1),
      step: step,
      windowStats: stats,
      cdf: cdf,
      pdf: function (x) {
        var t = (x - lo) / step;
        if (t < 0 || t > size - 1) return 0;
        var a = Math.floor(t), fr = t - a;
        var v0 = m[a] / step, v1 = a + 1 < size ? m[a + 1] / step : 0;
        return v0 + fr * (v1 - v0);
      },
    };
  };

  /** 分布を細グリッド上の {x, pdf, cdf} に評価する (図用)。 */
  ns.accum.distribution = function (opts, nGrid) {
    var dist = ns.accum.buildDist(opts);
    if (!dist || dist.error) return dist;
    nGrid = nGrid || 600;
    var lo = dist.supportLo, hi = dist.supportHi;
    if (dist.std > 0) {
      lo = Math.max(lo, dist.mean - 8 * dist.std);
      hi = Math.min(hi, dist.mean + 8 * dist.std);
    }
    var x = new Array(nGrid), pdf = new Array(nGrid), cdf = new Array(nGrid);
    var step = (hi - lo) / (nGrid - 1);
    for (var i = 0; i < nGrid; i++) {
      var xi = lo + i * step;
      x[i] = xi; pdf[i] = dist.pdf(xi); cdf[i] = dist.cdf(xi);
    }
    return { x: x, pdf: pdf, cdf: cdf, mean: dist.mean, std: dist.std,
             supportLo: dist.supportLo, supportHi: dist.supportHi,
             windowStats: dist.windowStats };
  };

  /** MC サンプル (計算方式「モンテカルロ」用)。 */
  ns.accum.sample = function (opts, nSamples) {
    var groups = ns.cos.buildGroups(opts.indices, opts.params, opts.globalCrit,
                                    opts.globalEvade, opts.damageMode,
                                    opts.globalStability);
    if (!groups.length) return null;
    var norm = normalize(groups, opts.pools);
    if (norm.error) return { error: norm.error };
    var live = norm.pools, i, k, t;

    // グループごとの合計ダメージサンプル
    var per = [];
    for (i = 0; i < groups.length; i++) {
      var acc = new Float64Array(nSamples);
      for (var c = 0; c < groups[i].count; c++) {
        for (t = 0; t < nSamples; t++) acc[t] += sampleMixture(groups[i].mix);
      }
      per.push(acc);
    }
    var total = new Float64Array(nSamples);
    for (i = 0; i < groups.length; i++) {
      for (t = 0; t < nSamples; t++) total[t] += per[i][t];
    }
    for (k = 0; k < live.length; k++) {
      var p = live[k];
      for (t = 0; t < nSamples; t++) {
        var s = 0, cap;
        for (i = 0; i < p.cards.length; i++) s += per[p.cards[i]][t];
        if (p.capKind === "cards") {
          cap = 0;
          for (i = 0; i < p.src.length; i++) cap += per[p.src[i]][t];
          cap *= p.capCoef;
        } else cap = p.capValue;
        total[t] += burstAmount(s, cap, p.rate, p.burstDecay, p.mult);
      }
    }
    return total;
  };

  function sampleMixture(mix) {
    var r = Math.random(), acc = 0;
    for (var i = 0; i < mix.length; i++) {
      acc += mix[i].weight;
      if (r <= acc || i === mix.length - 1) {
        return mix[i].hi > mix[i].lo
          ? mix[i].lo + Math.random() * (mix[i].hi - mix[i].lo)
          : mix[i].lo;
      }
    }
    return 0;
  }

  /** 蓄積スキルが 1 つでも有効かどうか。 */
  ns.accum.hasPools = function (pools) {
    for (var k = 0; k < (pools || []).length; k++) {
      var p = pools[k];
      if (p && p.cards && p.cards.length && parseFloat(p.rate) > 0) return true;
    }
    return false;
  };
})();
