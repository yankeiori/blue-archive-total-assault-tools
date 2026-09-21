/**
 * assets/cos_accumulate.js を Node で実行し、Python の参照実装
 * (app/backend/accumulate.py) と突き合わせるためのハーネス。
 *
 * 標準入力に {cards, pools, globalCrit, globalEvade, globalStability,
 * damageMode, xs} を JSON で受け取り、{mean, std, cdf, windowStats} を返す。
 * tests/test_accumulate_js.py から呼ばれる。
 */
"use strict";
const path = require("path");
global.window = {};
require(path.join(__dirname, "..", "..", "assets", "cos.js"));
require(path.join(__dirname, "..", "..", "assets", "cos_accumulate.js"));
const ns = global.window.dash_clientside;

let raw = "";
process.stdin.on("data", (d) => (raw += d));
process.stdin.on("end", () => {
  const req = JSON.parse(raw);
  const params = {};
  const indices = [];
  req.cards.forEach((c, i) => {
    params[i] = c;
    indices.push(i);
  });
  const opts = {
    indices: indices,
    params: params,
    globalCrit: req.globalCrit,
    globalEvade: req.globalEvade,
    globalStability: req.globalStability,
    damageMode: req.damageMode || "post_decay",
    pools: req.pools,
  };
  const dist = ns.accum.buildDist(opts);
  if (!dist || dist.error) {
    process.stdout.write(JSON.stringify({ error: (dist && dist.error) || "no dist" }));
    return;
  }
  process.stdout.write(
    JSON.stringify({
      mean: dist.mean,
      std: dist.std,
      supportLo: dist.supportLo,
      supportHi: dist.supportHi,
      cdf: req.xs.map((x) => dist.cdf(x)),
      windowStats: dist.windowStats,
    })
  );
});
