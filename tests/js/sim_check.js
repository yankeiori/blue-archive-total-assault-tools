/**
 * クライアントサイドコールバック ns.sim.runSimulation を Node で丸ごと実行する
 * ハーネス。Dash が渡す (値, id) の並びを組み立てて呼び、返り値を JSON で返す。
 *
 * 蓄積スキルの入力欄 → buildPools → cos_accumulate.js までの配線を、
 * ブラウザ無しで検証するために使う (tests/test_accumulate_ui.py)。
 */
"use strict";
const path = require("path");
global.window = {};
global.window.dash_clientside = {
  PreventUpdate: new Error("PreventUpdate"),
  no_update: "__no_update__",
  callback_context: { triggered: [] },
};
require(path.join(__dirname, "..", "..", "assets", "cos.js"));
require(path.join(__dirname, "..", "..", "assets", "cos_accumulate.js"));
require(path.join(__dirname, "..", "..", "assets", "simulation.js"));
const ns = global.window.dash_clientside;

const CARD_FIELDS = ["crit_min", "crit_max", "normal_min", "normal_max",
                     "hits", "crit_rate", "evade_rate", "enemies", "hp_dep"];

let raw = "";
process.stdin.on("data", (d) => (raw += d));
process.stdin.on("end", () => {
  const req = JSON.parse(raw);

  const values = [], ids = [];
  req.cards.forEach((c, i) => {
    CARD_FIELDS.forEach((f) => {
      values.push(c[f] === undefined ? null : c[f]);
      ids.push({ type: "param", param: f, index: i });
    });
  });
  const accumValues = [], accumIds = [];
  (req.accum || []).forEach((a, k) => {
    Object.keys(a).forEach((f) => {
      accumValues.push(a[f]);
      accumIds.push({ type: "accum", field: f, index: k });
    });
  });
  const indices = req.cards.map((_c, i) => i);

  let out;
  try {
    out = ns.sim.runSimulation(
      1, values, ids, indices, indices,
      req.globalCrit, req.globalEvade, req.globalStability,
      req.target, req.damageMode || "post_decay", req.method || "cos",
      req.hpMode || "off", 1e6, 1e6, 1, 2,
      accumValues, accumIds
    );
  } catch (e) {
    process.stdout.write(JSON.stringify({ threw: String(e.message || e) }));
    return;
  }
  const dist = out[0] === ns.no_update ? null : out[0];
  process.stdout.write(JSON.stringify({
    passText: out[1],
    summary: typeof out[5] === "string" && out[5] ? out[5].split("\n") : [],
    meanFromFig: dist && dist.layout ? true : false,
    cdfTable: out[3] === "__no_update__" ? null
      : { n: out[3].grid.length, lo: out[3].grid[0], hi: out[3].grid[out[3].grid.length - 1] },
  }));
});
