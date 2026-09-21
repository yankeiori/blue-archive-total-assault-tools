/**
 * 入力の自動保存: スナップショット収集 (クライアントサイド)
 *
 * 画面上の入力が変わるたびに、Dash から渡された値をそのまま 1 つの
 * オブジェクトに詰めて dcc.Store(id="autosave-store", storage_type="local")
 * へ返す。localStorage への書き出しは Dash が行うので、ここは値の素通しだけ。
 * 意味付け (カードの組み立てなど) は復元側 app/frontend/persist.py が持つ。
 *
 * サーバーを一切叩かないので、1 文字打つたびに保存しても通信は発生しない。
 */
(function () {
  "use strict";

  var ns = (window.dash_clientside = window.dash_clientside || {});
  ns.persist = {};

  // 形式を変えたら app/frontend/persist.py の SNAPSHOT_VERSION も上げること
  // (版が違う保存は復元側で読み捨てる)。
  var VERSION = 1;

  // main.py に並べた Input / State と 1:1 で対応する引数名。
  // 片方だけ並べ替えるとスナップショットの中身がずれるので必ず両方を直すこと。
  var FIELDS = [
    // --- ダメージシミュレータ ---
    "param_values",
    "memo_values",
    "order",
    "card_indices",
    "next_index",
    "target_damage",
    "global_crit",
    "global_evade",
    "global_stability",
    "calc_method",
    "damage_mode",
    "hp_mode",
    "hp_H",
    "hp_H1",
    "hp_R0",
    "hp_R1",
    "text_input",
    // --- 足切りライン最適化 ---
    "restart_D",
    "restart_cp",
    "restart_seg_time",
    "restart_seg_success",
    "restart_save",
    // --- スキル順探索 ---
    "so_hand_size",
    "so_card_count",
    "so_limit",
    "so_tl_text",
    "so_names",
    "so_copiers",
    "so_step_skill",
    "so_step_target",
    "so_step_slot",
    "so_step_draw",
    "so_step_memo",
    "so_step_order",
    "so_next_step",
    "so_con_type",
    "so_con_steps",
    "so_next_con",
    // --- 値と添字を対応づけるための id 群 (State) ---
    "param_ids",
    "memo_ids",
    "so_step_ids",
    "so_con_ids",
    // --- 復元が済むまで保存しないためのフラグ (State) ---
    "armed",
  ];

  ns.persist.collect = function () {
    // 復元前 (= 画面がまだ空の初期状態) に保存すると前回の入力を消してしまう。
    if (!arguments[FIELDS.length - 1]) {
      throw window.dash_clientside.PreventUpdate;
    }
    var snap = { v: VERSION };
    for (var i = 0; i < FIELDS.length - 1; i++) {
      var value = arguments[i];
      snap[FIELDS[i]] = value === undefined ? null : value;
    }
    return snap;
  };
})();
