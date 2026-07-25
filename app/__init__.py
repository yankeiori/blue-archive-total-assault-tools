import os
import sys
from pathlib import Path

from dash import Dash

# OCR (スクショ→カード自動生成) 機能の有効フラグ。
# ローカル実行では既定で有効。外部公開 (Render 等) では環境変数
# ENABLE_OCR=false を設定して無効化する (render.yaml 参照)。
OCR_ENABLED = os.environ.get("ENABLE_OCR", "true").strip().lower() not in (
    "false", "0", "no", "off",
)

# PyInstaller バンドル時は _MEIPASS、通常時はプロジェクトルート
if getattr(sys, "frozen", False):
    _BASE_DIR = Path(sys._MEIPASS)
else:
    _BASE_DIR = Path(__file__).resolve().parent.parent

app = Dash(
    __name__,
    assets_folder=str(_BASE_DIR / "assets"),
    suppress_callback_exceptions=True,
    title="ブルアカ総力戦・大決戦ツール集",
    update_title=None,
    meta_tags=[
        {"name": "viewport",
         "content": "width=device-width, initial-scale=1"},
    ],
)
