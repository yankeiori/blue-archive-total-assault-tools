"""PyInstaller ビルドスクリプト。

ローカル配布用の単一実行ファイルを生成する。
実行後、ブラウザで http://localhost:8050 を開いて使用する。

使い方:
    uv run python third_party_licenses.py
    uv run python build.py
"""

import shutil
import sys

import PyInstaller.__main__

args = [
    "main.py",
    "--name=damage-cutoff-simulator",
    "--onefile",
    "--console",
    # Dash/Flask の assets と docs を同梱
    "--add-data=assets:assets",
    "--add-data=docs:docs",
    # ライセンス表記 (THIRD_PARTY_LICENSES.txt は third_party_licenses.py で事前生成)
    "--add-data=LICENSE:.",
    "--add-data=THIRD_PARTY_LICENSES.txt:.",
    # 隠しインポート (PyInstaller が自動検出できないもの)
    "--hidden-import=app",
    "--hidden-import=app.frontend",
    "--hidden-import=app.frontend.callbacks",
    "--hidden-import=app.frontend.layout",
    "--hidden-import=app.backend",
    "--hidden-import=app.backend.simulation",
    "--hidden-import=dash",
    "--hidden-import=plotly",
    "--hidden-import=numpy",
    "--hidden-import=pandas",
    # 不要モジュール除外でサイズ削減
    "--exclude-module=tkinter",
    "--exclude-module=matplotlib",
    "--exclude-module=scipy",
    "--exclude-module=IPython",
    "--exclude-module=jupyter",
    "--exclude-module=notebook",
    "--exclude-module=pytest",
    "--exclude-module=pyinstaller",
]

# Windows の場合はパス区切りを ; に変換
if sys.platform == "win32":
    args = [a.replace(":", ";", 1) if a.startswith("--add-data=") else a for a in args]

PyInstaller.__main__.run(args)

# 配布物として exe と並べてライセンス表記を置く
for name in ("LICENSE", "THIRD_PARTY_LICENSES.txt"):
    shutil.copy(name, "dist")
