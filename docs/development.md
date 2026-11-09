# 開発ガイド

[README に戻る](../README.md) · [使い方](usage.md) · [構成とセキュリティ](architecture.md) · [性能](performance.md)

## ソースから起動する

uv を公式の配布元からインストールし、リポジトリのルートで次を実行します。通常の起動スクリプトは uv のインストールも確認してくれるので、利用者はそちらを使ってください（[README](../README.md#クイックスタート)）。

```powershell
uv sync --frozen --extra speakers --python 3.12
# Windows: 準備済みの .venv で直接起動する（依存の同期もブラウザを開くこともしない）
.\.venv\Scripts\python.exe -B -m yata
# どの OS でも: 依存を確認・同期して起動し、ブラウザを開く
uv run --frozen --inexact --extra speakers python -B -m yata --open
```

- `uv sync` は開発用のグループ（pytest と httpx）も入れます。`--inexact` を付けると、入っている追加パッケージを残します。
- アプリはこのソースフォルダから動かします（`tool.uv.package = false`）。`web/`、`yata/`、ルートの素材は一緒に置いてください。
- 起動すると `127.0.0.1:8765` で待ち受けます。すでに Yata が動いている場合は、2 つ目を立ち上げずに既存のページを開きます（同じセッションファイルを 2 つのプロセスが書き換えないようにするためです）。8765 番ポートを別のアプリが使っている場合は、メッセージを出して終了します。
- uv はリポジトリに `.venv` を作り、管理する Python とダウンロードのキャッシュをユーザーフォルダに置きます。検証用に別の環境を作るときは、データ・モデル・Hugging Face・uv のキャッシュの場所を分け、利用者の既存のモデルやセッションを使ったり消したりしないでください。

### 追加の機能（extras）

| extra | 内容 |
|---|---|
| `speakers` | 話者ラベル用の sherpa-onnx。起動スクリプトは最初から入れます。CAM++ モデルは準備画面で取得します |
| `cuda` | NVIDIA の CUDA 12 用 cuBLAS（Linux では cuDNN 9 も）。NVIDIA の GPU があるのにライブラリが足りない場合、起動スクリプトが入れるか確認します |
| `qwen` | Qwen3-ASR 用の torch と transformers。torch は NVIDIA の CUDA 13 版に固定しているため、NVIDIA GPU のある Windows・Linux だけで使えます（macOS は不可） |

`uv sync` は、指定しなかった extra のパッケージを削除します。使う extra をすべて指定するか、`--inexact` を付けてください。たとえば、話者ラベルを残したまま Qwen を追加するには次のようにします。

```powershell
uv sync --frozen --inexact --extra qwen
```

Qwen3-ASR のモデル（`Qwen/Qwen3-ASR-1.7B-hf`）はファイルが揃ったフォルダに取得し、`YATA_ASR_ENGINE=qwen3-asr` と `YATA_ASR_MODEL_PATH` を設定します。

```powershell
hf download Qwen/Qwen3-ASR-1.7B-hf --local-dir "C:\models\Qwen3-ASR-1.7B-hf"
```

`hf` は、依存パッケージに含まれる Hugging Face の CLI です。この取得は配布元に接続し、通常のキャッシュを使います。準備画面の取得（ハッシュを検証する）とは別物です。

## 起動スクリプトの動作

`start-windows.cmd` は `scripts/start.ps1` を、`start-mac.command` は `start-mac-linux.sh` を実行します。

1. uv を探し、なければ Astral の公式インストーラーで入れるか確認します。`YATA_ASSUME_YES=1` を設定すると、インストールと CUDA の確認にすべて「はい」と答えます。ダウンロードを許可すると決めた場合だけ使ってください。
2. `uv sync --frozen --inexact --no-dev --extra speakers` を実行します。新しい環境では Python 3.12 を使い、入っている追加パッケージは残します。
3. 必要なら CUDA のライブラリを入れるか確認します。断った場合は `.venv/.yata-no-cuda` に記録します。Windows ではプロセスの `PATH` の先頭に同梱の cuBLAS のフォルダを追加し、Linux では `LD_LIBRARY_PATH` に cuBLAS・cuDNN を設定します。CTranslate2 が読み込むライブラリが揃わない限り、GPU は使いません。
4. `python -B -m yata --open` を実行します。サービスが応答するとブラウザが開きます。Ctrl+C で止めます。

macOS は、固定しているバイナリパッケージの都合で Apple シリコン・macOS 14 以降に限られます。llama.cpp の実行環境を準備画面で取得するのは Windows だけで、macOS・Linux では `llama-server` を `PATH` に置くか `YATA_LLAMA_SERVER` を設定します。

## 設定（環境変数）

モデルの選択は、基本的に準備画面で行います。環境変数は保存した選択より優先され、準備画面では固定として表示されます。取り込み中の共有は始めたときの設定のまま動き、変更は次の共有から反映されます。API キーや Webhook を、Git で管理するファイルに書かないでください。

| 環境変数 | 内容 |
|---|---|
| `YATA_DATA_DIR` | データフォルダ（設定とセッション）。未設定なら OS ごとの既定の場所 |
| `YATA_MODEL_ROOT` | モデルの保存先を固定します |
| `YATA_ASR_PROFILE` | `auto`、`full`、`light`。`light` は CPU 向けの構成を強制します |
| `YATA_ASR_ENGINE` | `faster-whisper`（既定）または `qwen3-asr` |
| `YATA_ASR_DEVICE`、`YATA_ASR_COMPUTE_TYPE` | 装置（`auto`／`cuda`／`cpu`）と精度（CTranslate2 の compute type）。CPU の既定は int8 |
| `YATA_ASR_CPU_THREADS` | 音声認識の CPU スレッド数。0 はライブラリの既定（4） |
| `YATA_ASR_MODEL_PATH` | ファイルが揃った音声認識モデルのフォルダ。未設定なら、モデル保存先の既定のフォルダ、Hugging Face のキャッシュの順に探します |
| `YATA_ASR_PARTIALS` | 話している途中の字幕（`on`／`off`）。`light` 構成では既定でオフ |
| `YATA_ASR_PARTIAL_MODEL_PATH` | Qwen と組み合わせる、途中の字幕用の faster-whisper モデル |
| `YATA_SPEAKER_MODEL_PATH` | 話者ラベルの CAM++ モデルの場所 |
| `YATA_LLM_BACKEND` | `builtin`、`ollama`、`openai`、`foundry` |
| `YATA_OPENAI_BASE_URL`、`YATA_OPENAI_MODEL` | OpenAI 互換サーバーの URL（ループバックのみ）とモデル |
| `YATA_OPENAI_API_KEY` | 環境変数からだけ読みます。保存も表示もしません |
| `YATA_LLAMA_SERVER` | 使う `llama-server` の実行ファイル |
| `YATA_LLAMA_DEVICE` | llama.cpp の装置 ID（`none` で CPU）。未設定なら実行環境のテストで選んだもの |
| `YATA_SLACK_WEBHOOK_URL` | Slack の Webhook（秘密情報）。機能トグルとセッションごとの選択も必要です |

手で置いたモデルフォルダは、ファイルが揃った状態にしてください。faster-whisper は、フォルダに足りないトークナイザーのファイルがあると、配布元から取得しようとすることがあります。

## 機能トグル

`YATA_FEATURE_<名前>` は、`on`、`1`、`true`、`yes` のいずれかを設定したときだけ有効になります。サーバーが制御し、ページには `/api/v1/health` の `features` で伝えます。

| トグル | 既定と動作 |
|---|---|
| `YATA_FEATURE_THEME_SWITCH` | オフ（ソフトテーマ）。有効にすると、ソフトと HUD を切り替えられます。オフの間は、ブラウザに保存したテーマを無視します |
| `YATA_FEATURE_SLACK` | オフ。Webhook を設定しても送信しません。Slack の UI を隠し、Slack 付きのセッションの作成を拒否し、再送は 409 を返します |

```powershell
$env:YATA_FEATURE_THEME_SWITCH = "on"
# 要約を Slack に送ってよい場合だけ
$env:YATA_FEATURE_SLACK = "on"
.\.venv\Scripts\python.exe -B -m yata
```

## テスト

dev グループと `speakers` extra を入れておきます。JavaScript のテストには **Node.js 20 以降** が必要です（Node の標準のテストランナーを使い、ブラウザや npm パッケージは要りません）。

```powershell
uv run --frozen --extra speakers python -B -m pytest -p no:cacheprovider
node --test tests/app.test.mjs tests/audio-diagnostics.test.mjs tests/capture.test.mjs
```

- Python のテストは、偽の音声認識エンジン、音量で判定する VAD の代わり（実際の Silero のストリーミング確認も 1 つあります）、台本どおりに答える LLM、合成した音声、ローカルのファイルサーバーからのダウンロードを使います。OpenAI 互換・llama.cpp・Ollama・Foundry の動作は、偽のループバックサーバーで確かめます。
- JavaScript のテストは、ページのスクリプトを小さな DOM の代わりの上で動かします。
- `tests/conftest.py` は、データと Hugging Face のキャッシュを一時フォルダに向け、`YATA_MODEL_ROOT` を外します。また、外部への接続を拒否して記録するガードを入れています（範囲は [構成とセキュリティ](architecture.md#ダウンロードと依存ライブラリの通信) を参照）。

これらのテストは、実際の Whisper の認識、配布元の可用性、実際の外部の実行環境、ブラウザでの音声取り込み、デスクトップ環境での動作までは確かめません。

### 手動で確認すること

これまでに、Windows 11 の開発機でのソースとポータブル版の起動・モデルの利用、Ubuntu 24.04 のコンテナでの CPU 構成の起動とテストを確認しています。まだ確認していないのは次の項目です。

- Python・uv・C++ ランタイムが入っていない Windows と、配布する ZIP そのもの
- Mac での実行（Finder からの起動、CPU での推論、タブの音声の取り込み）
- Linux のデスクトップ環境、Linux での NVIDIA／CUDA、Linux での llama-server と PLaMo
- 最近の画面変更後の、実際のタブ音声を使った通しの動作
- 準備画面で指定した PLaMo の GGUF ファイルを、実際の llama-server で読み込むこと
- ネイティブライブラリのテレメトリ、完全なオフラインでの動作

確認した結果を残すときは、コミット、OS、Python、extra、実行したコマンド、確認できなかった範囲も一緒に記録してください。性能の測定は [性能と測定方法](performance.md) にまとめています。

## ポータブル版のビルド

Windows x64 向けに、CPython の embeddable package と固定した依存ライブラリをまとめ、`Yata.cmd` で起動するフォルダを作ります。利用者は Python・uv・CUDA Toolkit を入れずに起動できます。音声認識や LLM のモデルは同梱せず、起動後に準備画面で取得します（VAD は依存パッケージに含まれます）。

| 版 | 内容 | 音声認識 |
|---|---|---|
| CPU | faster-whisper、話者ラベル用の sherpa-onnx、共通の実行環境 | CPU（準備画面は whisper-small を推奨） |
| GPU | CPU 版に CUDA 12 の cuBLAS の DLL を追加 | 対応する NVIDIA GPU とドライバーがあれば CUDA、なければ CPU |

- GPU 版は CPU 版への追加パックではなく、単独で展開して使うフォルダです。CUDA の DLL を利用者に用意させず、ビルド時に版とハッシュを確認するためにこの形にしています。
- CPU 版のランチャーは、`YATA_ASR_DEVICE` も `CUDA_VISIBLE_DEVICES` も指定されていなければ、そのプロセスに `CUDA_VISIBLE_DEVICES=-1` を設定します（PC 全体の設定は変えません）。
- 音声認識の CUDA 対応と LLM の GPU 対応は別です。AMD・Intel の GPU は音声認識の CUDA の対象外ですが、Windows の LLM は llama.cpp の Vulkan 版で使えます。
- Windows ARM64・Linux・macOS 向けのポータブル版はありません。Qwen3-ASR もポータブル版の対象外です（CUDA 版 torch を別の追加パックにする案は未実装です）。

### ビルド環境

- Windows x64、Python 3.12 以上、uv、リポジトリのソースと `uv.lock`
- CPython・ホイール・GPU 版の cuBLAS を取得できるネットワーク（モデルの取得は不要）
- Visual Studio または Build Tools に含まれる x64 の Visual C++ 再頒布ファイルと、それを再頒布する権利。スクリプトは `vswhere` で `VC\Redist\MSVC\...\x64\Microsoft.VC14*.CRT` を探します
- GPU 版の CUDA の動作確認には、対応する NVIDIA GPU とドライバー（GPU がないビルド環境では実際の計算の確認を省くので、別の機械で補います）

### ビルドする

リポジトリのルートで実行します。

```powershell
uv run python tools/build_portable.py --edition cpu --zip
uv run python tools/build_portable.py --edition gpu --zip
```

Visual C++ ランタイムが自動で見つからない場合は、CRT のフォルダを指定します。

```powershell
uv run python tools/build_portable.py --edition cpu --zip --vc-runtime 'C:\path\to\Microsoft.VC143.CRT'
```

成果物は `dist/portable/Yata-<version>-cpu/`（または `-gpu/`）に作られ、`--zip` を付けると同じ名前の ZIP も作ります。ダウンロードのキャッシュは `dist/cache/` に置きます。同じ版をビルドし直すと、前の出力フォルダ（`BUILD-INFO.json` があるもの）と作業用の `.partial` フォルダは置き換えられるので、残したい成果物は先に移してください。

```text
Yata-<version>-cpu/       # GPU 版は -gpu
  Yata.cmd
  BUILD-INFO.json
  LICENSE.txt
  THIRD_PARTY_NOTICES.md
  licenses/
  app/                   # yata、web、ロゴ
  runtime/               # CPython、依存ホイール、VC ランタイム
```

### ビルド中に確認すること

1. CPython **3.14.10** の Windows x64 embeddable ZIP を、固定した SHA-256 で検証します。展開した `python.exe`・`python314.dll` の Authenticode の署名者が Python Software Foundation であることも確かめます。
2. `uv.lock` から依存の一覧を書き出し、`speakers` extra を含むホイールを、ハッシュ必須・再解決なし・バイナリのみで入れます。使った一覧は `runtime/requirements.lock.txt` に残します。
3. GPU 版には cuBLAS **12.9.2.10** の `cublas64_12.dll` と `cublasLt64_12.dll` を置きます。取得元のホイールの SHA-256 を固定し、ライセンスを `runtime/cuBLAS-LICENSE.txt` に保存します（DLL 自体には署名がありません）。
4. CTranslate2 のホイールに含まれる未使用の `cudnn64_9.dll` を、両方の版から削除します（対象の Windows ビルドは `WITH_CUDNN=OFF`）。CTranslate2 と組み込みライブラリの追加のライセンス文書は `licenses/ctranslate2/` に置きます。
5. ホイールのパッケージ・版・DLL と追加のライセンス文書を、`THIRD_PARTY_NOTICES.md` の定義と照合し、載っていないものや食い違いがあればビルドを止めます。ルートの `LICENSE`、`THIRD_PARTY_NOTICES.md`、`licenses/` を成果物にコピーします。この照合だけでは再頒布の条件を満たしたことになりません。
6. PE のインポートを調べて足りない Visual C++ ランタイムの DLL を追加し、ビルド元のパスが入るコンソールランチャーを取り除き、同梱の Python で主なネイティブ依存を読み込めるか確かめます。
7. GPU 版では同梱の cuBLAS の読み込みを確かめ、GPU がある環境では小さなランダム重みの Whisper で CUDA の計算を試します。

同梱の Python は `python314._pth` で `app` と同梱ライブラリだけを参照し、ランチャーは `-I -B` で実行します。システムの Python やユーザーの site-packages は使いません。`BUILD-INFO.json` には、Yata・Python・依存パッケージの版、版の種類、extra、`uv.lock` の SHA-256、VC ランタイム、CUDA の確認結果、取得できれば Git のコミットと未コミットの変更の有無を記録します。配布するときは、ZIP の SHA-256 と検証結果も記録してください。

### 利用者の起動とデータ

ZIP を展開して `Yata.cmd` を起動し、Chrome か Edge で `http://127.0.0.1:8765/` を開きます。モデル・設定・セッションは OS のデータフォルダに置き、ZIP には利用者のデータを含めません。更新は配布フォルダを置き換える方式で、利用者のデータは消しません。

GPU 版は NVIDIA のドライバーを同梱しません。対応しない GPU、古いドライバー、DLL の欠落は、GPU がないときの CPU への切り替えとは別に確認が必要です。CTranslate2 4.8.5 では、cuBLAS がない状態で CUDA を明示的に指定すると、最初の失敗のあと同じプロセスでの再試行が止まることがありました。失敗したプロセスは使い回さずに終了してください。

### スモークテスト

ビルドしたフォルダを指定します（`0.2.0` は成果物の版に読み替えてください）。

```powershell
uv run python tools/portable_smoke.py dist\portable\Yata-0.2.0-cpu
uv run python tools/portable_smoke.py dist\portable\Yata-0.2.0-gpu
uv run python tools/portable_smoke.py dist\portable\Yata-0.2.0-gpu --hide-gpu
```

- `YATA_DATA_DIR`・`LOCALAPPDATA`・`APPDATA`・`USERPROFILE`・`TEMP`・`TMP` を専用の一時フォルダに向け、`PATH` から Python・uv・CUDA Toolkit を外して起動します。8765 番ポートが使われていれば始めず、終了時は自分で起動したプロセスを止めます。
- 同梱の実行環境での起動、ページ・静的ファイル・ヘルスと準備の API、Cookie のない要求の拒否、モデルの初期状態、CPU／GPU に応じた推奨、その時点の待ち受けと接続先を確かめます。GPU 版では同梱の cuBLAS の場所と読み込み、GPU が見える場合は小さな CUDA の計算も確かめます。
- `--hide-gpu` は、NVIDIA GPU のある機械で CPU 側の分岐を試す補助です。NVIDIA GPU のない実機の代わりにはなりません。
- 外向きの HTTP/HTTPS プロキシを届かないローカルの宛先に向け、TCP・UDP の状態も確認します。ただし OS での通信遮断ではないため、「外部通信が一切ない」ことの保証には使えません。

### 配布前に解決すること

同梱物の再頒布条件に、未解決の項目があります。詳細と根拠は [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md) を正とし、次を解決してから ZIP を公開するか判断します。

- **FFmpeg（x264・x265）**：PyAV に含まれる FFmpeg は GPL の x264・x265 を含みます。GPL の条件（対応するソースやビルドスクリプトの提供）に従うか、商用ライセンスを使うか、GPL 部分を含まないビルドにするかを決め、残すライブラリのライセンス文書と libiconv のソースの提供方法を整えます。PyAV 自体の BSD ライセンスだけでは説明できません。
- **Intel OpenMP**：CTranslate2 に含まれます。対象の版の再頒布リストと Developer Tools EULA の条件、配布物に必要な条項を確認します。ライセンス本文を同梱するだけでは解決しません。
- **Visual C++ ランタイム**：ビルド元の再頒布の権利と条件を確認します。アプリと同じフォルダに DLL を置くため、Windows Update では更新されません。脆弱性が見つかったときの更新方法と、VC 再頒布可能パッケージをインストールする方式にするかも決めます。
- **cuBLAS（GPU 版）**：NVIDIA の EULA の条件を満たします。ライセンス本文の同梱とハッシュの固定は、配布条件全体の確認とは別です。

成果物の実機での確認も残っています。

- Python・uv・CUDA Toolkit・既存のデータがない Windows x64 で、ZIP の展開から準備・モデルの取得・実際の音声の認識までを確かめる（NVIDIA GPU あり、GPU なし、AMD・Intel GPU、対象とする古い GPU・ドライバーを分けて記録する）
- 実際のモデルでの CUDA の動作、CPU への切り替え、ストリーミング、LLM、長時間の動作、Ctrl+C・再起動・更新でデータが残ることを確かめる
- ダウンロードした ZIP に付く Mark of the Web、SmartScreen、Smart App Control、セキュリティ製品の影響を、新しい Windows で確かめる（`.cmd` と配布物全体には署名がありません）
- 配布ページに、対応する OS・GPU・ドライバー、未対応の機能、展開・起動・更新・削除の方法、ZIP のハッシュ、ライセンスと必要なソースの入手先を載せる

### サードパーティの表示

[THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md) と [licenses/](../licenses/) は配布物の一部です。`tools/third_party_notices.py` が固定した版から一覧を作り直し、テストが `uv.lock` と一致することを確かめます。依存を更新したら、次を実行して一覧を作り直してください。

```powershell
uv run python tools/third_party_notices.py
```
