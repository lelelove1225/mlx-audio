# ReFelicia TTS WebSocket Protocol (`/ws/tts`)

このドキュメントは、`mlx_audio.server` が提供する ReFelicia / SBV2 互換 TTS WebSocket の
ワイヤープロトコル仕様です。**この仕様を満たすサーバーであれば、TTS エンジンや実行環境
(MLX / CUDA / CPU など)を問わず、ReFeliciaV2 クライアントがそのまま接続できます。**

CUDA 環境などで互換サーバーを再実装する場合、満たすべきはこのプロトコルだけです。
ボイスクローンの参照音声の解決方法やサンプラーのパラメータは各サーバーの自由です。

---

## 1. エンドポイントと接続ライフサイクル

- エンドポイントは **`WS /ws/tts`** の 1 本のみ。
- 接続は **永続** で、1 接続が複数発話をループ処理する:

```
client → JSON request
server → {"type":"start", ...}
server → <binary PCM frame> (1 個以上)
server → {"type":"end", ...}
         ↑ ここまでで 1 発話。接続は閉じず、次の JSON request を待つ
...（切断まで繰り返し）
```

**1 リクエストで接続を閉じてはいけない。** 切断はクライアント側の disconnect で行われる。

---

## 2. リクエスト (client → server, JSON テキストフレーム)

クライアントは 1 発話ごとに次の JSON を 1 つ送る。互換性のために **必ず使う必要があるのは
`text` と `delivery_mode` のみ**。それ以外はエンジン固有パラメータにマップするか無視してよい。

```jsonc
{
  "text": "今日はいい天気ですね。",   // 必須。合成対象テキスト
  "delivery_mode": "full_audio",      // "full_audio" | "streaming"（5 節参照）

  // --- 以下は任意。互換サーバーは無視しても接続は成立する ---
  "model": null,                      // モデル指定（サーバー側のモデル選択に使ってよい）
  "voice": "",                        // 話者指定。Irodori 系では ref_audio / caption 解決に流用可
  "lang_code": null,
  "speed": 1.0,                       // 話速
  "pitch_scale": 1.0,
  "intonation_scale": 1.0,
  "streaming_interval": 0.24,

  // --- streaming モード時のセグメント分割パラメータ（full_audio では無視される）---
  "segment_target_chars": 18,
  "segment_max_chars": 28,
  "crossfade_ms": 80,
  "trim_silence_ms": 60,
  "silence_threshold": 192,

  // --- Irodori-TTS 固有の任意パラメータ（他エンジンでは無視してよい）---
  "seconds": null,
  "duration_scale": null,
  "num_steps": null,
  "sequence_length": null,
  "cfg_guidance_mode": null,
  "t_schedule_mode": null,
  "sway_coeff": null,
  "instruct": null,
  "caption": null
}
```

---

## 3. レスポンス (server → client)

発話ごとに **3 種類のメッセージを順番に** 送る。

### ① start（JSON テキストフレーム）

PCM より前に必ず送る。

```json
{
  "type": "start",
  "mode": "tts",
  "delivery_mode": "full_audio",
  "sample_rate": 48000,
  "channels": 1,
  "format": "pcm_s16le"
}
```

### ② 音声（バイナリフレーム、1 個以上）

生の PCM を WebSocket バイナリフレームとして送る。

- **符号付き 16-bit リトルエンディアン (PCM s16le)、モノラル (channels = 1)**
- WAV ヘッダ等は付けない。**生 PCM のみ**
- クライアントは `sample = (short)(b[i] | (b[i+1] << 8))` でデコードする

### ③ end（JSON テキストフレーム）

発話の終端マーカー。

```json
{
  "type": "end",
  "chunks": 1,
  "delivery_mode": "full_audio",
  "audio_seconds": 5.5,
  "generate_seconds": 0.8,
  "real_time_factor": 0.15,
  "first_pcm_ready_seconds": 0.8,
  "start_sent_seconds": 0.8,
  "first_chunk_sent_seconds": 0.8,
  "segments": 1
}
```

`end` の数値フィールドはトレース用テレメトリ。**クライアントの再生に必須なのは
`{"type":"end"}` が届くこと自体**であり、値はダミーでも接続は成立する。計測したい場合のみ埋める。

### エラー（JSON テキストフレーム）

生成中に例外が出た場合は次を送り、**接続は維持して**次のリクエストを待つ。

```json
{"type": "error", "message": "..."}
```

---

## 4. 音声フォーマット契約 ⚠️ 最重要

**`start` で宣言する `sample_rate` は、実際に送る PCM のサンプルレートと必ず一致させること。**

クライアントは `start.sample_rate` を信じてそのまま再生する。例えば 48kHz の PCM を送りながら
`sample_rate` を 44100 と宣言すると、再生がリサンプルされずピッチ・話速が狂い、
「声が似ない / 遅い」状態になる（実際に踏んだ不具合）。

- Irodori-TTS のネイティブ出力は **48000 Hz**
- `channels` は必ず **1**（クライアントはモノラル以外を例外で拒否する）
- `format` は `"pcm_s16le"`

クライアント側にサンプルレートを上書きする設定がある場合は **0（= サーバー申告値に従う）** にすること。

---

## 5. delivery_mode（2 モード）

メッセージの **種類と順序は両モードで同一**。違うのは **PCM を送るタイミングだけ**。

| モード | 挙動 | 用途 |
|---|---|---|
| `full_audio` | 全文を生成しきってから `start` → 全 PCM → `end` | 最小実装。まずこれだけで互換成立 |
| `streaming` | テキストを文単位に分割し、最初のセグメントが生成でき次第 PCM を送り始める | TTFA(最初の音が出るまで)の短縮 |

`streaming` でも `start` / `end` の形は同じで、PCM フレームがセグメントごとに分かれて届くだけ。
クライアントは最初のバイナリフレーム到着で即再生を開始する。**`streaming` は後回しでよい。**

---

## 6. 最小互換サーバー実装チェックリスト

CUDA + PyTorch などで再実装する場合、以下を満たせば ReFeliciaV2 クライアントが接続できる。

1. `WS /ws/tts` を立てる（FastAPI / websockets / aiohttp 等、何でもよい）
2. ループで `receive_json()` → `text` を取り出し → TTS 生成
3. 生成音声を **48kHz・モノラル・int16・リトルエンディアン** の bytes に変換
4. `start`（`sample_rate=48000, channels=1, format="pcm_s16le"`）→ `send_bytes(pcm)` → `end` の順で返す
5. 例外時は `{"type":"error","message":...}` を返し、**接続は維持**
6. `WebSocketDisconnect` でループを抜ける

ボイスクローンの参照音声はサーバー側の都合で解決してよい(環境変数 / リクエストの `voice`・`ref_audio` 等)。
クライアントは PCM が返ってくれば中身を問わない。

---

## 参考: 本サーバー(`mlx_audio.server`)の環境変数

MLX 実装ではモデルと既定パラメータを環境変数で設定する。互換サーバーで踏襲する必要はないが、
参考として記載する。

```bash
MLX_AUDIO_REFELICIA_TTS_MODEL=mlx-community/Irodori-TTS-600M-v3-VoiceDesign-8bit
MLX_AUDIO_REFELICIA_TTS_REF_AUDIO=/path/to/reference.mp3   # 10〜30秒のクリーンな単一話者音声
MLX_AUDIO_REFELICIA_TTS_NUM_STEPS=6                        # 拡散ステップ数（小さいほど速い）
MLX_AUDIO_REFELICIA_TTS_CFG_GUIDANCE_MODE=independent      # alternating は低ステップでクローン劣化
MLX_AUDIO_REFELICIA_TTS_INSTRUCT=...                       # VoiceDesign 用キャプション（任意）
MLX_AUDIO_REFELICIA_TTS_DUMP_WAV=/path/to/capture.wav      # デバッグ: 返した音声を WAV 保存（任意）
```
