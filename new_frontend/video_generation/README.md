# 批次影片產生工具

把「衣服 × 性別 × 身高 × 體重 × 動作」的各種組合跑成影片，放進影片資料庫。網頁展示時直接依使用者輸入播放對應的影片，不需要現場模擬。

這個 branch 自成一體：所有需要的程式都在 `new_frontend/video_generation/` 裡，不需要先取得 `main` 的任何內容。

多人分工的方式：每個人依分工，用範本填寫自己要跑的工作單，各自在自己的電腦上跑；跑完把輸出資料夾傳給負責合併的人，合併成同一個資料庫。

## 流程

```
 Hugging Face 資料包                         你的電腦
 (處理好的衣服、動作)      ┌───────────────────────────────────────────────┐
        │                  │                                               │
        ▼                  │  setup_data.py     安裝程式、接上資料         │
  下載到 data_root ──────▶ │          │                                    │
                           │          ▼                                    │
                           │  check_setup.py    自我檢查                   │
                           │     資料 → 環境與程式 → 試跑 20 幀            │
                           │          │ 全部通過                           │
                           │          ▼                                    │
  自己的工作單 ──────────▶ │  batch_render.py   批次執行器                 │
                           │     展開成一支支影片，依序處理                │
                           │     可暫停、可續跑、失敗自動重試              │
                           │          │ 每支影片                           │
                           │          ▼                                    │
                           │  render_one_video.py   單支影片 pipeline      │
                           └──────────┼────────────────────────────────────┘
                                      ▼
                            video_library/ 影片資料庫
```

## 資料夾內容

```
new_frontend/video_generation/
  README.md
  paths.example.json         路徑設定範本
  jobs.example.json          工作單範本
  catalog.example.json       catalog.json 的格式範例（實際的 catalog.json 在資料包內）
  toolkit_common.py          共用函式（設定檔、工作單展開、資料庫路徑）
  setup_data.py              安裝程式、接上資料
  sync_v3_controllers.py     把 V3 opacity controller 放進資料包
  check_setup.py             自我檢查
  diagnose_speed.py          影片跑得特別慢時，找出原因
  batch_render.py            批次執行器
  render_one_video.py        單支影片入口
  merge_library.py           合併資料庫
  build_index.py             產生索引
  overlay/                   跑 pipeline 需要的程式（見步驟 3）
    ContourCraft/
    Gaussian-Garments/
```

## 前提

你的電腦已有完整的官方 Gaussian-Garments 與 ContourCraft 程式碼，以及兩個可執行的 Python 環境：一個用來跑 ContourCraft（身體擬合、布料模擬），一個用來跑 Gaussian-Garments（Blender 算圖、影片合成）。環境名稱不限，只要在 `paths.json` 填入各自 Python 的路徑即可。官方 ContourCraft 的 `contourcraft.pth` 與 SMPL-X 身體模型應已安裝在 `ContourCraft/ccraft_data/` 之下。需要一張可用的 NVIDIA GPU。

## 步驟

### 1. 取得這個 branch

```bash
git clone -b new-frontend https://github.com/Realfit3D/frontend_temp.git
cd frontend_temp/new_frontend/video_generation
```

分支名稱是 **`new-frontend`**（不是 `main`）。這個分支只有 `new_frontend/` 這個資料夾，和 `main` 的內容完全不同，所以**請直接 clone 到新的資料夾**，不要在舊的 repo 裡切換分支。

之後如果作者有更新，在 `frontend_temp` 資料夾裡執行 `git pull`，再執行一次 `python setup_data.py --config paths.json` 把新程式放進你的官方資料夾。確認版本：`git log -1 --format=%h` 會顯示最新的版本編號。

### 2. 下載資料包

資料包放在 Hugging Face 的私人儲存庫：`https://huggingface.co/datasets/Realfit3D/tryon-garment-package`。只有組織 `Realfit3D` 的成員看得到，所以要先做下面的「第一次使用」。要使用 `--opacity-mode v3`，每件目標衣服都必須包含自己的 `opacity_controller_v3/controller.pt`。

**第一次使用（每個人做一次）：**

1. 註冊 Hugging Face 帳號，並接受組織 `Realfit3D` 的邀請。
2. 安裝指令：`pip install -U huggingface_hub`（裝好後有 `hf` 指令）。
3. 登入：`hf auth login`。它會印出網址和代碼，用瀏覽器打開、輸入代碼授權。想確認登入的是誰：`hf auth whoami`。

**下載資料包（約 2.2 GB，中斷了再執行同一行會接著下載）：**

```bash
hf download Realfit3D/tryon-garment-package --repo-type dataset --local-dir /你的/data_root
```

把本地已訓練好的 V3 controller 更新進資料包：

```bash
python sync_v3_controllers.py \
  --data-root /你的/data_root \
  --controller-root /你的/Gaussian_Garments_restart/output_actorshq_gg \
  --subject-map Actor08_shirt=Actor08
```

`--subject-map` 用來處理資料包衣服名稱和訓練 subject 名稱不同的情況。確認無誤後，將更新後的資料包上傳：

```bash
hf upload Realfit3D/tryon-garment-package /你的/data_root \
  --repo-type dataset --path-in-repo .
```

正式複製前可加 `--dry-run` 檢查對應關係；不要把整個 `output_actorshq_gg` 上傳，只同步每件衣服的 controller。

資料包的結構：

```
data_root/
  catalog.json                     衣服、動作、身高體重區間的清單
  thumbnails/<衣服名稱>.png        衣服縮圖（網頁用，由負責統整的人提供）
  garments/<衣服名稱>/
    stage1/template_uv.obj
    stage1/cameras.json
    stage3/ckpt/net.pt
    stage3/ckpt/info.txt
    opacity_controller_v3/controller.pt  # V3 透明度控制器
    cache/garment_dict.pkl         已處理好的衣服
    cache/source_body_fit.pkl
  motions/CMU/...                  動作檔（.npz）
```

### 3. 設定路徑

```bash
cp paths.example.json paths.json
```

編輯 `paths.json`：

```json
{
  "contourcraft_root": "/你的/ContourCraft",
  "gaugar_root": "/你的/Gaussian-Garments",
  "contourcraft_python": "/你的/ContourCraft環境/bin/python",
  "gaugar_python": "/你的/GaussianGarments環境/bin/python",
  "data_root": "/你的/data_root",
  "library_root": "/你的/video_library",
  "work_root": "/你的/work"
}
```

兩個 `_python` 欄位填各自環境裡 Python 的完整路徑。`library_root` 放最終影片，`work_root` 放執行時的中間檔。所有路徑都由這個檔案提供，不需要修改任何程式碼。

### 4. 安裝程式、接上資料

```bash
python setup_data.py --config paths.json
```

這一步做兩件事。第一，把 `overlay/ContourCraft/` 與 `overlay/Gaussian-Garments/` 內的程式複製進你的官方程式碼資料夾。其中 `inference.py` 是官方檔案的修改版，其餘是新增的檔案；若你的官方檔案與要複製的內容不同，原檔會先備份成 `.orig` 再覆蓋。第二，把資料包內的衣服以符號連結接到程式預期的位置：Gaussian-Garments 的 `output_root/<衣服>/`（位置由你的 `utils/defaults.py` 決定，也可在 `paths.json` 加 `gaugar_output_root` 指定），以及 `work_root/<衣服>/`。資料包內的 `opacity_controller_v3/controller.pt` 也會一起接到該衣服的模型目錄，影片即可使用 `--opacity-mode v3`。動作檔不需要連結，程式直接從 `data_root/motions/` 讀取。已存在的連結不會重複建立，已存在的實體檔案不會被覆蓋。可以重複執行。

### 5. 自我檢查

```bash
python check_setup.py --config paths.json
```

依序檢查三項，任何一項失敗都會說明缺什麼以及怎麼補（缺縮圖只會提醒，不影響跑影片）：

1. 資料：資料包內每件衣服的必要檔案、動作檔、官方模型是否都在。
2. 環境與程式：兩個 Python 環境、GPU、Warp 是否真的看得到 GPU、overlay 的程式是否已安裝到正確位置。
3. 試跑：用 20 幀產生一支測試影片，確認整條路線通，並列出各階段耗時；如果布料模擬比參考值（約 5 秒/步）慢 3 倍以上會提醒。測試影片放在 `work_root/smoke_test/`。

全部通過後再往下。

### 6. 填寫工作單

```bash
cp jobs.example.json my_jobs.json
```

編輯 `my_jobs.json`：

```json
{
  "defaults": { "fps": 25 },
  "jobs": [
    {
      "garments": ["Actor02_lower", "Actor02_upper"],
      "gender": "male",
      "heights": [165, 175],
      "weights": [55, 65],
      "motions": ["137_26_t05", "16_34"]
    }
  ]
}
```

| 欄位 | 說明 |
|---|---|
| `garments` | 一件下身加一件上身，下身在前、上身在後，例如 `["Actor02_lower", "Actor02_upper"]`。名稱要出現在 `catalog.json`，且 `slot` 必須分別是 `lower` 與 `upper` |
| `gender` | 身體性別，`male` 或 `female` |
| `heights` | 身高清單，整數，單位 cm。必須是 `catalog.json` 的 `heights` 裡各區間的 `value`（區間的中間值，例如 150–160 公分這一格要填 155） |
| `weights` | 體重清單，整數，單位 kg。必須是 `catalog.json` 的 `weights` 裡各區間的 `value`（例如 40–50 公斤這一格要填 45） |
| `motions` | 動作編號清單，要出現在 `catalog.json` |
| `frames` | **通常不用寫。** 不寫（或寫 `"auto"`）= 跑完整個動作檔，幀數由動作的長度決定（動作步數 − 10）。資料包裡的動作都已經裁成要用的長度，所以直接跑就是整個動作。需要固定長度時才寫整數 |
| `fps` | 影片的每秒幀數 |
| `opacities` | 選填，0 到 1 的透明度清單；預設 `[1.0]`，所以舊工作單行為不變 |
| `opacity_mode` | 選填：`direct`（預設）、`optical_depth` 或專題的 `v3`。`v3` 的 0 與 1 是精確端點；中間值需要套裝內每件衣服各自的 controller |
| `opacity_garments` | 選填，只調整套裝內指定的衣服，例如 `["Actor02_upper"]`；省略表示整套都調整 |

網頁讓使用者選的是身高、體重「區間」，所以影片要用區間的中間值來跑。`batch_render.py` 開跑前會檢查這點，填了網頁選不到的數值（例如 170）會直接列出錯誤，不會浪費時間跑出來。

**影片長度 = 動作長度。** 模擬每秒 30 步，一支影片的幀數 = 動作步數 − 10（開頭 10 格是布料落定、會被丟掉）。資料包的動作已經裁好，工具會自動算出幀數，不用自己填；`--dry-run` 會列出每支影片的幀數。只有手動填了 `frames` 且動作不夠長時才會報錯。`fps` 填 30 影片就是真實速度，填 25 會慢約 17%。

每個工作會對 `heights × weights × motions × opacities` 展開成所有組合。網頁讓使用者自由搭配上身和下身，但只有「有跑過的搭配」才能選，所以每一套「下身＋上身」都要在工作單裡列出才會有影片。`jobs` 內可以放多個工作。分工有調整時，只要修改自己的工作單。

例如要用專題 PDF 中的 V3 單調控制器產生四個透明度版本：

```json
{
  "defaults": {
    "fps": 25,
    "opacities": [0.25, 0.5, 0.75, 1.0],
    "opacity_mode": "v3"
  },
  "jobs": [
    {
      "garments": ["Actor02_lower", "Actor02_upper"],
      "gender": "male",
      "heights": [175],
      "weights": [65],
      "motions": ["137_26_t05"],
      "opacity_garments": ["Actor02_upper"]
    }
  ]
}
```

V3 controller 是依衣服訓練，不能把 Actor02_upper 的權重套給另一件衣服。上面的例子只調整 Actor02_upper，下身維持原本不透明，因此只需要 Actor02_upper 的 controller。若省略 `opacity_garments`，整套每件衣服都要有自己的 controller。權重的標準位置是 `data_root/garments/<衣服>/opacity_controller_v3/controller.pt`；也可以在 `paths.json` 加上 `"v3_controllers": {"衣服名稱": "/完整路徑/controller.pt"}`。缺權重時，`--dry-run` 會在開始耗時的模擬前直接列出問題。若只是先驗證流程，可用不需權重的 `direct` 或 `optical_depth`。

目前前端介面還沒有透明度選擇器；索引會優先把 1.0 版本交給現有介面播放，所以加入透明度影片不會偷偷改掉目前展示結果。各透明度與模式已寫進 `manifest.json`，之後只需在前端加選項與篩選，不必再改影片生成格式。

單獨跑一支透明度影片也可以：

```bash
python render_one_video.py --config paths.json \
  --garments Actor02_lower Actor02_upper --gender male \
  --height 175 --weight 65 --motion 137_26_t05 \
  --opacity 0.5 --opacity-mode v3 --opacity-garments Actor02_upper
```

### 7. 預估數量與時間

```bash
python batch_render.py my_jobs.json --config paths.json --dry-run
```

列出總共有幾支影片、已經完成幾支、預估還要多久，不會真的執行。**預估時間是「全部待產生影片加起來的總和」**，不是單支影片的時間。第一次執行時用的是作者電腦的參考速度，你的電腦可能差很多；跑完第一支影片後，會改用你這台電腦實際量到的速度重新推算。每支影片跑完會印出各階段（untangle、擬合身體、布料模擬、算圖）花了多久。

### 8. 開跑

```bash
python batch_render.py my_jobs.json --config paths.json
```

### 9. 暫停與續跑

執行中隨時可以用 Ctrl+C 暫停，或電腦中途關機、斷線也沒關係。再次執行同一個指令，程式會自動跳過已完成的影片，從剩下的繼續。

被中斷當下正在跑的那一支影片會從頭開始，但已完成的影片不會重做，也不會重做已經算好的衣服纏繞修正與身體擬合（這些結果快取在 `work_root`，只要衣服與身體相同就會重用）。

### 10. 失敗處理

某支影片失敗時，程式會自動重試。重試後仍失敗的，記錄在 `library_root/failed.log`，並繼續跑其他影片，不會中斷整批。下次執行同一個指令時，失敗過的影片會再嘗試一次。

要單獨查看失敗原因，看 `work_root/<該影片>/log.txt`。

### 11. 回傳結果

跑完後把影片上傳到 Hugging Face 的影片儲存庫 `Realfit3D/tryon-video-library`（私人，需要已登入且是組織成員、角色為 write）。先進到 `paths.json` 裡 `library_root` 那個資料夾，再執行：

```bash
cd /你的/library_root
hf upload Realfit3D/tryon-video-library . . --repo-type dataset \
  --include "*.mp4" --include "*.json" \
  --exclude "manifest.json" --exclude "catalog.json" --exclude "failed.log" \
  --commit-message "batch_01 by 你的名字"
```

- 只傳影片（`.mp4`）和說明檔（`.json`）。`manifest.json`、`catalog.json`、`failed.log` 是各自電腦的索引或紀錄，不要傳，會互相覆蓋。
- 資料夾結構會原樣保留，不要改名、不要搬動；不同批次的影片路徑不同，不會互相覆蓋。
- 建議等一批完全跑完再上傳。中斷了再執行同一行即可。
- 也可以用網頁：儲存庫頁面 → Files and versions → Contribute → Upload files，把資料夾拖進去（一次可上傳的檔案數量有上限，檔案多時用指令比較穩）。

另外，用 `--skip-smoke` 可以讓 `check_setup.py` 略過 20 幀試跑，只做前兩項檢查（資料、環境與程式）。

## 合併（只有負責合併的人需要做）

先把影片儲存庫整個下載下來：

```bash
hf download Realfit3D/tryon-video-library --repo-type dataset --local-dir ~/all_videos
```

再合併並產生索引：

```bash
python merge_library.py ~/all_videos --config paths.json
python build_index.py --config paths.json
```

`merge_library.py` 把別人的影片複製進總資料庫，已經存在的路徑會跳過，內容不同的衝突會列出來。分工是滾動調整的，不同人偶爾可能跑到同一支影片，合併時自動處理。`build_index.py` 掃描整個資料庫，重新產生 `video_library/manifest.json`，網頁用這個檔案找影片；同時把資料包裡的 `catalog.json` 與 `thumbnails/` 複製進資料庫資料夾（網頁需要它們來顯示衣服名稱與縮圖），所以做完這一步，資料庫資料夾就是網頁可以直接使用的狀態。縮圖不是跑影片產生的，由負責統整的人提供在資料包內；缺縮圖時 `check_setup.py` 會提醒，但不影響跑影片。

## 影片資料庫格式

```
video_library/
  <下身>+<上身>/            例：Actor02_lower+Actor02_upper
    <gender>/               male 或 female
      h170_w060/            身高 cm、體重 kg，體重補零到三位
        01_01.mp4
        01_01.json          說明檔
        01_01__opacity_v3_0p5__target_Actor02_upper.mp4
        01_01__opacity_v3_0p5__target_Actor02_upper.json
  manifest.json
  failed.log
```

路徑完全由參數決定，所以相同參數的影片永遠在同一個位置，不同人產生的資料夾可以直接合併。

說明檔欄位：

```json
{
  "garments": ["Actor02_lower", "Actor02_upper"],
  "gender": "male",
  "height": 170,
  "weight": 60,
  "motion": "01_01",
  "frames": 250,
  "fps": 25,
  "opacity": 0.5,
  "opacity_mode": "v3",
  "opacity_garments": ["Actor02_upper"],
  "created_at": "2026-10-30T12:00:00",
  "elapsed_seconds": 2400,
  "machine": "主機名稱",
  "status": "done"
}
```

`status` 的值：`done`（完成）、`failed`（失敗）。

## 各程式的實作

**`render_one_video.py`** 產生單支影片。輸入是衣服、性別、身高、體重、動作、幀數、fps、透明度與模式。它呼叫 `ContourCraft/run_outfit_preview_pipeline.py`，依序完成：依身高體重擬合 SMPL-X 身體、把衣服重新綁定到這個身體、布料模擬、Blender 算圖、在 Gaussian 光柵化前調整每個 Gaussian 的 opacity、合成 mp4。`frames` 是最後影片的幀數；模擬開頭 10 幀布料尚未穩定，會被捨棄，而軌跡長度是模擬步數 + 2，所以模擬步數是 `frames + 8`。渲染完成後程式會檢查實際幀數是否等於 `frames`，不符就視為失敗。完成後把影片和說明檔寫到資料庫對應路徑。

**`batch_render.py`** 讀工作單、展開成影片清單，逐支呼叫 `render_one_video.py`。展開後的順序先依衣服組合、再依身體（性別、身高、體重）、最後依動作排列，讓同一套衣服只做一次纏繞修正，同一個身體只擬合一次。每支開跑前先檢查說明檔，狀態為 `done` 且幀數與 fps 都和工作單相同才跳過（長度不同視為另一支影片，會重新產生）；失敗就重試；每支影片的結果都立刻寫入，所以任何時刻中斷都不會損壞已完成的部分。

**`check_setup.py`** 逐項檢查資料、環境與程式，最後以 20 幀實際跑一支影片。

**`setup_data.py`** 安裝 overlay 程式並把資料包接到官方程式碼預期的位置，資料用符號連結，不複製。

**`merge_library.py`** 與 **`build_index.py`** 如上節所述。

**`catalog.json`**（在資料包內，格式見 `catalog.example.json`）記錄：身高與體重的區間（`id`、顯示名稱 `label`、跑影片用的中間值 `value`）、所有衣服（顯示名稱、`slot` 為 `upper` 或 `lower`、處理時使用的性別）、動作（編號、動作檔路徑、顯示名稱）。網頁也讀同一份檔案來顯示名稱與區間。

## 影片跑得比預期慢很多時

作者的電腦（RTX 5060 筆電）上，動作一、三（150 幀）每支約 12~15 分鐘，動作二（80 幀）每支約 7~11 分鐘；每套衣服的每個性別，第一次多約 4~5 分鐘（untangle，只做一次，之後同一套衣服的其他身形不用再做）。如果你的電腦差很多（例如單支要好幾個小時），請照下面做：

1. 確認拿到的是最新的程式：`git pull`，再執行一次 `python setup_data.py --config paths.json`。
2. 執行診斷（約 1 分鐘）：

   ```bash
   python diagnose_speed.py --config paths.json
   ```

   它會檢查：你的程式檔是否和 GitHub 上的版本一致（有沒有漏拿新程式）、GPU 有沒有被降速或被其他程式佔用、顯示記憶體夠不夠、PyTorch 與 Warp 有沒有真的用到 GPU、磁碟速度、會拖慢速度的環境變數，並把最近一次影片各階段的耗時和參考值並排列出，慢在哪個階段一眼就看得出來。報告最後有一段「結論」，用白話說明哪裡有問題。
3. 把整份報告（也存在 `diagnose_report.txt`）貼給負責人。如果結論說程式檔不一致，照第 1 步更新即可。

常見的原因：筆電沒接電源或在省電模式、GPU 被瀏覽器或其他程式佔用、顯示記憶體不夠、Warp 沒有用到 GPU（會退回 CPU，不會報錯但慢好幾倍）、`work_root` 放在 Windows 磁碟（`/mnt/c` 之類）。

## 常見問題

| 狀況 | 處理 |
|---|---|
| `check_setup.py` 說找不到某件衣服的檔案 | 確認資料包已完整下載到 `data_root`，再重新執行 `setup_data.py` |
| 執行時出現 CUDA out of memory | 關掉其他使用 GPU 的程式後重跑；已完成的影片不會重做 |
| 影片跑得比預期慢很多（例如 56 幀超過 30 分鐘） | 執行 `python diagnose_speed.py --config paths.json`（約 30 秒），它會檢查 GPU 是否被降速或被其他程式佔用、**剩餘的顯示記憶體夠不夠**（模擬會用到約 7.9 GB，不夠時 Windows 會悄悄改用系統記憶體，速度慢好幾倍而且不會報錯）、PyTorch 與 Warp 是否真的用到 GPU、磁碟是否太慢，並列出結論。把完整輸出貼給負責人 |
| 某支影片一直失敗 | 查看 `library_root/failed.log` 與 `work_root/<該影片>/log.txt` |
| 中途關機或斷線 | 重新執行同一個指令即可繼續 |
| 說找不到 Python 或套件 | 檢查 `paths.json` 內兩個 `_python` 路徑，確認該環境裝有對應套件 |
