# Actor08 袖口／LBS 初始化修正

這份工具整理自已實際跑完影片的 Actor08 襯衫修正版，讓另一台電腦能以相同流程使用。**預設關閉，只有 `paths.json` 明確列出的衣服與性別會啟用。** 不需要修改 GG Stage 1/2/3 成果，也不需要 pin 袖口或 finetune。

## 修正的問題與範圍

Actor08 男生的部分 `untangle` 結果把左袖口從手腕附近拉回手肘。後續直接對這份幾何做 nearest-body-vertex 綁定，袖口就可能綁到上臂；只修穿透仍無法恢復袖長。

本工具有兩個模式：

| 模式 | 用途 | 實際修改 |
|---|---|---|
| `restore-rebind` | 本次 Actor08 男生驗證版本 | 從整理前參考檔恢復袖子、肩部平滑銜接，按目標人體重採 LBS weights 與 posedirs，再做徑向置中與人體表面間距修正，最後重新綁定袖子 |
| `radial` | 本次 Actor08 女生使用的版本 | 只做徑向置中與人體表面間距修正；保留原本 weights/posedirs |

這是**幾何初始化與對應關係的修正**；蒙皮仍是原本的 LBS，不是新的 LBS 演算法。`rest_pos`、faces、node_type、材質設定與短褲均保留。`restore-rebind` 會改動襯衫袖子區域的 `lbs.v`、`lbs_weights`、`posedirs`；`radial` 只改 `lbs.v`。

目前數值設定針對 **Actor08 長袖襯衫、SMPL-X 55 joints、原始頂點順序、以公尺為單位的 canonical T-pose（手臂沿 ±X）** 驗證。不是自動適用所有服裝的通用修復器。其他衣服請先另外測試，勿直接全域啟用。

## 需要哪些檔案

Git 分支內提供：

```text
video_generation/
  SLEEVE_LBS.md
  paths.sleeve-repair.example.json
  export_sleeve_reference.py
  overlay/ContourCraft/repair_sleeve_lbs.py
  overlay/ContourCraft/check_sleeve_rollout.py
```

另有一個必要的私人資料檔：

```text
reference_assets/Actor08_shirt_unrelaxed.pkl
```

這是同拓撲、**relax／untangle 前**的袖子參考，只包含 faces、原始 `lbs.v/shapedirs/lbs_weights` 與來源 betas 等對照資訊，不包含 SMPL-X 模型或 GG checkpoint。參考檔約 **3.1 MB**，放在 `.gitignore` 中，不會跟程式一起 commit。請向提供這次修正版的同學取得附帶的參考檔分享包，解壓到本目錄。**Git 分支本身不含這個私人資料檔。**

這次整理的參考檔 SHA256：

```text
45dd2c8ddf27ec7e403b12787a04dfdda8da42fc8f6ca635ac421b4a4ce70c53
```

不要把已縮袖的 untangled dict 或一般下載到的 relaxed dict 當成 unrelaxed reference。檔案內舊的 `recovery_metadata.relaxed` 可能是沿用的標記，**無法僅憑這個欄位判定幾何是否 unrelaxed**。參考檔供本工具讀取，不要拿它取代 pipeline 的完整服裝快取。

若你自己確實保存了整理前的完整 dict，可以匯出最小參考檔：

```bash
CC_PY=/你的/ContourCraft環境/bin/python

"$CC_PY" export_sleeve_reference.py \
  --input-garment-dict /你的/原始未relax襯衫.pkl \
  --source-fit /你的/Actor08_shirt/cache/source_body_fit.pkl \
  --output-reference reference_assets/Actor08_shirt_unrelaxed.pkl
```

匯出工具只挑選既有陣列，**不會重新 unpose 或重建 unrelaxed 幾何**。來源 fit 必須與這件衣服相符；不會把其他使用者或女生的目標 betas 當作男生設定。

## 用原本批次入口跑影片（建議）

1. 在這個分支的 `new_frontend/video_generation/` 操作，將參考檔放好。保留原本可執行的 GG／ContourCraft 環境，並指定其 Python：

   ```bash
   CC_PY=/你的/ContourCraft環境/bin/python
   ```

2. 複製設定範本並填入你自己的路徑：

   ```bash
   cp paths.sleeve-repair.example.json paths.sleeve.json
   ```

   使用新的 `work_root` 與 `library_root`，例如 `work_sleeve_repair`、`video_library_sleeve_repair`。原本標記為 done 的影片會被 batch 跳過，因此**不要直接沿用已完成舊版影片的 library_root**。既有的 `work_root/_betas` 可以自行複製到新 work_root 重用，不要把舊的 outfit untangle／fixedsize dict 當作新修復結果。

   額外設定只有這段；相對路徑以 **paths.sleeve.json 所在目錄**為基準：

   ```json
   "sleeve_repairs": {
     "Actor08_shirt": {
       "reference_dict": "reference_assets/Actor08_shirt_unrelaxed.pkl",
       "gender_modes": {
         "male": "restore-rebind",
         "female": "radial"
       }
     }
   }
   ```

   只想修男生時，刪掉 `female` 那一行。移除整個 `sleeve_repairs` 就是原流程。衣服名稱可與 catalog 一致地換名，但參考檔必須是該件衣服的同頂點、同 faces 順序版本。

3. 安裝此分支的 overlay：

   ```bash
   "$CC_PY" setup_data.py --config paths.sleeve.json
   "$CC_PY" check_setup.py --config paths.sleeve.json --skip-smoke
   ```

   `setup_data.py` 會按照原本流程安裝整份 overlay，包含修改版 `inference.py`；不同的既有檔案會先備份成 `.orig`。若電腦另有 EGL／CPU bake 等平台修改，請使用隔離的 repo 或先比較差異，避免直接蓋掉可用版本。本修復工具本身沒有更改 GG 渲染演算法或相機。

   修復使用 ContourCraft Python 中既有的 `numpy torch smplx scipy trimesh open3d`；預覽圖另外需要 `matplotlib`。安裝在自己的專案環境（伺服器遵守 global micromamba / local uv 規則），不用 sudo 或更動共用環境。

4. 先跑一支 20 幀檢查，不要一開始就跑整批：

   ```bash
   "$CC_PY" render_one_video.py --config paths.sleeve.json \
     --garments Actor08_shorts Actor08_shirt \
     --gender male --height 175 --weight 75 \
     --motion 35_01 --frames 20 --fps 25 \
     --output /你的/檢查輸出/actor08_male17575_20frames.mp4
   ```

   這個額外 output 不會把正式資料庫中的 `35_01` 標成完成。檢查通過後照原命令批次跑即可：

   ```bash
   "$CC_PY" batch_render.py batches/batch_A08.json --config paths.sleeve.json --dry-run
   "$CC_PY" batch_render.py batches/batch_A08.json --config paths.sleeve.json
   ```

   動作由工作單決定；若只要 `35_01`，請把工作單 `motions` 改成 `["35_01"]`，不要直接用包含其他動作的完整工作單。

## 修復在流程的哪裡

```text
原本 recover/relax 與 outfit untangle
    → 依目標體型做 fixed-size prepare
    → 此次袖口修復（每個性別、體型分開計算）
    → 原本 ContourCraft 模擬
    → GG 渲染／影片
```

修復只寫入該次 run 的獨立 dict，並保存：

```text
work_root/outfits/<outfit>__<gender>/runs_preview/<run>/fixedsize_dicts/<outfit>__<gender>/
  Actor08_shirt.before_sleeve_repair.pkl  修復前備份
  Actor08_shirt.pkl                      模擬實際讀取的修復結果
  Actor08_shirt.sleeve_repair.json        幾何／來源 SHA256 報告
  Actor08_shirt.sleeve_repair.png         canonical 前後對照
```

完成影片的 JSON sidecar 也有 `sleeve_repairs` 報告，即使 `render_one_video.py` 清除暫存 run，仍能核對採用哪個模式及參考檔。

**順序不能顛倒。** 修復後若再跑 relax、untangle 或 fixed-size prepare，可能把修復蓋掉。共用 reference、HF garment cache、untangle cache 均只讀；原始資料不會被工具覆寫。已有同名輸出會拒絕覆蓋，請使用新的 run 名稱或輸出目錄。

## 只修一份 dict，不跑影片

獨立工具也能處理單件襯衫，不需要提供短褲或建立 legacy `Actor05/Actor08` 假檔名：

```bash
"$CC_PY" overlay/ContourCraft/repair_sleeve_lbs.py \
  --input-garment-dict /你的/fixedsize_dicts/Actor08_shirt.pkl \
  --reference-garment-dict reference_assets/Actor08_shirt_unrelaxed.pkl \
  --body-model-root /你的/ContourCraft/ccraft_data/aux_data/body_models \
  --target-gender male --mode restore-rebind \
  --output-garment-dict /你的/新輸出/Actor08_shirt.pkl \
  --report /你的/新輸出/repair.json \
  --preview /你的/新輸出/canonical.png
```

輸入必須是 `contourcraft_fixed_size_upper.py prepare` 之後的 dict，含來源／目標 betas、正確目標 gender 與已清零的 garment shapedirs。`--mode radial` 不需要 reference，且不會改 weights/posedirs。這個入口只修改 dict，並未替公開 Web 的單件服裝流程增加完整支援。

## 怎麼驗證，不能只看「穿透 0」

1. 看 canonical 對照：兩邊袖口應靠近手腕，不能已經縮到手肘。
2. 跑短動畫，再看完整動畫首／中／尾及手部動作最大的幀：檢查滑脫、穿透、被身體黏住、碎片、異常拉扯。
3. 用保存的 trajectory 做動態檢查：

   ```bash
   "$CC_PY" overlay/ContourCraft/check_sleeve_rollout.py \
     --trajectory /你的/完整trajectory.pkl \
     --garment-dict /你的/模擬實際使用的/Actor08_shirt.pkl \
     --reference-garment-dict reference_assets/Actor08_shirt_unrelaxed.pkl \
     --body-model-root /你的/ContourCraft/ccraft_data/aux_data/body_models \
     --garment-name Actor08_shirt \
     --report /你的/新輸出/rollout_check.json
   ```

這個指標以參考袖子 ±X 最外側 4% 頂點當作袖口候選，追蹤其中心至 SMPL-X 手腕表面 patch 中心的距離；不是人工標註的 cuff，也不是 GT 誤差。報告會列出最大距離所在幀，請另外查看那一幀。可自行加 `--max-distance-cm 15` 作該次實驗門檻，但**沒有預設硬性 15 cm 退件**：固定尺寸衣服穿在較小身體上時，本來就可能超過手掌；女生 155/45 的一支就是經畫面審核後接受的結果。單看距離、fitness 或初始穿透都不足以宣告整段品質良好。

## 本機驗證紀錄

移植版應與既有實驗的數值結果對照，不是只做語法檢查。隨附 `tests/regression_sleeve_repair.py` 接受你自己的驗證資料路徑，比較 `lbs.v`、weights、posedirs，並檢查 rest_pos、faces、node_type 與 reference 未變。另有小型設定／防誤用測試：

```bash
"$CC_PY" -m unittest discover -s tests -p 'test_sleeve*.py' -v
```

預期效果依體型與動作而異；目前已驗證的男生 175/75 `35_01` 完整動畫，袖口中心至手腕 proxy 最大約左 5.05 cm、右 4.41 cm。不要把這些數字當作其他衣服的保證。
