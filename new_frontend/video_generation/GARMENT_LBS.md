# 跨衣服的 anatomical LBS 初始化修復

2026-11-05 已在本機 WSL 完成修正及驗證：33 項測試、Actor08 男女兩組動作完整模擬與 GG 抽幀、Actor05 長褲與 Actor08 短褲受控損壞測試通過。完整數據及限制見 [REAL_ASSET_VALIDATION.md](tests/REAL_ASSET_VALIDATION.md)。

這版把 Actor08 的共同問題整理成「同拓撲參考 → 人體區域 → 有界修正 → 重綁定」流程。
可用於上衣手臂、褲管、軀幹／腰部區域；不用 Actor08 專屬 X 座標裁切或舊頂點索引。
**程式可共用，每件衣服的參考檔與通過條件仍須各自驗證。**

目前支援固定尺寸、SMPL-X 55 joints 的 garment dict。這不是所有 mesh 都能自動修好的通用模型。
原本驗證過的 `sleeve_repairs` 保留，未設定新 `garment_repairs` 時不會改變既有結果。

## 找出的共同規則

| 規則 | 實作 |
|---|---|
| 壞掉的幾何可能導致錯誤最近鄰綁定 | 用整理前參考的 LBS weights 分區，不用壞結果的 weights 判定衣服部位 |
| 各體型和衣服尺寸不同 | 位置差、綁定距離、修正上限除以目標人體對應骨鏈長度，不使用固定袖口 X 座標 |
| 不是每個區域都需要恢復 | 推薦 axial profile 以沿骨鏈縮短量 P90 觸發恢復；僅側向偏移則保留輸入形狀做 body fit；健康區域保留 |
| 只搬動幾何還可能繼續綁錯 | 在相同解剖區域內找最近人體頂點，同步轉移 weights 和 posedirs |
| 修正不能無限制移動或撕扯網格 | 檢查最大位移、離人體距離、參考邊長變化、三角形方向；不通過就拒絕整份輸出 |
| 一件衣服的 indices 不能沿用到另一件 | 每件都提供自己的同拓撲、同頂點順序參考，檢查 faces、source betas 和陣列形狀 |

分區為 `left_arm`、`right_arm`、`left_leg`、`right_leg`、`torso`。
頂點由參考 weights 在這五區中的最大總權重決定歸屬，且需超過 membership 門檻。
v2 用末端區域作為參考錨點，沿網格連接關係解 harmonic residual：在修復範圍內平滑求解相對參考的位移，範圍外保持輸入。
若邊長／法向警示仍不過，最多擴張有限的鄰接範圍；不放寬門檻。綁定仍是最近人體頂點轉移，沒有學習新的蒙皮演算法。

人體分區不等於服裝語意：實際 Actor08 襯衫參考有 1,445／1,553 個頂點歸入左右臂，
4,142 個歸入軀幹，另有 584 個下擺頂點歸入腿區。因此不能把「腿區」自動解讀成褲子，
也不能只靠這些權重判定哪一片是袖口。配置要先按每件衣服檢查；參考 weights 本身錯誤時應先修參考。

## 什麼能修、什麼需要其他方法

- `restore-rebind`：針對 relax／untangle 後縮袖、局部偏移等問題，從可信的 canonical 參考恢復已選區域，再重綁定。
- `rebind`：幾何已合理但綁到錯誤人體區域時，只更新選定區域的 weights／posedirs。`clearance_m=0` 時不移動幾何。
- 袖子、長褲、短褲共用相同程式；腰部可先診斷 `torso` 分區。短褲不一定需要修復。
- 不能補救錯誤拓撲、缺布、鬚鬚、錯的 template pose、錯的 SMPL-X fit；這些應回原步驟修正。
- 裙子、披風、寬鬆跨多人體部位的布料可能無法用這五區可靠描述，不應直接套用。
- 不新增 pins、不訓練、不改碰撞器或材質；不保證解決後續布料模擬黏住、爆炸。

## 輸入與順序

```text
原本 fixed-size prepare（目標性別／betas）
    → anatomical repair 的診斷／修復
    → 20 幀 ContourCraft 測試＋視覺檢查
    → 完整 ContourCraft 模擬
    → GG inference／影片
```

需要：

1. 目標體型已完成 fixed-size 綁定的 dict：`shapedirs=0`，具有 source／target betas。
2. **這件衣服**整理前、canonical、同拓撲／同頂點順序的可信參考。
3. 本機合法取得的 SMPL-X 模型，以及原本可運行的 ContourCraft 環境。
4. 人工確認的區域 profile。左右依 SMPL-X 解剖定義。

參考檔用 `export_garment_reference.py` 匯出；它沿用已存在的安全匯出實作，並不限袖子。

```bash
"$CC_PY" export_garment_reference.py \
  --input-garment-dict /data/this_garment_unrelaxed.pkl \
  --source-fit /data/source_body_fit.pkl \
  --output-reference /data/this_garment_reference.pkl
```
不要用 Actor08 襯衫參考修褲子，也不要用另一個 topology 的衣服參考。
兩份 dict 必須位於相同 canonical 座標系；本工具不會先做 ICP 或反 LBS 對齊。

## 先診斷，不寫入修正版

在 `new_frontend/video_generation/`，指定既有環境的 Python：

```bash
CC_PY=/你的/ContourCraft環境/bin/python

"$CC_PY" overlay/ContourCraft/repair_garment_lbs.py \
  --input-garment-dict /data/target_bound_garment.pkl \
  --reference-garment-dict /data/this_garment_unrelaxed.pkl \
  --body-model-root /data/body_models \
  --profile repair_profiles/upper.example.json \
  --report /output/test01/diagnosis.json \
  --preview /output/test01/candidate.png
```

上衣可複製 `upper.example.json`，短褲用 `lower.example.json`，長褲用 `long_lower.example.json`。這組設定已在本次資料驗證；不同衣服仍需獨立檢查。
若要只看各區統計，建立內容為 `{"regions": []}` 的 profile；這時無法 apply。
診斷圖顯示**候選**幾何，標有 gate 狀態；rejected 的圖不能當已成功結果。
這是 canonical 正交檢查圖，不是相機照片對齊圖，也不是可見區域 IoU。

### Profile 參數

| 參數 | 初始值 | 意義 |
|---|---:|---|
| `mode` | `restore-rebind` | 或 `rebind` |
| `regions` | 必須人工選定 | 可以只選左手臂等單側；不自動修改所有分區 |
| `membership_threshold` | 0.45 | 參考頂點在所屬人體區域的最低權重總和 |
| `body_membership_threshold` | 0.50 | 可作為最近鄰候選的目標人體頂點區域權重 |
| `damage_fraction` | 0.10 | 觸發量 P90／骨鏈長度；axial 使用正向長度損失，spatial 使用空間偏差 |
| `max_displacement_fraction` | 0.75 | 任一候選頂點的最大位移／骨鏈長度上限 |
| `max_binding_distance_fraction` | 0.50 | 候選頂點離同區人體最近頂點的最大距離／骨鏈長度 |
| `max_edge_strain` | 0.35 | 涉及修正區的每條非退化邊，與參考長度相比的最大相對變化 |
| `max_support_fraction` | 0.80 | 拓撲過渡最多涵蓋整件衣服頂點的比例；擴張仍不通過則拒絕 |
| `restore_trigger` | `spatial` | 向後相容預設；本次通過的 profiles 明確選 `axial` |
| `centering_fraction` | 0 | 截面平移上限／所用骨段長度；推薦 0.30，需 trimesh |
| `align_limb_ends` | false | 長袖／長褲可設 true，末端沿軸平移到腕／踝；不延長短袖短褲 |
| `clearance_m` | 0 | 預設不做表面推離；需要時如 0.006，需 open3d 及可靠封閉人體表面 |

骨鏈：手臂為肩→肘→腕；腿為髖→膝→踝；軀幹為骨盆→脊椎→頸。
這些值是保守起始設定，**不是論文參數，也尚未證明適用所有衣服／體型**。
不要只為通過 gate 不斷放寬限制。寬鬆衣物離人體較遠不一定是錯；這時應人工判讀。
恢復後的邊長與方向檢查只防部分明顯異常，不是自交或穿透的完整證明。
過渡可能涵蓋選定解剖區域旁的混合權重頂點，報告會記錄支援範圍；需檢查肩部／髖部等交界。

## 確認後另存修正版

重跑同一指令，使用新的輸出路徑／report，另加：

```bash
  --apply --output-garment-dict /output/test02/garment_repaired.pkl
```

通過所有 gate 才寫入 dict；失敗仍會留下診斷報告，但不會輸出可供模擬的修正版。
輸入、參考、既有輸出均不可覆寫。
`rest_pos`、faces、node_type、材質、zero shapedirs 保留；修正的是初始化 `lbs.v` 及其蒙皮對應。
保留 rest mesh 並不等於初始化完全沒有拉伸，這正是需要 edge gate 與後續模擬檢查的原因。

## 接入同學原本的影片入口

先完成單件診斷，確認成功後，在原本 `paths.json` 增加：

```json
"garment_repairs": {
  "Your_shirt": {
    "reference_dict": "reference_assets/Your_shirt_unrelaxed.pkl",
    "profile": "repair_profiles/your_shirt.json",
    "genders": ["male"]
  },
  "Your_trousers": {
    "reference_dict": "reference_assets/Your_trousers_unrelaxed.pkl",
    "profile": "repair_profiles/your_trousers.json",
    "genders": ["male", "female"]
  }
}
```

衣服 key 必須符合 catalog；相對路徑相對於 `paths.json`。
同一件衣服不可同時列入 `sleeve_repairs` 和 `garment_repairs`。
需要不同性別的獨立 profile 時，使用不同的 paths 設定檔跑各組，不要共用不適合的參數。
`setup_data.py` 安裝整個 overlay：先按原 README 使用獨立 GG／ContourCraft repo，避免蓋掉其他平台修正。
設定新的 work_root／library_root／run-name，防止影片快取跳過新修正。

原本 `check_setup.py`／`render_one_video.py`／`batch_render.py` 會傳入配置；
prepare 後先修復，再模擬。保留 `.before_garment_repair.pkl`、JSON、PNG，並把修復報告寫入影片 sidecar。
若 gate 失敗，整個 pipeline 停在修復階段，不用壞候選跑模擬。

## Body fit 與形狀保護

axial profile 將「長度丟失」和「側向偏移」分開。前者以未 relax 參考為 shape reference，後者使用輸入本身（報告中的 `input_preserving_fit`）。邊長限制相對各自的 shape reference 計算，不可把兩者當同一基準比較。

截面平移保持截面尺寸，不直接縮放衣服。短衣物使用實際覆蓋的近端骨段，避免將短褲投到小腿軸上。harmonic residual 平滑接回相鄰區域，再交替限制邊長／法向與人體間距；支援範圍最多 80%，失敗就拒絕，沒有為通過測試放寬 gate。

SMPL-X 本次有 32 條嘴部開口邊。signed-distance 查詢使用獨立副本，只封小型、簡單、位於臉部權重區域的邊界，並確認封閉後使用 5-ray voting。非臉部／大開口會拒絕。來源模型、原模擬碰撞器與 garment topology 不修改。這符合 [Open3D signed-distance 的封閉表面假設](https://www.open3d.org/docs/latest/python_api/open3d.t.geometry.RaycastingScene.html)。6 mm clearance 只保證本次修復的支援頂點，不保證全衣服零穿透。

## 驗證與使用範圍

詳見 [實測報告](tests/REAL_ASSET_VALIDATION.md)。33 項測試通過；真實襯衫男女各測 `35_01` 90 個 trajectory frames 與 `137_26_t05` 160 個 frames，並查看首尾及極端動作 GG 圖。長短褲通過健康不變對照與真實網格上的受控 18 cm 縮管測試；後者不是自然發生的失敗，也不是完整長褲動態品質證明。

初始化前幾幀的 Gaussian 模糊仍存在。女性未選取的原始軀幹區域仍有少量穿透。normal-reversal proxy 不是自交／物理品質的完整證明。不同衣物仍須先短測，再驗證完整動作；15 cm 腕距只適用這次長袖實驗，不能套所有衣服。

```bash
"$CC_PY" -m unittest discover -s tests -p 'test_*.py'
```

只使用既有 WSL 環境；選用套件為 numpy、scipy、matplotlib，以及 centering 的 trimesh／clearance 的 open3d。未設定 `garment_repairs` 時維持原流程。所有工作只在本機分支，未 push；私人 garment reference 透過分享包提供，不加入 Git。
