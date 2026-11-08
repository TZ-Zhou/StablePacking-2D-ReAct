# Stonepacker2D：把不規則石塊逐顆堆成穩定石牆的 2D 堆疊函式庫

## 內容
| 項目 | 說明 |
| --- | --- |
| `entity/` | 幾何實體：`Base`（已放置石塊的牆底）、`Stone`／`Brick`／`Wedge`（石塊、矩形磚、楔塊）；`transform.py` 做旋轉、穩定檢查與下移／左移，`wedge.py` 找出並補上楔塊 |
| `evaluate/` | 候選姿態的評分：`local_void.py` 算局部空隙與卷積指標，`tilting_angle.py` 建剛體元素並解運動學極限，`ksteps.py` 向前多放 k 顆做試探 |
| `stack/` | 疊牆流程：`build_wall.py` 的 `build_wall_vendor` 逐批取石並放置，`place_stone.py` 的 `place_one_stone` 列舉姿態並算特徵，`selection.py` 做階層式篩選（含 `paper`／`code_original` 兩種策略與篩選順序控制），`vendor.py` 的 `Vendor` 管理石塊庫存與分群 |
| `utils/` | 共用工具：`constant.py` 集中全域設定的 getter／setter，`kine_2d.py` 為 2D 剛體運動學的元素與接觸點，`logger.py` 為日誌，`plot.py` 為矩陣存圖 |
| `__archive__/` | 舊的評估函式（`first.py`），未被任何模組引用 |

## 使用
在 repo 根目錄 `pip install -e .` 後，以 `import Stonepacker2D` 使用；`__init__.py` 已匯出 `entity`、`evaluate`、`stack`、`utils` 的公開名稱。
運動學求解依賴 `mosek`（已列入 repo 根目錄的套件依賴），執行時需要 MOSEK 授權。
