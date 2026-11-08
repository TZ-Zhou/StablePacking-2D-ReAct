# Stones

data/stone_wall_freepik_mask_cut.png

The binary image is generated from segmenting a stone masonry wall image from [freepik](https://www.freepik.com/premium-photo/detail-traditional-dry-stone-masonry-from-south-france-cevennes_37918828.htm).



# Plotting

selected walls : wall9 and wall12

## 內容

| 項目 | 說明 |
| --- | --- |
| `00_run.py` | 讀入石塊遮罩圖，依序挑選、擺放並穩定石塊以堆出多面石牆，輸出各牆的位移旋轉紀錄與圖。 |
| `01_view_data_set.py` | 將石塊依形狀分群，輸出各石塊的群組編號表與依群組上色的原牆圖。 |
| `02_view_transformation_ori_reso.py` | 把堆牆得到的位移旋轉套回原解析度石塊，重建高解析度石牆影像。 |
| `03_evaluate_reconstructed_wall_metrics.py` | 計算重建石牆的石材占比、垂直與水平咬合指標，寫成指標文字檔。 |
| `04_evaluate_ori_wall_metrics.py` | 以相同指標計算原始石牆，結果寫成指標文字檔。 |
| `05_generate_gif.py` | 把逐顆放置的石塊影像與石塊資料集合成堆牆過程動畫 GIF。 |
| `graph.py` | 由石牆影像建立灰縫路徑圖，並提供咬合路徑的繪圖函式。 |
