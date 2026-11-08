# Stones

data/stones*noise*(10-80)\_labels.png: Binary wall images labeled from images in [Thangavelu et al.](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8460562)

# Wall construction

- 00_run.py: Generate 20 different layouts for each stone set. The possible stone poses range from 0-360 degrees with 90 degree interval. The input stone images are scaled by 0.5. Wall height is 5 times the average bounding box height of all regions larger than 50. Wall width is 5 times the average width of that. Built with Variant method.
- 01_run_with_10_interval.py: The possible stone poses range from 0-360 degrees with 10 degree interval.

## 內容

| 項目 | 說明 |
| --- | --- |
| `02_view_transformation_ori_reso.py` | 把牆的石塊變換結果套回原始石塊影像，輸出高解析度的牆影像。 |
| `03_evaluate_wall_metrics.py` | 對每組資料集所產生的牆計算咬合、填充與穩定度指標，寫入 `result_all` 的各雜訊等級指標檔。 |
| `04_evaluate_wall_ref.py` | 計算原始參考牆的同一組指標，寫入 `result_all/metrics_reference_wall.txt`。 |
| `05_plot_fav_fah.py` | 讀取各雜訊等級與參考牆的指標檔，繪製 FAV、FAH、填充與穩定度對雜訊等級的散布圖。 |
| `06_plot_filling_stability.py` | 讀取各雜訊等級的指標檔，繪製填充對穩定度的散布圖。 |
| `graph.py` | 由牆的二值影像建立灰漿路徑圖，供 `03_evaluate_wall_metrics.py` 與 `04_evaluate_wall_ref.py` 計算指標。 |
