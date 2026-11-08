# 濾波順序完整交叉批次

每個 `factorial-s<種子>-<場景>.yaml` 是一個種子加一個場景的 120 面牆，5 個種子 × 5 個場景共 25 個，3,000 面。`run_batch.ps1` 把它們分成 6 個 8 小時批次來跑。

要跑多少、花多少時間與空間，以及為什麼這樣設計：`docs/plans/factorial-filter-order-batch.md`。

```powershell
powershell -File research\walls\factorial\run_batch.ps1 -Batch 1 -Python <python.exe>
```

同一個指令再跑一次就續跑，已完成的牆跳過。`-DryRun` 只列出每面牆的 `run_wall` 參數。
