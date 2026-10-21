# AI 協作紀錄

本作業程式由 Codex 協助建立，依據本地第五堂講義第 54–56 頁，資料夾配置參考 `C:\Users\u1021\Downloads\MMIP\week4`。

實作範圍：Beans CNN/ViT 檢索、Flickr8k 圖文融合配對分類、CLIP encoder + GRU decoder，以及 Week 4/5 同影像 BLEU/BERTScore 比較。

程式使用本地 allen_liver 的 PyTorch 與 CUDA；未改動 TensorFlow、Keras、NumPy、原有 Week 4 程式與權重。
訓練結果只由實際執行產生。smoke 為隨機資料驗證；results/validation 為小樣本整合驗證，兩者不可當作完整作業結果。

請閱讀模型、訓練／驗證／測試切分與指標定義，並依課堂規定揭露 AI 協作。
https://chatgpt.com/s/cx_6ad8597462a08191bc61778134ac215a