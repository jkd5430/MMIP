# 第五週完整實驗結果分析

## 是否還有必跑項目

依第五週三題的基礎與進階要求，目前五個模型的完整資料訓練、測試推論、結果展示、Backbone 比較，以及 Week 4/5 的 BLEU/BERTScore 比較都有輸出，沒有發現未執行的必跑實驗。這代表實驗項目齊備，實際評分仍由授課老師判定。

此次分析使用 `results/` 的正式輸出，所有模型 `limit=null`；不使用 `results/validation/` 的 32 張除錯結果。Beans train/validation/test = 1034/133/128；Flickr8k = 6000/1000/1000 張影像。

本次核對 checkpoint、history、metrics、predictions 與圖表，確認官方 split 的 ID 不重疊；從預測檔獨立重算檢索 P@5/mAP@5、圖文 Accuracy/Macro-F1/AUC、三種描述的 BLEU-1 至 BLEU-4，與已存正式指標一致。BERTScore 已有完成紀錄與模型設定；本次未重跑 BERTScore 網路計算。

## Q1：ViT 的檢索結果較好

測試 128 張 query，檢索 train 中的 1034 張 gallery，以同疾病／健康類別視為相關。

| 模型 | P@5 | mAP@5 | Hit@5 |
| --- | --- | --- | --- |
| ResNet18 | 0.8875 | 0.8738 | 0.9297 |
| ViT-B/16 | 0.9391 | 0.9356 | 0.9688 |

ViT 的 P@5 高 5.16 個百分點；mAP@5 高 0.0619。每張查詢取五張結果，平均相關結果約從 4.44 張增加至 4.70 張，表示結果清單較少混入其他類別。Hit@5 對應至少找到一張相關結果的查詢數，由 119/128 增至 124/128。

兩者最佳 validation loss 都出現在本次最後的第 10 epoch，曲線仍有改善空間，因此延長訓練可作為後續實驗，但不應直接宣稱一定提升 test 表現。影像 backbone 凍結，訓練的是投影層及分類 head；結果僅支持這個設定下 ViT 特徵較有效，不能推論所有資料上 ViT 都優於 CNN。

Recall@5 約 1.3% 並不代表檢索失敗：每個類別在 gallery 中有數百張相關影像，只取五張，能找回的相關總數比例本來就小。這題以 P@5、mAP@5 及結果圖較容易解讀。

## Q2：ViT+GRU 提高分類表現，主要減少錯配誤判

兩種模型使用完全相同的圖文配對。訓練 60000 pairs、validation 10000 pairs、test 10000 pairs；test 的 match/not_match 各 5000。

| 模型 | Accuracy | Macro-F1 | ROC-AUC | 錯誤數 |
| --- | --- | --- | --- | --- |
| ResNet18+GRU | 83.05% | 0.8299 | 0.9074 | 1695 |
| ViT-B/16+GRU | 87.43% | 0.8743 | 0.9446 | 1257 |

ViT 的 Accuracy 高 4.38 個百分點，錯誤減少 438 筆（相對減少 25.84%）。

| 錯誤類型 | ResNet18+GRU | ViT+GRU |
| --- | --- | --- |
| 不配對卻判成配對 | 1141 | 692 |
| 配對卻判成不配對 | 554 | 565 |

改善主要來自較能拒絕不相符的文字，第一種錯誤減少 449 筆；對真正相符配對的漏判則略增 11 筆。這比只說「Accuracy 較高」更能描述模型行為。

兩者第 2 epoch 之後 train loss 持續下降，但 validation loss 上升，顯示過擬合。early stopping 在第 5 epoch 停止，並讀回第 2 epoch 權重；因此沒有跑滿 10 epochs 是正常的模型選擇流程，並非漏跑。validation accuracy 稍升而 loss 變差，表示正確率沒有同幅改善，但部分錯誤預測變得更有信心。

負例從不同影像抽取，仍可能碰到語意相符的描述，所以這是弱標記圖文配對分類。這些分數不能直接視為任意圖文內容真實性的判斷能力。

## Q3：CLIP+GRU 在相同測試影像上優於第四週

三組結果對齊相同 1000 張 test 影像，確認每張的五句 references 一致。BLEU 採 0–1 corpus BLEU；BERTScore 統一使用 DistilBERT 第 5 層、無 IDF、無 baseline rescale。

| 模型／解碼 | BLEU-1 | BLEU-4 | BERTScore-F1 |
| --- | --- | --- | --- |
| Week 4 MobileNetV2+GRU／greedy | 0.5255 | 0.1459 | 0.8424 |
| Week 4 MobileNetV2+GRU／beam3 | 0.5572 | 0.1624 | 0.8454 |
| Week 5 CLIP+GRU／greedy | 0.6296 | 0.2066 | 0.8652 |

相較第四週 greedy，BLEU-4 增加 0.0607（相對提升 41.63%），BERTScore-F1 增加 0.0228。相較第四週 beam3，BLEU-4 增加 0.0441（相對提升 27.18%），BERTScore-F1 增加 0.0199。BERTScore 0.8652 不等於 86.52% 的句子完全正確。

實際範例中，兩隻狗在雪地玩耍生成 `two dogs playing in the snow`，狗游泳生成 `a dog swims in a pool`，有抓到主體、動作與場景。錯誤也仍存在：舞蹈照片生成了不符影像的 `yellow shirt`；單人美式足球照片生成了 `tackling` 及另一名球員，加入影像中沒有的互動。模型已能生成較相關的句子，但人物數量、服裝及動作細節仍會出錯。

validation loss 在第 4 epoch 最低；之後 train loss 繼續下降、validation loss 回升，故第 7 epoch early stop 並採第 4 epoch 權重。這同樣是正常完成訓練，並非少跑。teacher-forcing 的 token accuracy 不等於完整描述正確率。

Week 4 與 Week 5 的 encoder、decoder 實作及訓練設定均有差異，因此這是兩套完整系統的比較，無法把所有提升單獨歸因於 CLIP。

## 整體判讀

在本次設定下，Q1、Q2 可選 ViT 組合作為較佳實驗結果；Q3 的 CLIP+GRU 在 BLEU 與 BERTScore 均優於第四週兩種解碼。正式交件可使用現有結果，並附上上述限制與失敗範例。

目前為單一 seed 的實驗，尚無多次重跑的平均與變異，不能宣稱差異具有統計顯著性。端到端微調、hard negatives、更多 seeds 與 caption beam search 都是可選延伸，並非這次作業缺少的必跑項目。

訓練秒數僅計可訓練模組及 validation，不含下載、凍結 backbone 的影像特徵萃取及最終測試；例如 Q1 約 1–2 秒不能代表整個 ViT 流程只需要該時間。

完整圖表見 [正式實驗報告](report.md)，原始數據見各模型 `metrics.json`、`history.json` 與 `comparison/caption_comparison.json`。
