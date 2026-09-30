# MMIP
## Q1
我使用現在研究正在用的資料，原本是影像資料及，所以我使用能從影像中找出的一些特徵
1. 黑區 / 有效影像範圍
zero_pct
影像中灰階值等於 0 的比例。通常代表全黑背景、扇形外區域、遮蔽區域。
dark_le5_pct
灰階值小於等於 5 的比例。比 zero_pct 更寬鬆，包含接近全黑的區域。
nonzero_pct
非 0 像素比例。可以理解成影像中不是純黑的區域比例。
roi_gt5_pct
ROI 內灰階值大於 5 的比例。代表 ROI 裡有效非黑影像的比例。
2. 全圖灰階統計
full_mean
整張影像平均亮度。越高代表整體越亮。
full_std
整張影像亮度標準差。越高代表亮暗變化越大。
full_entropy
整張影像灰階分布的 entropy。越高代表灰階分布越複雜、資訊量較多；越低代表灰階較集中或黑區較多。
full_snr
全圖 signal-to-noise ratio，通常是 mean / std。越高代表訊號相對變異較穩定。
full_cv
coefficient of variation，通常是 std / mean。越高代表亮度變異相對平均亮度更大。
full_enl
equivalent number of looks，常用於 speckle noise 評估。通常和 mean/std 有關，越高表示 speckle 相對較穩定、雜訊感較低。
3. ROI 灰階統計
roi_mean
ROI 區域平均亮度。比 full_mean 更接近肝實質亮度，較不受外部黑背景影響。
roi_std
ROI 區域亮度標準差。反映肝實質內亮度變化。
roi_entropy
ROI 內灰階 entropy。比 full_entropy 更適合討論肝臟實質紋理複雜度。
roi_snr
ROI 的 signal-to-noise ratio。
roi_cv
ROI 的 coefficient of variation。
roi_enl
ROI 的 equivalent number of looks，反映 ROI speckle 穩定度。
4. 邊緣 / 清晰度 / 紋理銳利度
sobel_mean_128roi
ROI 中 Sobel edge response 的平均值。越高代表邊緣或亮度梯度較明顯。
sobel_std_128roi
Sobel response 的標準差。越高代表局部邊緣強弱差異較大。
laplacian_var_128roi
Laplacian variance，常用於影像清晰度或 focus measure。越高通常代表影像更銳利、細節更多；越低可能較模糊。
gradient_mean_128roi
平均梯度強度。反映整體邊緣/亮度變化強度。
gradient_p95_128roi
梯度強度的第 95 百分位。可代表影像中較強邊緣的強度。
tenengrad_mean_128roi
Tenengrad focus measure，基於 Sobel 梯度。常用來衡量清晰度。
brenner_mean_128roi
Brenner focus measure。也是一種清晰度指標，對局部像素差異敏感。
canny_edge_density_128roi
Canny 偵測到的邊緣像素比例。越高代表邊緣結構較多。
5. 頻域特徵
fft_high_freq_ratio_r025_128roi
FFT 中高頻能量比例，半徑門檻約 0.25。高頻較多通常代表細節、speckle、雜訊或銳利紋理較多。
fft_high_freq_ratio_r040_128roi
更高頻區域的能量比例，門檻約 0.40。比 r025 更偏向細微紋理或高頻雜訊。
6. GLCM 紋理特徵
GLCM 是 gray-level co-occurrence matrix，描述鄰近像素灰階共同出現的關係，用來量化紋理。
glcm_contrast_128roi
紋理對比度。越高代表鄰近像素灰階差異較大，紋理較粗或對比更強。
glcm_dissimilarity_128roi
灰階不相似度。和 contrast 類似，但權重較線性。
glcm_homogeneity_128roi
同質性。越高代表鄰近像素灰階較相近，紋理更平滑。
glcm_ASM_128roi
angular second moment。越高代表紋理分布更規律、集中。
glcm_energy_128roi
ASM 的平方根。越高代表紋理較規律、灰階共現模式較集中。
glcm_correlation_128roi
鄰近像素灰階相關性。越高代表紋理結構有較強的線性關係。

原本使用邏輯回歸後續改用隨機森林，兩個模型結果是邏輯回歸的 Precision 0.6587而 隨機森林是0.5986  邏輯回歸的Recall 0.9022 和隨機森林是 0.9565 ，在目前各自選定的最佳 Threshold 下隨機森林比邏輯回歸更傾向將樣本預測為類別1，都是類別0錯的較多FP大
## Q2
使用tensorflow 
第1次訓練
EPOCHS:100 BATCH_SIZE:256 LR:1e-3 架構為64 32 1 Optimizer: Adam Loss: Binary Crossentropy
Final Training Loss   : 0.3861
Final Validation Loss : 0.4602
Minimum Validation Loss: 0.4374
Validation_Accuracy  : 0.8055
Validation_Precision : 0.5946
Validation_Recall    : 0.3791
Validation_F1-Score  : 0.4630
Validation ROC-AUC : 0.7597
第2次訓練
EPOCHS:100 BATCH_SIZE:256 LR:1e-3 架構為128 64 32 1中間加入L2 Regularization以及Dropout Optimizer: Adam Loss: Binary Crossentropy
Final Training Loss   : 0.4335
Final Validation Loss : 0.4464
Minimum Validation Loss: 0.4415
Validation_Accuracy  : 0.8157
Validation_Precision : 0.6308
Validation_Recall    : 0.4017
Validation_F1-Score  : 0.4908
Validation ROC-AUC : 0.7694
最終結果變好，從最後的訓練和驗證loss來看有效的解決了Overfitting主要有用的是L2 Regularization以及Dropout因為我把架構加深了照理來說最後的loss訓練和驗證會差異更大，看起來關閉Neuron以及限制Neuron的大小是有幫助的
# Q3
ROC Curve 與 AUC 代表的意義。 ROC Curve是把模型在不同閾值下的FPR TPR組成，X軸FPR Y軸TPR。AUC則是ROC曲線下面積 AUC可以看類別1和類別0的資料，模型預測能有多大的機率把類別1的機率排的比類別0的機率高。
進階題目我使用邏輯回歸，
Logistic Regression AUC = 0.7076
MLP AUC = 0.7694
從數字來看MLP的AUC較高代表說他能夠有較好的能區分正負樣本