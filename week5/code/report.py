from .common import read_json, save_csv


def table(rows, columns):
    lines = ['| ' + ' | '.join(label for _, label in columns) + ' |', '| ' + ' | '.join(['---'] * len(columns)) + ' |']
    for row in rows:
        values = []
        for key, _ in columns:
            value = row.get(key, '尚未執行')
            values.append('%.4f' % value if isinstance(value, float) else str(value))
        lines.append('| ' + ' | '.join(values) + ' |')
    return '\n'.join(lines)


def run(args):
    out = args.output_dir
    q1, q2, q3 = [], [], []
    text = ['# MMIP Week 5 實驗報告', '',
            '所有表格只讀取本次輸出資料夾中的實測檔案。未執行的項目不填造數值。', '',
            'Beans 以同疾病／健康類別視為相關影像；train 作為 gallery、test 作為 query。',
            'Flickr8k 沿用官方影像切分，詞彙表只在 train 建立。影像 backbone 凍結；投影層、文字 GRU、融合分類器或 decoder 有實際訓練。', '']
    for name, label in [('q1_cnn', 'ResNet18'), ('q1_vit', 'ViT-B/16'),
                        ('q2_cnn_gru', 'ResNet18 + GRU'), ('q2_vit_gru', 'ViT-B/16 + GRU'), ('q3_clip_gru', 'CLIP + GRU')]:
        path = out / name / 'metrics.json'
        if not path.exists():
            text += ['- %s：尚未完成訓練與評估。' % label]
            continue
        row = dict(read_json(path), model=label)
        (q1 if name.startswith('q1') else q2 if name.startswith('q2') else q3).append(row)
        if row.get('limit'):
            text += ['- %s：**除錯子集**，每 split 最多 %s 張，結果不可當作完整作業成績。' % (label, row['limit'])]
    text += ['', '## Q1 影像檢索', '']
    if q1:
        text += [table(q1, [('model', '模型'), ('precision_at_k', 'P@k'), ('recall_at_k', 'R@k'),
                 ('hit_at_k', 'Hit@k'), ('map_at_k', 'mAP@k'), ('k', 'k'), ('training_seconds', '訓練秒數')]), '']
        text += ['R@k 是找回的相關影像比例；Hit@k 是至少找到一張相關影像的 query 比例。mAP@k 分母為 min(相關總數, k)。', '']
        for kind in ['cnn', 'vit']:
            if (out / ('q1_' + kind) / 'retrieval_examples.png').exists():
                text += ['![%s retrieval](q1_%s/retrieval_examples.png)' % (kind, kind), '']
        if len(q1) == 2:
            winner = max(q1, key=lambda r: r['map_at_k'])
            text += ['本次 %s 的 mAP@k 較高（平手時取第一個模型）；此結果只適用於目前資料量與固定凍結 backbone 設定。' % winner['model'], '']
    text += ['## Q2 圖文融合分類', '']
    if q2:
        text += [table(q2, [('model', '模型'), ('accuracy', 'Accuracy'), ('macro_f1', 'Macro-F1'),
                 ('auc', 'ROC-AUC'), ('test_pairs', '測試 pairs'), ('training_seconds', '訓練秒數')]), '',
                 '負例為不同影像的描述，排除同影像與完全相同描述；仍可能偶然語意相符。這是弱標記配對實驗，不能推廣成任意圖文的語意正確率。', '']
        for kind in ['cnn', 'vit']:
            if (out / ('q2_' + kind + '_gru') / 'classification.png').exists():
                text += ['![%s classification](q2_%s_gru/classification.png)' % (kind, kind), '']
        if len(q2) == 2:
            winner = max(q2, key=lambda r: r['macro_f1'])
            text += ['在相同影像切分與正負配對下，%s 的 Macro-F1 較高（平手時取第一個模型）。' % winner['model'], '']
    text += ['## Q3 CLIP 影像描述', '']
    if q3:
        text += [table(q3, [('model', '模型'), ('bleu_1', 'BLEU-1'), ('bleu_2', 'BLEU-2'),
                  ('bleu_3', 'BLEU-3'), ('bleu_4', 'BLEU-4'), ('bertscore_f1', 'BERTScore-F1'), ('samples', '測試影像')]), '',
                  '![Caption examples](q3_clip_gru/caption_examples.png)', '']
    comparison = out / 'comparison' / 'caption_comparison.json'
    if comparison.exists():
        data = read_json(comparison)
        text += ['### 與 Week 4 比較', '',
                 '依影像 ID 對齊 %d 張，並檢查五句 references 一致；三種輸出以相同 BLEU/BERTScore 設定重新計算。' % len(data['image_ids']), '',
                 table(data['results'], [('model', '模型'), ('decoding', '解碼'), ('bleu_1', 'BLEU-1'), ('bleu_4', 'BLEU-4'),
                     ('bertscore_f1', 'BERTScore-F1'), ('samples', '共同影像')]), '',
                 '比較包含 encoder、decoder 實作與訓練設定的差異，不能將全部提升／下降歸因於 CLIP。', '']
    else:
        text += ['Week 4 比較尚未執行；Q3 完成後執行 `python main.py compare`。', '']
    for name in ['q1_cnn', 'q1_vit', 'q2_cnn_gru', 'q2_vit_gru', 'q3_clip_gru']:
        if (out / name / 'learning_curves.png').exists():
            text += ['### %s 學習曲線' % name, '', '![Learning curves](%s/learning_curves.png)' % name, '']
    text += ['## 評估設定與限制', '',
             '- 模型以最低 validation loss 選取；test 不參與訓練與 early stopping。',
             '- 訓練秒數只計可訓練模組的訓練／validation，排除下載、凍結影像特徵萃取與最終測試。',
             '- BLEU 是 0–1 的 corpus BLEU，NLTK smoothing method1，每張使用全部五句參考。',
             '- BERTScore 使用輸出 JSON 記錄的模型、層數與 hash；預設 DistilBERT 第 5 層、idf=False、無 baseline rescale。各模型使用同一設定，不能與其他設定的分數直接比較。',
             '- 固定 seed 有助重現，但不同 GPU／套件版本仍可能有微小浮點差異。', '']
    (out / 'report.md').write_text('\n'.join(text), encoding='utf-8')
    save_csv(out / 'model_comparison.csv', q1 + q2 + q3)
    print('Report:', out / 'report.md', flush=True)
