"""只讀取真實實驗輸出；尚未訓練的模型明確列為未完成。"""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def number(value):
    return '-' if value is None else f'{value:.4f}'


def table(headers, rows):
    return ['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |'] + [
        '| ' + ' | '.join(str(v) for v in row) + ' |' for row in rows]


def run(args):
    result = ROOT / 'results'
    result.mkdir(parents=True, exist_ok=True)
    lines = ['# MMIP 第四堂實驗結果', '',
             '指標只由已儲存的實驗結果產生。未執行訓練、測試或 Gemini 的部分不視為完成。', '',
             '## Q1 資料準備', '']
    summary_file = result / 'q1_data' / 'dataset_summary.json'
    if summary_file.exists():
        summary = json.loads(summary_file.read_text(encoding='utf-8'))
        lines += ['```json', json.dumps(summary, ensure_ascii=False, indent=2), '```', '',
                  '![資料範例](q1_data/dataset_examples.png)', '']
    else:
        lines += ['尚未執行 `python main.py prepare`。', '']
    all_rows = []
    for section, tasks in [('Q2 IMDb 情緒分類', ('q2_rnn', 'q2_lstm')),
                            ('Q3 Beans 分類', ('q3_vit', 'q3_resnet'))]:
        lines += ['## ' + section, '']
        rows = []
        for task in tasks:
            path = result / task / 'metrics.json'
            if not path.exists():
                rows.append([task, '未執行', '-', '-', '-'])
                continue
            metrics = json.loads(path.read_text(encoding='utf-8'))
            rows.append([task, number(metrics['accuracy']), number(metrics['macro_f1']),
                         number(metrics['macro_auc']), number(metrics['train_seconds'])])
            all_rows.append({'task': task, 'method': 'classification',
                             'accuracy': metrics['accuracy'], 'macro_f1': metrics['macro_f1'],
                             'macro_auc': metrics['macro_auc'], 'bleu_1': '', 'bleu_4': '',
                             'mean_total_ms': '', 'train_seconds': metrics['train_seconds']})
        lines += table(['模型', 'Accuracy', 'Macro-F1', 'Macro-AUC', '訓練秒數'], rows) + ['']
    lines += ['## Q4 Flickr8k 影像描述', '',
              '比較 MobileNetV2 + GRU/LSTM，兩者均使用 Greedy 與 Beam Search（width=3）。',
              'BLEU 為 0–1 的 corpus BLEU，使用每張影像的全部參考描述與 smoothing method1。',
              '時間為 warm batch=1：讀圖、resize、CNN 與 decoder，兩種解碼共享該張圖的 CNN 計時。', '']
    rows = []
    for task in ('q4_gru', 'q4_lstm'):
        path = result / task / 'metrics.json'
        if not path.exists():
            rows.append([task, '未執行', '-', '-', '-', '-', '-'])
            continue
        metrics = json.loads(path.read_text(encoding='utf-8'))
        for method in ('greedy', 'beam3'):
            values = metrics[method]
            rows.append([task, method, number(values['bleu_1']), number(values['bleu_2']),
                         number(values['bleu_3']), number(values['bleu_4']), number(values['mean_total_ms'])])
            all_rows.append({'task': task, 'method': method, 'accuracy': '', 'macro_f1': '', 'macro_auc': '',
                             'bleu_1': values['bleu_1'], 'bleu_4': values['bleu_4'],
                             'mean_total_ms': values['mean_total_ms'], 'train_seconds': metrics['train_seconds']})
        lines += [f"{task}：評估 {metrics['evaluated_images']} 張／test {metrics['test_split_images']} 張。", '']
    lines += table(['模型', '解碼', 'BLEU-1', 'BLEU-2', 'BLEU-3', 'BLEU-4', '平均回覆 ms'], rows) + ['']
    lines += ['## Gemini 描述符合度', '']
    gemini = result / 'q4_gemini' / 'summary.json'
    if gemini.exists():
        data = json.loads(gemini.read_text(encoding='utf-8'))
        lines += ['模型：' + data['gemini_model'], '']
        lines += table(['模型', '方法', '樣本數', '符合比例', '平均分數 / 5', '評審 API ms'], [
            [r['model'], r['method'], r['samples'], number(r['match_rate']), number(r['mean_score']),
             number(r['mean_judge_ms'])] for r in data['comparisons']])
        lines += ['', 'Gemini 為同一批影像的主觀評審；評審 API 時間與 caption 生成時間分開列出。', '']
    else:
        lines += ['尚未完成 Gemini 評估；設定 API Key 與模型名稱後執行 `python main.py gemini`。', '']
    lines += ['## 實驗解讀與交作業前檢查', '',
              '- 從 Q2 的 Accuracy、Macro-F1、AUC 比較 SimpleRNN 與 LSTM，搭配錯誤評論說明。',
              '- 從 Q3 的 Macro-AUC、ROC 與混淆矩陣比較 ViT、ResNet；記錄實際微調層與最佳驗證階段。',
              '- 從 Q4 的 BLEU 與回覆時間討論 GRU/LSTM、Greedy/Beam 的取捨，搭配影像與 Gemini 判斷。',
              '- 各模型必須使用相同 seed、資料範圍與測試影像；不同設定的結果須分開保存。',
              '- `--limit` 或 `--eval-images` 產生的子集結果只用來除錯，正式比較請評估完整測試集。',
              '- 個別實驗資料夾保存超參數與 predictions；檢查數字後自行撰寫實際觀察。', '']
    target = result / 'homework_report.md'
    target.write_text('\n'.join(lines), encoding='utf-8')
    if all_rows:
        with (result / 'comparison.csv').open('w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0]))
            writer.writeheader()
            writer.writerows(all_rows)
    print('Report:', target)
