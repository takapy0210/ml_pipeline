# 概要

タイタニックのデータを用いて、LightGBMとxgboostによる学習・推論の一連の流れが行えるようにしています。

## 動作検証済み環境
OS: MacOS Catalina
python: 3.7.2

# 手順

## クローン
```sh
git clone https://github.com/takapy0210/ml_pipeline.git
```

## フォルダ移動
```sh
cd ml_pipeline/code
```

## 特徴量生成
```sh
python 10_titanic_fe.py
```

### 生成された特徴量の確認（確認したい場合）
```sh
python 15_show_all_features.py
```

## 学習
```sh
python 20_run.py
```
