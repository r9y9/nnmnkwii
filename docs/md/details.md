# Design documentation details

## 0. Data representation

音声は、フレーム毎に特徴ベクトル（スペクトル包絡、メルケプストラム、F0, 非周期性指標等、コンテキスト）が抽出されることを基本的に想定する。声質変換は、フレーム単位で独立に変換する場合と、発話毎に変換する場合とにわけられる。前者の場合は、学習に用いる発話データをすべて結合して、NxDの特徴量ベクトルの配列（N:学習データの全フレーム数、D: 特徴ベクトルの次元）として表せれば十分である。一方で、後者を考えると、発話毎にデータを取得できるような構造が必要である。
本ライブラリでは、NxTxDの密配列（numpy.ndarray）として、発話の集合を表すこととする。音声の長さは発話によって異なるため、Tを十分大きな値とし、末尾にゼロ詰めを行うことで、固定長の配列として扱う。データセットに非常に長い音声があった場合に、この方法ではメモリ使用量が無駄に大きくなってしまうため、音声を複数の小さな発話にセグメンテーションする機能も提供する。

### Parallel data

パラレルデータを用いた声質変換の場合は、話者Aのデータセットを(N, T¹, D)のテンソル、話者Bのデータセットを(N, T², D)のテンソルとすると、パラレルデータを(N, max(T¹,T²), D)のテンソルとして表す。T¹ = T²として設定すると、都合がよい。

### Non parallel data

話者Aのデータセットを(N¹, T¹, D)のテンソル、話者Bのデータセットを(N², T². D)のテンソルとして、(N¹+N², max(T¹,T²), D) として表す。

## 1. Data source

音声データを (T, D) 配列に変換するクラスである。データセットを読み込むソースとして、例えば以下が挙げられる。

- Wav files
- Text files
- Precomputed acoustic/linguistic features

すべてのケースに柔軟に対応可能にするため、ソースの読み込みはユーザーの責任とする。具体的には、ユーザはDataSourceクラスを継承したクラスを作成し、以下のメソッドを実装しなければならない。

- **collect_files**: collect all files paired with speaker ids
- **process_file**: process a file to (T, D) array

## 2. Dataset

データセットは、データを保持するクラスである。データソースをattributeとして持ち、音声データを配列に変換する処理は、データソースが担う。DataSourceはユーザによって与えられる必要がある。

### BatchDataset

データセットをすべてメモリに読み込み、保持するクラスである。小さいデータセットを扱うのに適している。

### IncrementalDataset

データセットをインクリメンタルに必要に応じて読み込み、保持を行うクラスである。大規模なデータセットを扱うのに適している。

### To be addressed

- [ ] 特徴ベクトルの結合、分離
- [ ] CMUArctic
- [ ] FolderDataset
- [ ] キャッシュ機能

## 3. Preprocessing

長い音声ファイルを含むデータセットをナイーブに (N, T, D) 配列で表現すると、無駄に大きなメモリを使用することになる。本モジュールでは、長い音声ファイルを短いいくつかのサブセットにセグメンテーションする機能の提供をメインとする。また、無音区間削除など、デルタ特徴量の計算等、必要最低限の前処理も提供する。ただし、あらゆる前処理を提供しようとするわけではないことに注意する。前処理は、ユーザが用途に応じて実装することを想定する。

## 4. Parameter generation

音声特有である、静的・動的特徴量から、静的特徴量を生成するアルゴリズム Maximum Likelihood Parameter Generation (MLPG) を提供する。DNNにおける微分可能なロス関数として提供する。

## 5. Metrics

メルケプストラムなどの評価尺度を提供する

## 6. Baseline algorithms

- GMMベース声質変換
- DIFFGMMベース声質変換
- Highway Networks声質変換

## Module structure

- **datasets**: DataSource, Dataset
- **preprocess**: セグメンテーション、前処理
- **paramgen**: パラメータ生成、MLPG
- **metrics**: Evalation metrics
- **baseline**: Baseline algorithms
- **display**: 可視化パッケージ

## Plan

### Step 1: Data API stabilization

- [x] Data as 3D tensor
- [ ] Dataset, Datasource abstraction
- [x] CMUArctic dataset

### Step 2: Towards GMM / DNN voice conversion

- [x] paramgen: Correct MLPG implemntation
- [x] 一対一声質変換にフォーカスし、高品質な変換手法をbaselineに実装する
- [x] baseline: GMM声質変換

### Step 3: Add Evaluation utilities and refinement

- [ ] Add display module similar to librosa.display
- [ ] metrics
- [ ] GPU accerelated MLPG

### Step 4: Extend for DNN Statistical speech synthesis

Very hard!

- [ ] IncrementalDataLoader for large dataset
- [ ] Multiprocess for dataset iterating

Have fun!
