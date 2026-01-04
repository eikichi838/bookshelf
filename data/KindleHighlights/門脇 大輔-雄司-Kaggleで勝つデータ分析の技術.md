---
kindle-sync:
  bookId: '51195'
  title: Kaggleで勝つデータ分析の技術
  author: '門脇 大輔, 阪田 隆司, 保坂 桂佑, and 平松 雄司'
  asin: B07YTDBC3Z
  lastAnnotatedDate: Invalid date
  bookImageUrl: 'https://m.media-amazon.com/images/I/71kBM0KZSAL._SY160.jpg'
  highlightsCount: 72
---
# Kaggleで勝つデータ分析の技術
## Metadata
* Author: [門脇 大輔, 阪田 隆司, 保坂 桂佑, and 平松 雄司](https://www.amazon.comundefined)
* ASIN: B07YTDBC3Z
* Reference: https://www.amazon.com/dp/B07YTDBC3Z
* [Kindle link](kindle://book?action=open&asin=B07YTDBC3Z)

## Highlights
https://github.com/ghmagazine/kagglebook — location: [83](kindle://book?action=open&asin=B07YTDBC3Z&location=83) ^ref-31321

---
from sklearn.metrics import mean_squared_error # y_trueが真の値、y_predが予測値 y_true = [1.0, 1.5, 2.0, 1.2, 1.8] y_pred = [0.8, 1.5, 1.8, 1.3, 3.0] rmse = np.sqrt(mean_squared_error(y_true, y_pred)) print(rmse) # 0.5532 — location: [1414](kindle://book?action=open&asin=B07YTDBC3Z&location=1414) ^ref-8455

---
分母は予測値に依らず、分子は二乗誤差を差し引いているため、この指標を最大化することはRMSEを最小化することと同じ意味です。 — location: [1444](kindle://book?action=open&asin=B07YTDBC3Z&location=1444) ^ref-40837

---
preds = 1.0 / (1.0 + np.exp(-preds)) # シグモイド関数 — location: [1762](kindle://book?action=open&asin=B07YTDBC3Z&location=1762) ^ref-3281

---
pred = 1.0 / (1.0 + np.exp(-pred_val)) — location: [1769](kindle://book?action=open&asin=B07YTDBC3Z&location=1769) ^ref-34207

ジグモイド関数

---
Nelder-Mead — location: [1805](kindle://book?action=open&asin=B07YTDBC3Z&location=1805) ^ref-2882

---
COBYLA — location: [1805](kindle://book?action=open&asin=B07YTDBC3Z&location=1805) ^ref-45436

---
1.　学習データをいくつかに分ける（ここでは4つに分け、それぞれfold1、fold2、fold3、fold4とする） 2.　fold2、fold3、fold4の真の値と予測確率から最適となる閾値を求め、その閾値でfold1のF1-scoreを計算する 3.　他のfoldについても同様に、自身以外のfoldの真の値と予測確率から最適となる閾値を求め、その閾値でF1-scoreを計算する（このように、各foldで自身の値を使わずに計算した閾値でのF1-scoreが評価できるのが、この方法のメリット） 4.　テストデータに適用する閾値は各foldの閾値の平均とする — location: [1829](kindle://book?action=open&asin=B07YTDBC3Z&location=1829) ^ref-41876

---
from scipy.optimize import minimize from sklearn.metrics import f1_score from sklearn.model_selection import KFold # サンプルデータ生成の準備 rand = np.random.RandomState(seed=71) train_y_prob = np.linspace(0, 1.0, 10000) # 真の値と予測値が以下のtrain_y, train_pred_probであったとする train_y = pd.Series(rand.uniform(0.0, 1.0, train_y_prob.size) < train_y_prob) train_pred_prob = np.clip(train_y_prob * np.exp(rand.standard_normal(train_y_prob.shape) * 0.3), 0.0, 1.0) # クロスバリデーションの枠組みで閾値を求める thresholds = [] scores_tr = [] scores_va = [] kf = KFold(n_splits=4, random_state=71, shuffle=True) for i, (tr_idx, va_idx) in enumerate(kf.split(train_pred_prob)): tr_pred_prob, va_pred_prob = train_pred_prob[tr_idx], train_pred_prob[va_idx] tr_y, va_y = train_y.ilock[tr_idx], train_y.ilock[va_idx] # 最適化の目的関数を設定 def f1_opt(x): return -f1_score(tr_y, tr_pred_prob >= x) # 学習データで閾値の最適化を行い、バリデーションデータで評価を行う result = minimize(f1_opt, x0=np.array([0.5]), method="Nelder-Mead") threshold = result['x'].item() score_tr = f1_score(tr_y, tr_pred_prob >= threshold) score_va = f1_score(va_y, va_pred_prob >= threshold) print(threshold, score_tr, score_va) thresholds.append(threshold) scores_tr.append(score_tr) scores_va.append(score_va) # 各foldの閾値の平均をテストデータには適用する threshold_test = np.mean(thresholds) print(threshold_test) — location: [1836](kindle://book?action=open&asin=B07YTDBC3Z&location=1836) ^ref-62137

---
# Fair 関数 def fair(preds, dtrain): x = preds - dtrain.get_labels() # 残差を取得 c = 1.0 # Fair関数のパラメータ den = abs(x) + c # 勾配の式の分母を計算 grad = c * x / den # 勾配 hess = c * c / den ** 2 # 二階微分値 return grad, hess # Pseudo-Huber関数 def psuedo_huber(preds, dtrain): d = preds - dtrain.get_labels() # 残差を取得 delta = 1.0 # Pseudo-Huber関数のパラメータ scale = 1 + (d / delta) ** 2 scale_sqrt = np.sqrt(scale) grad = d / scale_sqrt # 勾配 hess = 1 / scale / scale_sqrt # 二階微分値 return grad, hess — location: [1984](kindle://book?action=open&asin=B07YTDBC3Z&location=1984) ^ref-12492

---
label encodingで置き換えた数値がそのまま演算に使われてしまうため、カテゴリ変数の変換はlabel encodingよりはone-hot encodingの方が良いでしょう。 — location: [2169](kindle://book?action=open&asin=B07YTDBC3Z&location=2169) ^ref-2392

---
GBDTを使う場合には、欠損値のまま取り扱うというのが基本的な選択となります。 — location: [2216](kindle://book?action=open&asin=B07YTDBC3Z&location=2216) ^ref-57754

---
欠損値のまま扱う場合と埋める場合の両方を試してみるのも良いでしょう。 — location: [2220](kindle://book?action=open&asin=B07YTDBC3Z&location=2220) ^ref-6348

---
Bayesian averageという方法があります。 — location: [2246](kindle://book?action=open&asin=B07YTDBC3Z&location=2246) ^ref-6765

---
この際、補完のためのモデルの特徴量に本来の目的変数を含めてしまうと、テストデータについて補完ができなくなりますので注意しましょう。 — location: [2261](kindle://book?action=open&asin=B07YTDBC3Z&location=2261) ^ref-27662

---
最初の段階で変数の分布をヒストグラムなどで見て、欠損として認識すべき値がないかを確認しておくことが望ましいでしょう。 — location: [2281](kindle://book?action=open&asin=B07YTDBC3Z&location=2281) ^ref-50749

---
scikit-learnのpreprocessingモジュールのStandardScalerクラスで標準化を行うことができます。 — location: [2321](kindle://book?action=open&asin=B07YTDBC3Z&location=2321) ^ref-10360

---
from sklearn.preprocessing import StandardScaler # 学習データに基づいて複数列の標準化を定義 scaler = StandardScaler() scaler.fit(train_x[num_cols]) # 変換後のデータで各列を置換 train_x[num_cols] = scaler.transform(train_x[num_cols]) test_x[num_cols] = scaler.transform(test_x[num_cols]) — location: [2324](kindle://book?action=open&asin=B07YTDBC3Z&location=2324) ^ref-51070

---
学習データとテストデータは同じ変換を行う必要があり、学習データとテストデータで、それぞれ別の変換を行うことは避けるべきです。 — location: [2357](kindle://book?action=open&asin=B07YTDBC3Z&location=2357) ^ref-57450

---
0が値として含まれる場合にはそのまま対数をとることができないので、log(x+1)による変換がよく使われます。 — location: [2385](kindle://book?action=open&asin=B07YTDBC3Z&location=2385) ^ref-23120

---
x = np.array([1.0, 10.0, 100.0, 1000.0, 10000.0]) # 単に対数をとる x1 = np.log(x) # 1を加えたあとに対数をとる x2 = np.log1p(x) # 絶対値の対数をとってから元の符号を付加する x3 = np.sign(x) * np.log(np.abs(x)) — location: [2389](kindle://book?action=open&asin=B07YTDBC3Z&location=2389) ^ref-27320

---
x = [1, 7, 5, 4, 6, 3] # pandasのcut関数でbinningを行う # binの数を指定する場合 binned = pd.cut(x, 3, labels=False) print(binned) # [0 2 1 1 2 0] - 変換された値は3つのbinのどれに入ったかを表す # binの範囲を指定する場合（3.0以下、3.0より大きく5.0以下、5.0より大きい） bin_edges = [-float('inf'), 3.0, 5.0, float('inf')] binned = pd.cut(x, bin_edges, labels=False) print(binned) # [0 2 1 1 2 0] - 変換された値は3つのbinのどれに入ったかを表す — location: [2443](kindle://book?action=open&asin=B07YTDBC3Z&location=2443) ^ref-2749

---
x = [10, 20, 30, 0, 40, 40] # pandasのrank関数で順位に変換する rank = pd.Series(x).rank() print(rank.values) # はじまりが1、同順位があった場合は平均の順位となる # [2. 3. 4. 1. 5.5 5.5] # numpyのargsort関数を2回適用する方法で順位に変換する order = np.argsort(x) rank = np.argsort(order) print(rank) # はじまりが0、同順位があった場合はどちらかが上位となる # [1 2 3 0 4 5] — location: [2461](kindle://book?action=open&asin=B07YTDBC3Z&location=2461) ^ref-48693

---
RankGauss — location: [2466](kindle://book?action=open&asin=B07YTDBC3Z&location=2466) ^ref-57298

---
scikit-learnのpreprocessingモジュールのQuantileTransformerクラスにおいて、n_quantilesを十分大きくした上でoutput_distribution='normal'を指定すると、この変換を行うことができます。 — location: [2471](kindle://book?action=open&asin=B07YTDBC3Z&location=2471) ^ref-34449

---
from sklearn.preprocessing import QuantileTransformer # 学習データに基づいて複数列のRankGaussによる変換を定義 transformer = QuantileTransformer(n_quantiles=100, random_state=0, output_distribution='normal') transformer.fit(train_x[num_cols]) # 変換後のデータで各列を置換 train_x[num_cols] = transformer.transform(train_x[num_cols]) test_x[num_cols] = transformer.transform(test_x[num_cols]) — location: [2474](kindle://book?action=open&asin=B07YTDBC3Z&location=2474) ^ref-40792

---
テストデータにのみ存在する水準 注8 がある場合、 — location: [2496](kindle://book?action=open&asin=B07YTDBC3Z&location=2496) ^ref-4853

---
もしある場合には以下のいずれかの対応をとると良いでしょ — location: [2500](kindle://book?action=open&asin=B07YTDBC3Z&location=2500) ^ref-1319

---
target encoding — location: [2512](kindle://book?action=open&asin=B07YTDBC3Z&location=2512) ^ref-39923

---
one-hot encoding — location: [2514](kindle://book?action=open&asin=B07YTDBC3Z&location=2514) ^ref-26151

---
# 学習データとテストデータを結合してget_dummiesによるone-hot encodingを行う all_x = pd.concat([train_x, test_x]) all_x = pd.get_dummies(all_x, columns=cat_cols) # 学習データとテストデータに再分割 train_x = all_x.iloc[:train_x.shape[0], :].reset_index(drop=True) test_x = all_x.iloc[train_x.shape[0]:, :].reset_index(drop=True) — location: [2526](kindle://book?action=open&asin=B07YTDBC3Z&location=2526) ^ref-29347

---
変数をループしてfrequency encoding for c in cat_cols: freq = train_x[c].value_counts() # カテゴリの出現回数で置換 train_x[c] = train_x[c].map(freq) test_x[c] = test_x[c].map(freq) — location: [2599](kindle://book?action=open&asin=B07YTDBC3Z&location=2599) ^ref-39856

---
from sklearn.model_selection import KFold # 変数をループしてtarget encoding for c in cat_cols: # 学習データ全体で各カテゴリにおけるtargetの平均を計算 data_tmp = pd.DataFrame({c: train_x[c], 'target': train_y}) target_mean = data_tmp.groupby(c)['target'].mean() # テストデータのカテゴリを置換 test_x[c] = test_x[c].map(target_mean) # 学習データの変換後の値を格納する配列を準備 tmp = np.repeat(np.nan, train_x.shape[0]) # 学習データを分割 kf = KFold(n_splits=4, shuffle=True, random_state=72) for idx_1, idx_2 in kf.split(train_x): # out-of-foldで各カテゴリにおける目的変数の平均を計算 target_mean = data_tmp.iloc[idx_1].groupby(c)['target'].mean() # 変換後の値を一時配列に格納 tmp[idx_2] = train_x[c].iloc[idx_2].map(target_mean) # 変換後のデータで元の変数を置換 train_x[c] = tmp — location: [2620](kindle://book?action=open&asin=B07YTDBC3Z&location=2620) ^ref-29928

---
from sklearn.model_selection import KFold # クロスバリデーションのfoldごとにtarget encodingをやり直す kf = KFold(n_splits=4, shuffle=True, random_state=71) for i, (tr_idx, va_idx) in enumerate(kf.split(train_x)): # 学習データからバリデーションデータを分ける tr_x, va_x = train_x.iloc[tr_idx].copy(), train_x.iloc[va_idx].copy() tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx] # 変数をループしてtarget encoding for c in cat_cols: # 学習データ全体で各カテゴリにおけるtargetの平均を計算 data_tmp = pd.DataFrame({c: tr_x[c], 'target': tr_y}) target_mean = data_tmp.groupby(c)['target'].mean() # バリデーションデータのカテゴリを置換 va_x.loc[:, c] = va_x[c].map(target_mean) # 学習データの変換後の値を格納する配列を準備 tmp = np.repeat(np.nan, tr_x.shape[0]) kf_encoding = KFold(n_splits=4, shuffle=True, random_state=72) for idx_1, idx_2 in kf_encoding.split(tr_x): # out-of-foldで各カテゴリにおける目的変数の平均を計算 target_mean = data_tmp.iloc[idx_1].groupby(c)['target'].mean() # 変換後の値を一時配列に格納 tmp[idx_2] = tr_x[c].iloc[idx_2].map(target_mean) tr_x.loc[:, c] = tmp # 必要に応じてencodeされた特徴量を保存し、あとで読み込めるようにしておく — location: [2640](kindle://book?action=open&asin=B07YTDBC3Z&location=2640) ^ref-7729

---
多クラス分類の場合は、クラスの数だけ二値分類があると考えて、クラスの数だけtarget encodingによる特徴量を作る — location: [2676](kindle://book?action=open&asin=B07YTDBC3Z&location=2676) ^ref-34930

---
上、4～10程度を推奨します。 — location: [2712](kindle://book?action=open&asin=B07YTDBC3Z&location=2712) ^ref-8613

---
embedding — location: [2716](kindle://book?action=open&asin=B07YTDBC3Z&location=2716) ^ref-5538

---
ABC-00123 や XYZ-00200 のような型番の場合、前半の英字3文字と、後半の数字5文字に分割する — location: [2744](kindle://book?action=open&asin=B07YTDBC3Z&location=2744) ^ref-31584

---
● 　 3、 E のように数字のものと英字のものが混じっている場合、数字か否かを特徴量にする ● 　 AB、 ACE、 BCDE のように文字数に違いがある場合、文字数を特徴量にする — location: [2745](kindle://book?action=open&asin=B07YTDBC3Z&location=2745) ^ref-34579

---
年の特徴量を単純に加える ● 　年の特徴量を加えるが、テストデータにのみ存在する年を学習データの最新の年に置換する ● 　年の特徴量をあえて含めない ● 　年や月の情報を使って学習データとして使う期間を制限する — location: [2832](kindle://book?action=open&asin=B07YTDBC3Z&location=2832) ^ref-63345

---
# 図の形式のデータフレームがあるとする # train : 学習データ（ユーザID, 商品ID, 目的変数などの列がある） # product_master: 商品マスタ（商品IDと商品の情報を表す列がある） # user_log : ユーザの行動のログデータ（ユーザIDと各行動の情報を表す列がある） # 商品マスタを学習データと結合する train = train.merge(product_master, on='product_id', how='left') # ログデータのユーザごとの行数を集計し、学習データと結合する user_log_agg = user_log.groupby('user_id').size().reset_index().rename(columns={0: 'user_count'}) train = train.merge(user_log_agg, on='user_id', how='left') — location: [2952](kindle://book?action=open&asin=B07YTDBC3Z&location=2952) ^ref-22869

---
# ワイドフォーマットのデータを読み込む df_wide = pd.read_csv('../input/ch03/time_series_wide.csv', index_col=0) # インデックスの型を日付型に変更する df_wide.index = pd.to_datetime(df_wide.index) print(df_wide.iloc[:5, :3]) ''' A B C date 2016-07-01 532 3314 1136 2016-07-02 798 2461 1188 2016-07-03 823 3522 1711 2016-07-04 937 5451 1977 2016-07-05 881 4729 1975 ''' # ロングフォーマットに変換する df_long = df_wide.stack().reset_index(1) df_long.columns = ['id', 'value'] print(df_long.head(10)) ''' id value date 2016-07-01 A 532 2016-07-01 B 3314 2016-07-01 C 1136 2016-07-02 A 798 2016-07-02 B 2461 2016-07-02 C 1188 2016-07-03 A 823 2016-07-03 B 3522 2016-07-03 C 1711 2016-07-04 A 937 ... ''' # ワイドフォーマットに戻す df_wide = df_long.pivot(index=None, columns='id', values='value') — location: [3161](kindle://book?action=open&asin=B07YTDBC3Z&location=3161) ^ref-15922

---
# 7期前, 14期前, 21期前, 28期前の値の平均 x_e7_avg = (x.shift(7) + x.shift(14) + x.shift(21) + x.shift(28)) / 4.0 — location: [3207](kindle://book?action=open&asin=B07YTDBC3Z&location=3207) ^ref-21941

---
# train_xは学習データで、ユーザID, 日付を列として持つDataFrameとする # event_historyは、過去に開催したイベントの情報で、日付、イベントを列として持つDataFrameとする # occurrencesは、日付、セールが開催されたか否かを列として持つDataFrameとなる dates = np.sort(train_x['date'].unique()) occurrences = pd.DataFrame(dates, columns=['date']) sale_history = event_history[event_history['event'] == 'sale'] occurrences['sale'] = occurrences['date'].isin(sale_history['date']) # 累積和をとることで、それぞれの日付での累積出現回数を表すようにする # occurrencesは、日付、セールの累積出現回数を列として持つDataFrameとなる occurrences['sale'] = occurrences['sale'].cumsum() # 日付をキーとして学習データと結合する train_x = train_x.merge(occurrences, on='date', how='left') — location: [3243](kindle://book?action=open&asin=B07YTDBC3Z&location=3243) ^ref-10091

---
テストデータのレコードの時点ごとに、使える過去の期間が異なる場合です。 — location: [3263](kindle://book?action=open&asin=B07YTDBC3Z&location=3263) ^ref-37071

---
です。 — location: [3321](kindle://book?action=open&asin=B07YTDBC3Z&location=3321) ^ref-6642

---
from sklearn.decomposition import PCA # データは標準化などのスケールを揃える前処理が行われているものと — location: [3323](kindle://book?action=open&asin=B07YTDBC3Z&location=3323) ^ref-50451

---
する — location: [3324](kindle://book?action=open&asin=B07YTDBC3Z&location=3324) ^ref-33264

---
線形判別分析（ — location: [3360](kindle://book?action=open&asin=B07YTDBC3Z&location=3360) ^ref-52967

---
線形判別分析（Linear Discriminant — location: [3361](kindle://book?action=open&asin=B07YTDBC3Z&location=3361) ^ref-16527

---
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA # データは標準化などのスケールを揃える前処理が行われているものとする # 学習データに基づいてLDAによる変換を定義 lda = LDA(n_components=1) lda.fit(train_x, train_y) # 変換の適用 train_x = lda.transform(train_x) test_x = lda.transform(test_x) — location: [3367](kindle://book?action=open&asin=B07YTDBC3Z&location=3367) ^ref-46335

---
scikit-learnのmanifoldモジュールにTSNEがあるのですが、2019年8月時点での実装では遅く、python-bhtsne 注28（pipインストールを行うときのパッケージ名はbhtsne）を使用した方が良いでしょう 注29。 — location: [3375](kindle://book?action=open&asin=B07YTDBC3Z&location=3375) ^ref-23190

---
対数化してから平均をとることの狙いは、祝日など例外的に来客数が多い日があったときにその影響を緩和することです。 — location: [3747](kindle://book?action=open&asin=B07YTDBC3Z&location=3747) ^ref-39237

---
例えば、2017/5/2は火曜日ですが、ゴールデンウィーク直前であることから、通常時の金曜日と似た傾向があるだろうことを予想し、あえて金曜日とみなした上で特徴量を作成し、学習しました。同様に、5/3～5/5は土曜日とみなして特徴量を作成し、学習しました。 — location: [3760](kindle://book?action=open&asin=B07YTDBC3Z&location=3760) ^ref-46606

---
このコンペでは、前月に購入していない顧客が新たに購入することの予測が求められていたため、むしろ0→1に遷移する確率が本質となります。そのような考えから、筆者は単純にフラグの1の数や割合を集計するのではなく、フラグ値の変化に注目した変数を作成しました。 — location: [3786](kindle://book?action=open&asin=B07YTDBC3Z&location=3786) ^ref-7362

---
● 　対象範囲：すべての期間、特定の曜日・時間帯・期間、特定のイベントのみなど ● 　集約方法：履修登録単位、ユーザ単位、コース単位など ● 　指標：ログ数、アクセス — location: [3842](kindle://book?action=open&asin=B07YTDBC3Z&location=3842) ^ref-22901

---
日数など — location: [3845](kindle://book?action=open&asin=B07YTDBC3Z&location=3845) ^ref-20592

---
ユーザの「真面目さ」を表現する： — location: [3850](kindle://book?action=open&asin=B07YTDBC3Z&location=3850) ^ref-24668

---
カーブフィッティング — location: [3873](kindle://book?action=open&asin=B07YTDBC3Z&location=3873) ^ref-44112

---
from sklearn.metrics import log_loss from sklearn.model_selection import KFold # 学習データ・バリデーションデータを分けるためのインデックスを作成する # 学習データを4つに分割し、うち1つをバリデーションデータとする kf = KFold(n_splits=4, shuffle=True, random_state=71) tr_idx, va_idx = list(kf.split(train_x))[0] # 学習データを学習データとバリデーションデータに分ける tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx] tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx] # モデルを定義する model = Model(params) # 学習データに対してモデルを学習させる # モデルによっては、バリデーションデータを同時に与えてスコアをモニタリングすることができる model.fit(tr_x, tr_y) # バリデーションデータに対して予測し、評価を行う va_pred = model.predict(va_x) score = log_loss(va_y, va_pred) print(f'logloss: {score:.4f}') — location: [3968](kindle://book?action=open&asin=B07YTDBC3Z&location=3968) ^ref-32685

---
from sklearn.metrics import log_loss from sklearn.model_selection import KFold # 学習データを4つに分け、うち1つをバリデーションデータとする # どれをバリデーションデータとするかを変えて学習・評価を4回行う scores = [] kf = KFold(n_splits=4, shuffle=True, random_state=71) for tr_idx, va_idx in kf.split(train_x): tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx] tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx] model = Model(params) model.fit(tr_x, tr_y) va_pred = model.predict(va_x) score = log_loss(va_y, va_pred) scores.append(score) # クロスバリデーションの平均のスコアを出力する print(f'logloss: {np.mean(scores):.4f}') — location: [3987](kindle://book?action=open&asin=B07YTDBC3Z&location=3987) ^ref-64788

---
特徴量作成が最も重要で、分析コンペの半分から8割の作業時間は特徴量作成に費やす ● 　ハイパーパラメータは、変更したときにどのくらい影響があるか時折見ながら、本格的な調整は終盤に行う ● 　モデルはGBDTでまずは進めていき、タスクの性質によってニューラルネットを検討したり、アンサンブルを考える場合には他のモデルも作成する ● 　データやタスクの理解が進むとともに、バリデーションの枠組みを変更することもある — location: [4008](kindle://book?action=open&asin=B07YTDBC3Z&location=4008) ^ref-48829

---
学習時にモデルが複雑なときに罰則を科すこと — location: [4058](kindle://book?action=open&asin=B07YTDBC3Z&location=4058) ^ref-63298

---
同じ種類のモデルを並列に複数作成し、それらの予測値の平均などを用いて予測します。 — location: [4082](kindle://book?action=open&asin=B07YTDBC3Z&location=4082) ^ref-28893

---
勾配ブースティング木（GBDT） ● 　ニューラルネット ● 　線形モデル ● 　その他のモデル ■ 　k近傍法（k-nearest neighbor algorithm, kNN） ■ 　ランダムフォレスト（Random Forest, RF） ■ 　Extremely Randomized Trees（ERT） ■ 　Regularized Greedy Forest（RGF） ■ 　Field-aware Factorization Machines（FFM） — location: [4100](kindle://book?action=open&asin=B07YTDBC3Z&location=4100) ^ref-22094

---
GBDTは精度・計算速度・使いやすさともに優れているため、通常まず最初に作られるモデルです。 — location: [4113](kindle://book?action=open&asin=B07YTDBC3Z&location=4113) ^ref-24296

---
教師あり学習のモデルとしてよく紹介されるモデルのうち、サポートベクターマシンは精度や計算速度が見劣りするため、あまり使われることはありません。 — location: [4120](kindle://book?action=open&asin=B07YTDBC3Z&location=4120) ^ref-23615

---
import xgboost as xgb from sklearn.metrics import log_loss # 特徴量と目的変数をxgboostのデータ構造に変換する dtrain = xgb.DMatrix(tr_x, label=tr_y) dvalid = xgb.DMatrix(va_x, label=va_y) dtest = xgb.DMatrix(test_x) # ハイパーパラメータの設定 params = {'objective': 'binary:logistic', 'silent': 1, 'random_state': 71} num_round = 50 # 学習の実行 # バリデーションデータもモデルに渡し、学習の進行とともにスコアがどう変わるかモニタリングする # watchlistには学習データおよびバリデーションデータをセットする watchlist = [(dtrain, 'train'), (dvalid, 'eval')] model = xgb.train(params, dtrain, num_round, evals=watchlist) # バリデーションデータでのスコアの確認 va_pred = model.predict(dvalid) score = log_loss(va_y, va_pred) print(f'logloss: {score:.4f}') # 予測（二値の予測値ではなく、1である確率を出力するようにしている） pred = model.predict(dtest) — location: [4190](kindle://book?action=open&asin=B07YTDBC3Z&location=4190) ^ref-61763

---
予測時にntree_limitパラメータを設定しないと、最適ではなく学習が止まったところまでの木の本数で計算されるので注意が必要です。 — location: [4225](kindle://book?action=open&asin=B07YTDBC3Z&location=4225) ^ref-38012

---
colsample_bytreeで指定します。また、それぞれの決定木を作るときに、学習データの行もサンプリングします。その割合はパラメータsubsampleで指定します。 — location: [4368](kindle://book?action=open&asin=B07YTDBC3Z&location=4368) ^ref-44491

---
Kaggleの「Two Sigma Financial Modeling Challenge」や「Walmart Recruiting II: Sales in Stormy Weather」 — location: [4629](kindle://book?action=open&asin=B07YTDBC3Z&location=4629) ^ref-7009

---
1正則化を行う線形回帰モデルのことをLasso、L2正則化を行う線形回帰モデルのことをRidgeと言います（ — location: [4632](kindle://book?action=open&asin=B07YTDBC3Z&location=4632) ^ref-7889

---
