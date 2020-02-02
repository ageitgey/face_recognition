
# Face Recognition

_このファイルは [英語（オリジナル） in English](https://github.com/ageitgey/face_recognition/blob/master/README.md)、 [中国語 简体中文版](https://github.com/ageitgey/face_recognition/blob/master/README_Simplified_Chinese.md) 、 [韓国語 한국어](https://github.com/ageitgey/face_recognition/blob/master/README_Korean.md)で読むこともできます。_


世界で最もシンプルな顔認識ライブラリを使って、Pythonやコマンドラインで顔を認識・操作することができるライブラリです。

[dlib](http://dlib.net/)のディープラーニングを用いた最先端の顔認識を使用して構築されており、このモデルは[Labeled Faces in the Wild](http://vis-www.cs.umass.edu/lfw/)ベンチマークにて99.38%の正解率を記録しています。

シンプルな`face_recognition`コマンドラインツールも用意しており、コマンドラインでフォルダ内の画像を顔認識することもできます。

[![PyPI](https://img.shields.io/pypi/v/face_recognition.svg)](https://pypi.python.org/pypi/face_recognition)
[![Build Status](https://travis-ci.org/ageitgey/face_recognition.svg?branch=master)](https://travis-ci.org/ageitgey/face_recognition)
[![Documentation Status](https://readthedocs.org/projects/face-recognition/badge/?version=latest)](http://face-recognition.readthedocs.io/en/latest/?badge=latest)

## 特徴

#### 画像から顔を探す

画像に写っているすべての顔を探します。

![](https://cloud.githubusercontent.com/assets/896692/23625227/42c65360-025d-11e7-94ea-b12f28cb34b4.png)

```python
import face_recognition
image = face_recognition.load_image_file("your_file.jpg")
face_locations = face_recognition.face_locations(image)
```
#### 画像から顔の特徴を取得する

画像の中の顔から目、鼻、口、あごの場所と輪郭を得ることができます。

![](https://cloud.githubusercontent.com/assets/896692/23625282/7f2d79dc-025d-11e7-8728-d8924596f8fa.png)

```python
import face_recognition
image = face_recognition.load_image_file("your_file.jpg")
face_landmarks_list = face_recognition.face_landmarks(image)
```

顔の特徴を見つけることは多くの重要なことに役立ちますが、[デジタルメイクアップ](https://github.com/ageitgey/face_recognition/blob/master/examples/digital_makeup.py) のようにさほど重要ではないことにも使うことができます。

![](https://cloud.githubusercontent.com/assets/896692/23625283/80638760-025d-11e7-80a2-1d2779f7ccab.png)

#### 画像の中の顔を特定する

それぞれの画像に写っている人物を認識します。

![](https://cloud.githubusercontent.com/assets/896692/23625229/45e049b6-025d-11e7-89cc-8a71cf89e713.png)

```python
import face_recognition
known_image = face_recognition.load_image_file("biden.jpg")
unknown_image = face_recognition.load_image_file("unknown.jpg")

biden_encoding = face_recognition.face_encodings(known_image)[0]
unknown_encoding = face_recognition.face_encodings(unknown_image)[0]

results = face_recognition.compare_faces([biden_encoding], unknown_encoding)
```

他のPythonライブラリと一緒に用いてリアルタイムに顔認識することも可能です。

![](https://cloud.githubusercontent.com/assets/896692/24430398/36f0e3f0-13cb-11e7-8258-4d0c9ce1e419.gif)

試す場合は[こちらのサンプルコード](https://github.com/ageitgey/face_recognition/blob/master/examples/facerec_from_webcam_faster.py) を参照してください。

## デモ

ユーザーがコントリビュートした共有のJupyter notebookのデモがあります。（公式なサポートはありません）[![Deepnote](https://beta.deepnote.org/buttons/try-in-a-jupyter-notebook.svg)](https://beta.deepnote.org/launch?template=face_recognition)

## インストール

### 必要なもの

 * Python 3.3+ もしくは Python 2.7
 * macOS もしくは Linux (Windowsは公式にはサポートしていませんが動くかもしれません)

### インストールオプション：

#### MacもしくはLinuxにインストール

はじめに、dlibをインストールします。（Pythonの拡張機能も有効にします）

  * [macOSもしくはUbuntuにdlibをソースコードからインストールする方法](https://gist.github.com/ageitgey/629d75c1baac34dfa5ca2a1928a7aeaf)

次に、このモジュールをpypiから`pip3`（Python2の場合は`pip2`）を使ってインストールします。

```bash
pip3 install face_recognition
```

あるいは、[Docker](https://www.docker.com/)でこのライブラリを試すこともできます。詳しくは [こちらのセクション](#deployment)を参照してください。

もし、インストールが上手くいかない場合は、すでに用意されているVMイメージで試すこともできます。詳しくは[事前構成済みのVM](https://medium.com/@ageitgey/try-deep-learning-in-python-now-with-a-fully-pre-configured-vm-1d97d4c3e9b)を参照してください。(VMware Player もしくは VirtualBoxが対象)

#### Nvidia Jetson Nanoボードにインストール

 * [Jetson Nanoインストール手順](https://medium.com/@ageitgey/build-a-hardware-based-face-recognition-system-for-150-with-the-nvidia-jetson-nano-and-python-a25cb8c891fd)
   * この記事の手順通りにインストールを行ってください。現在、Jetson NanoのCUDAライブラリにはバグがあり、記事の手順通りにdlibの一行をコメントアウトし再コンパイルしないと失敗する恐れがあります。

#### Raspberry Pi 2+にインストール

  * [Raspberry Pi 2+インストール手順](https://gist.github.com/ageitgey/1ac8dbe8572f3f533df6269dab35df65)

#### Windowsにインストール

Windowsは公式サポートされていませんが、役立つインストール手順が投稿されています。

 * [@masoudr's Windows 10 インストールガイド (dlib + face_recognition)](https://github.com/ageitgey/face_recognition/issues/175#issue-257710508)

<!--

#### Installing a pre-configured Virtual Machine image

  * [Download the pre-configured VM image](https://medium.com/@ageitgey/try-deep-learning-in-python-now-with-a-fully-pre-configured-vm-1d97d4c3e9b) (for VMware Player or VirtualBox). -->

## 使用方法

### コマンドライン

`face_recognition`をインストールすると、2つのシンプルなコマンドラインがついてきます。

* `face_recognition` - 画像もしくはフォルダの中の複数の画像から顔を認識します

* `face_detection` - 画像もしくはフォルダの中の複数の画像から顔を検出します

#### `face_recognition` コマンドラインツール

`face_recognition` コマンドによって、画像もしくはフォルダの中の複数の画像から顔を認識することができます。

まずは、フォルダに知っている人たちの画像を一枚ずつ入れます。一人につき１枚の画像ファイルを用意し、画像のファイル名はその画像に写っている人物の名前にします。

![知っている人](https://cloud.githubusercontent.com/assets/896692/23582466/8324810e-00df-11e7-82cf-41515eba704d.png)

次に、2つ目のフォルダに特定したい画像を入れます。

![知らない人](https://cloud.githubusercontent.com/assets/896692/23582465/81f422f8-00df-11e7-8b0d-75364f641f58.png)

そして、`face_recognition`コマンドを実行し、知っている人の画像を入れたフォルダのパスと特定したい画像のフォルダ（もしくは画像ファイル）のパスを渡すと、それぞれの画像に誰がいるのかが分かります。

```bash
$ face_recognition ./pictures_of_people_i_know/ ./unknown_pictures/

/unknown_pictures/unknown.jpg,Barack Obama
/face_recognition_test/unknown_pictures/unknown.jpg,unknown_person
```

一つの顔につき一行が出力され、ファイル名と特定した人物の名前がカンマ区切りで表示されます。

`unknown_person`は知っている人の画像の中の誰ともマッチしなかった顔です。

#### `face_detection` コマンドラインツール

`face_detection` コマンドによって、画像の中にある顔の位置（ピクセル座標）を検出することができます。

`face_detection` コマンドを実行し、顔を検出したい画像を入れたフォルダ（もしくは画像ファイル）のパスを渡してあげるだけです。

```bash
$ face_detection  ./folder_with_pictures/

examples/image1.jpg,65,215,169,112
examples/image2.jpg,62,394,211,244
examples/image2.jpg,95,941,244,792
```

検出された顔一つにつき一行が出力され、顔の上・右・下・左の座標（ピクセル単位）が表示されます。

##### 許容誤差の調整 / 感度

もし同一人物に対して複数の一致があった場合、画像の中に写っている人たちの顔が非常に似ている可能性があるので、顔の比較をより厳しくするために許容誤差の値を下げる必要があります。

`--tolerance` コマンドによってそれが可能になります。デフォルトの許容誤差の値（tolerance value）を0.6よりも低くすると、より厳密に顔の比較をすることができます。

```bash
$ face_recognition --tolerance 0.54 ./pictures_of_people_i_know/ ./unknown_pictures/

/unknown_pictures/unknown.jpg,Barack Obama
/face_recognition_test/unknown_pictures/unknown.jpg,unknown_person
```

もし許容誤差の設定を調整するために一致した顔の距離値（face distance）を確認したい場合は `--show-distance true` を使ってください。

```bash
$ face_recognition --show-distance true ./pictures_of_people_i_know/ ./unknown_pictures/

/unknown_pictures/unknown.jpg,Barack Obama,0.378542298956785
/face_recognition_test/unknown_pictures/unknown.jpg,unknown_person,None
```

##### その他の例

ファイル名は出力せずに人物の名前だけを表示することもできます。

```bash
$ face_recognition ./pictures_of_people_i_know/ ./unknown_pictures/ | cut -d ',' -f2

Barack Obama
unknown_person
```

##### Face Recognition の高速化

マルチコア搭載コンピューターの場合は並列で実行することも可能です。例えば4CPUコアの場合、同じ時間で約4倍の画像を処理することができます。

Python 3.4 以上を使っている場合は`--cpus <number_of_cpu_cores_to_use>` パラメータを渡します。

```bash
$ face_recognition --cpus 4 ./pictures_of_people_i_know/ ./unknown_pictures/
```

`--cpus -1` のパラメータを渡すことで、システムのすべてのCPUコアを使うことも可能です。

#### Pythonモジュール

`face_recognition` モジュールをインポートすると、数行のコードでとても簡単に操作を行うことができます。

API Docs: [https://face-recognition.readthedocs.io](https://face-recognition.readthedocs.io/en/latest/face_recognition.html).

##### 自動的に画像の中のすべての顔を見つける

```python
import face_recognition

image = face_recognition.load_image_file("my_picture.jpg")
face_locations = face_recognition.face_locations(image)

# face_locations is now an array listing the co-ordinates of each face!
```

試す場合は[こちらのサンプルコード](https://github.com/ageitgey/face_recognition/blob/master/examples/find_faces_in_picture.py)を参照してください。

さらに正確でディープラーニングをもとにした顔検出モデルを選択することも可能です。

注意：このモデルで良いパフォーマンスを出すにはGPUアクセラレーション（NVidiaのCUDAライブラリ経由）が必要です。また、`dlib` をコンパイルする際にCUDAサポートを有効にする必要あります。

```python
import face_recognition

image = face_recognition.load_image_file("my_picture.jpg")
face_locations = face_recognition.face_locations(image, model="cnn")

# face_locations is now an array listing the co-ordinates of each face!
```

試す場合は[こちらのサンプルコード](https://github.com/ageitgey/face_recognition/blob/master/examples/find_faces_in_picture_cnn.py)を参照してください。

大量の画像をGPUを使って処理する場合は、[こちらのサンプルコード](https://github.com/ageitgey/face_recognition/blob/master/examples/find_faces_in_batches.py)のようにバッチ処理することも可能です。

##### 自動的に画像の中の顔特徴を見つける

```python
import face_recognition

image = face_recognition.load_image_file("my_picture.jpg")
face_landmarks_list = face_recognition.face_landmarks(image)

# face_landmarks_list is now an array with the locations of each facial feature in each face.
# face_landmarks_list[0]['left_eye'] would be the location and outline of the first person's left eye.
```

試す場合は[こちらのサンプルコード](https://github.com/ageitgey/face_recognition/blob/master/examples/find_facial_features_in_picture.py)を参照してください。

##### 画像の中の顔を認識し、その人物を特定する

```python
import face_recognition

picture_of_me = face_recognition.load_image_file("me.jpg")
my_face_encoding = face_recognition.face_encodings(picture_of_me)[0]

# my_face_encoding now contains a universal 'encoding' of my facial features that can be compared to any other picture of a face!

unknown_picture = face_recognition.load_image_file("unknown.jpg")
unknown_face_encoding = face_recognition.face_encodings(unknown_picture)[0]

# Now we can see the two face encodings are of the same person with `compare_faces`!

results = face_recognition.compare_faces([my_face_encoding], unknown_face_encoding)

if results[0] == True:
    print("It's a picture of me!")
else:
    print("It's not a picture of me!")
```

試す場合は[こちらのサンプルコード](https://github.com/ageitgey/face_recognition/blob/master/examples/recognize_faces_in_pictures.py)を参照してください。

## Pythonコードのサンプル

すべてのサンプルは[こちら](https://github.com/ageitgey/face_recognition/tree/master/examples)で見ることができます。

#### 顔検出

* [画像から顔を見つける](https://github.com/ageitgey/face_recognition/blob/master/examples/find_faces_in_picture.py)
* [画像から顔を見つける（ディープラーニングを使用する）](https://github.com/ageitgey/face_recognition/blob/master/examples/find_faces_in_picture_cnn.py)
* [大量の画像からGPUを用いて顔を見つける（ディープラーニングを使用する）](https://github.com/ageitgey/face_recognition/blob/master/examples/find_faces_in_batches.py)
* [WEBカメラによるライブ動画のすべての顔をぼかす(OpenCVのインストールが必要)](https://github.com/ageitgey/face_recognition/blob/master/examples/blur_faces_on_webcam.py)

#### 顔の特徴

* [画像から顔の特徴を特定する](https://github.com/ageitgey/face_recognition/blob/master/examples/find_facial_features_in_picture.py)
* [デジタルメイクアップを施す](https://github.com/ageitgey/face_recognition/blob/master/examples/digital_makeup.py)

#### 顔認識

* [知っている人の画像をもとに画像の中の知らない顔を発見する](https://github.com/ageitgey/face_recognition/blob/master/examples/recognize_faces_in_pictures.py)
* [画像の中の顔を四角で囲む](https://github.com/ageitgey/face_recognition/blob/master/examples/identify_and_draw_boxes_on_faces.py)
* [顔の距離値（face distance）によって比較する](https://github.com/ageitgey/face_recognition/blob/master/examples/face_distance.py)
* [WEBカメラによるライブ動画で顔認識する シンプル／低速バージョン (OpenCVのインストールが必要)](https://github.com/ageitgey/face_recognition/blob/master/examples/facerec_from_webcam.py)
* [WEBカメラによるライブ動画で顔認識する - 高速バージョン (OpenCVのインストールが必要)](https://github.com/ageitgey/face_recognition/blob/master/examples/facerec_from_webcam_faster.py)
* [動画ファイルを顔認識して新しいファイルに書き出す (OpenCVのインストールが必要)](https://github.com/ageitgey/face_recognition/blob/master/examples/facerec_from_video_file.py)
* [カメラ付きのRaspberry Piによって顔認識する](https://github.com/ageitgey/face_recognition/blob/master/examples/facerec_on_raspberry_pi.py)
* [顔認識ウェブサービスをHTTP経由で実行する(Flaskのインストールが必要)](https://github.com/ageitgey/face_recognition/blob/master/examples/web_service_example.py)
* [k近傍法で顔認識する](https://github.com/ageitgey/face_recognition/blob/master/examples/face_recognition_knn.py)
* [人物ごとに複数の画像をトレーニングし、SVM（サポートベクターマシン）を用いて顔認識する](https://github.com/ageitgey/face_recognition/blob/master/examples/face_recognition_svm.py)

## スタンドアロンの実行ファイルの作成

`python` や `face_recognition`のインストールをせずに実行することができるスタンドアロンの実行ファイルを作る場合は、[PyInstaller](https://github.com/pyinstaller/pyinstaller)を使います。しかし、このライブラリを使用するにはカスタム設定が必要です。

## `face_recognition`をカバーする記事とガイド

- 顔認識の仕組みについての記事: [ディープラーニングによる最新の顔認識](https://medium.com/@ageitgey/machine-learning-is-fun-part-4-modern-face-recognition-with-deep-learning-c3cffc121d78)
  - アルゴリズムとそれらがどのように動くかを取り上げています。
- Adrian Rosebrock氏の [OpenCV、Python、ディープラーニングによる顔認識](https://www.pyimagesearch.com/2018/06/18/face-recognition-with-opencv-python-and-deep-learning/)
  - 実際に顔認識を使用する方法について取り上げています。
- Adrian Rosebrock氏の [Raspberry Pi 顔認識](https://www.pyimagesearch.com/2018/06/25/raspberry-pi-face-recognition/)
  - Raspberry Piで使用する方法について取り上げています。
- Adrian Rosebrock氏の [Pythonによる顔のクラスタリング](https://www.pyimagesearch.com/2018/07/09/face-clustering-with-python/)
  - それぞれの画像に出現する人物に基づき、教師なし学習を用いて自動的に画像をクラスター化する方法について取り上げています。

## 顔認識の仕組み

ブラックボックスライブラリに依存せず、顔の位置や認識の仕組みを知りたい方は[こちらの記事](https://medium.com/@ageitgey/machine-learning-is-fun-part-4-modern-face-recognition-with-deep-learning-c3cffc121d78)を読んでください。

## 注意事項

* この顔認識モデルは大人でトレーニングされており、子どもではあまり上手く機能しません。比較する閾値をデフォルト（0.6）のままで使用すると子どもを混同しやすくなります。

* 精度は民族グループによって異なる可能性があります。詳しくは[こちらのwikiページ](https://github.com/ageitgey/face_recognition/wiki/Face-Recognition-Accuracy-Problems#question-face-recognition-works-well-with-european-individuals-but-overall-accuracy-is-lower-with-asian-individuals)を参照してください。

## <a name="deployment">クラウドにデプロイ (Heroku, AWSなど)</a>

`face_recognition`はC++で書かれた`dlib`に依存しているため、HerokuやAWSのようなクラウドサーバにこれらを使ったアプリをデプロイするのは難しい場合があります。

それを簡単にするために、このレポジトリには[Docker](https://www.docker.com/)コンテナ内で`face_recognition`のビルドされたアプリを実行する方法を示したサンプルDockerfileがあります。これによって、Dockerイメージをサポートしているすべてのサービスにデプロイできるようになるはずです。

コマンドを実行し、ローカルでDockerイメージを試すことができます。: `docker-compose up --build`

GPU (drivers >= 384.81) および [Nvidia-Docker](https://github.com/NVIDIA/nvidia-docker) がインストールされているLinuxユーザーはGPUでサンプルを実行することができます。[docker-compose.yml](docker-compose.yml) を開き、`dockerfile: Dockerfile.gpu`と`runtime: nvidia`の行をコメントアウトしてください。

## なにか問題が発生したら

もし問題が発生した場合はGitHubにIssueをあげる前に、まずはwikiの[よくあるエラー](https://github.com/ageitgey/face_recognition/wiki/Common-Errors)をお読みください

## 謝意

* dlibを作り、このライブラリで使っているトレーニングされた顔の特徴検出とフェイスエンコーディングモデルを提供してくれた[Davis King](https://github.com/davisking) ([@nulhom](https://twitter.com/nulhom))、本当にありがとうございます。
  フェイスエンコーディングを動かしているResNetについての情報は彼の[ブログ](http://blog.dlib.net/2017/02/high-quality-face-recognition-with-deep.html)を見てください。

* このようなライブラリがPythonで簡単に楽しくできるためのnumpy, scipy, scikit-image, pillow など全ての素晴らしいPythonデータサイエンスライブラリに取り組んでいる人たちに感謝しています。

* Pythonプロジェクトのパッケージングをより易しくする[Cookiecutter](https://github.com/audreyr/cookiecutter)と[audreyr/cookiecutter-pypackage](https://github.com/audreyr/cookiecutter-pypackage)に感謝しています。