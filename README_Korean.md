# Face Recognition

본 문서는 _[중국어 简体中文版](https://github.com/ageitgey/face_recognition/blob/master/README_Simplified_Chinese.md) 로부터 번역되어 한국 사용자들의 기여를 통해 만들어진 문서입니다. 

본 라이브러리는 세계에서 가장 간단한 얼굴 인식 라이브러리로, Python 또는 명령 줄(CLI)에서 얼굴을 인식하고 조작해 볼 수 있습니다.

본 라이브러리는 딥러닝 기반으로 제작된 [dlib](http://dlib.net/)의 최첨단 얼굴 인식 기능을 사용하여 구축되었습니다. 이 모델은 [Labeled Faces in the Wild](http://vis-www.cs.umass.edu/lfw/) 기준으로 99.38%의 정확도를 가집니다.

또한, 명령 줄(CLI)에서 이미지 폴더 안에 있는 얼굴 인식 기능을 위한 간단한 `face_recognition` 도구를 제공합니다!


[![PyPI](https://img.shields.io/pypi/v/face_recognition.svg)](https://pypi.python.org/pypi/face_recognition)
[![Build Status](https://travis-ci.org/ageitgey/face_recognition.svg?branch=master)](https://travis-ci.org/ageitgey/face_recognition)
[![Documentation Status](https://readthedocs.org/projects/face-recognition/badge/?version=latest)](http://face-recognition.readthedocs.io/en/latest/?badge=latest)

## 특징

#### 사진에서 얼굴 찾기

사진에 등장하는 모든 얼굴들을 찾습니다:

![](https://cloud.githubusercontent.com/assets/896692/23625227/42c65360-025d-11e7-94ea-b12f28cb34b4.png)

```python
import face_recognition
image = face_recognition.load_image_file("your_file.jpg")
face_locations = face_recognition.face_locations(image)
```

#### 사진에 있는 얼굴의 특징을 찾기&조작하기

각각의 사람의 눈, 코, 입, 턱의 위치와 윤곽을 잡아냅니다.

![](https://cloud.githubusercontent.com/assets/896692/23625282/7f2d79dc-025d-11e7-8728-d8924596f8fa.png)

```python
import face_recognition
image = face_recognition.load_image_file("your_file.jpg")
face_landmarks_list = face_recognition.face_landmarks(image)
```

얼굴의 특징을 찾는 기능은 여러 중요한 일들에 유용하게 쓰입니다. 예를 들어 [디지털 메이크업](https://github.com/ageitgey/face_recognition/blob/master/examples/digital_makeup.py) (Meitu 같은 것)을 적용하는 것과 같은 정말 멍청한 것들에도 쓰일 수 있습니다:

![](https://cloud.githubusercontent.com/assets/896692/23625283/80638760-025d-11e7-80a2-1d2779f7ccab.png)

#### 사진 속 얼굴의 신원 확인하기 

각각의 사진에서 누가 등장하였는지 인식합니다.

![](https://cloud.githubusercontent.com/assets/896692/23625229/45e049b6-025d-11e7-89cc-8a71cf89e713.png)

```python
import face_recognition
known_image = face_recognition.load_image_file("biden.jpg")
unknown_image = face_recognition.load_image_file("unknown.jpg")

biden_encoding = face_recognition.face_encodings(known_image)[0]
unknown_encoding = face_recognition.face_encodings(unknown_image)[0]

results = face_recognition.compare_faces([biden_encoding], unknown_encoding)
```

이 라이브러리를 다른 Python 라이브러리와 함께 사용한다면 실시간 얼굴 인식도 가능합니다:

![](https://cloud.githubusercontent.com/assets/896692/24430398/36f0e3f0-13cb-11e7-8258-4d0c9ce1e419.gif)

코드에 대해서는 [이 예제](https://github.com/ageitgey/face_recognition/blob/master/examples/facerec_from_webcam_faster.py) 를 참조하십시오.

## 온라인 데모

실제 사용자가 공유한 Jupyter notebook demo (공식은 아닙니다): [![Deepnote](https://beta.deepnote.org/buttons/try-in-a-jupyter-notebook.svg)](https://beta.deepnote.org/launch?template=face_recognition)

## 설치

### 요구 사항

  * Python 3.3+ 또는 Python 2.7
  * macOS 또는 Linux (Windows는 공식적으로 지원하지 않으나, 작동할 수도 있음)

### 설치 옵션들:

#### Mac 또는 Linux에서의 설치

우선, Python 바인딩을 통해 dlib이 이미 설치가 되어있는지를 확인해야 합니다:

  * [macOS 또는 Ubuntu에서 소스에서 dlib을 설치하는 방법](https://gist.github.com/ageitgey/629d75c1baac34dfa5ca2a1928a7aeaf)

다음으로, `pip3` (또는 Python2의 경우 `pip2`)을 사용하여 pypi에서의 모듈을 설치하십시오:

```bash
pip3 install face_recognition
```

또는, [이 부분](#deployment)을 참조하여, [Docker](https://www.docker.com/)로 이 라이브러리를 시도해보십시오.

설치에 대해 문제가 발생하였으면, [미리 구성된 VM](https://medium.com/@ageitgey/try-deep-learning-in-python-now-with-a-fully-pre-configured-vm-1d97d4c3e9b)을 사용해 볼 수도 있습니다.

#### Raspberry Pi 2+에서의 설치

  * [Raspberry Pi 2+ 설치 설명서](https://gist.github.com/ageitgey/1ac8dbe8572f3f533df6269dab35df65)

#### Windows에서 설치하기

Windows는 공식적으로 지원하지는 않지만, 친절한 유저들이 이 라이브러리를 어떻게 설치하는지 설명서를 작성했습니다:

  * [@masoudr의 Windows 10 설치 가이드 (dlib + face_recognition)](https://github.com/ageitgey/face_recognition/issues/175#issue-257710508)

#### 미리 구성된 가상머신 이미지(VM)를 설치하기

  * [미리 구성된 VM 이미지를 다운로드하기](https://medium.com/@ageitgey/try-deep-learning-in-python-now-with-a-fully-pre-configured-vm-1d97d4c3e9b) (VMware Player 또는 VirtualBox용).

## 사용법

### 명령 줄 인터페이스

`face_recognition`을 설치하면, 두 가지 간단한 명령 줄(CLI) 프로그램을 얻습니다:

* `face_recognition` - 사진 혹은 사진이 들어있는 폴더에서, 얼굴을 인식합니다.
* `face_detection` - 사진 혹은 사진이 들어있는 폴더에서, 얼굴을 찾습니다.

#### `face_recognition` 명령 줄 도구

`face_recognition` 명령을 사용하면 사진 혹은 사진이 들어있는 폴더에서, 얼굴을 인식할 수 있습니다.

그러기 위해서는 먼저, 이미 알고 있는(인식하고자 하는) 각 사람의 사진 한 장이 폴더에 있어야 합니다. 그리고 사진 속 그 사람의 이름을 딴 이미지 파일이 각각 하나씩 있어야 합니다:

![known](https://cloud.githubusercontent.com/assets/896692/23582466/8324810e-00df-11e7-82cf-41515eba704d.png)

다음으로, 식별하고 싶은 파일들이 있는 두 번째 폴더가 필요합니다:

![unknown](https://cloud.githubusercontent.com/assets/896692/23582465/81f422f8-00df-11e7-8b0d-75364f641f58.png)

그런 다음, 알고 있는 사람의 폴더와 모르는 사람의 폴더(또는 단일 이미지)를 전달하는 `face_recognition` 명령을 실행하면, 각 이미지에 있는 사람이 누군지 알 수 있습니다:

```bash
$ face_recognition ./pictures_of_people_i_know/ ./unknown_pictures/

/unknown_pictures/unknown.jpg,Barack Obama
/face_recognition_test/unknown_pictures/unknown.jpg,unknown_person
```

각각의 얼굴의 결과는 한 줄로 나타납니다. 각 줄은 파일 이름과 식별된 결과인 사람 이름이 쉼표로 구분되어 나타납니다.

`unknown_person`은 이미지 속에 알고 있는 사람의 폴더에 있는 그 누구와도 일치하지 않는 얼굴임을 의미합니다.

#### `face_detection` 명령 줄 도구

`face_detection` 명령을 사용하면 이미지에서 얼굴의 위치 (픽셀 좌표)를 찾을 수 있습니다.

`face_detection` 명령을 실행하여 검사 할 이미지 폴더 (또는 단일 이미지)를 전달하십시오:

```bash
$ face_detection  ./folder_with_pictures/

examples/image1.jpg,65,215,169,112
examples/image2.jpg,62,394,211,244
examples/image2.jpg,95,941,244,792
```

감지된 각 얼굴에 대해 한 줄씩 인쇄합니다. 결과값의 좌표는 각각 얼굴의 위쪽, 오른쪽, 아래쪽 및 왼쪽 좌표 (픽셀 단위)입니다.
 
##### 오차 조절 / 민감도

같은 사람에 대해 여러 개의 항목을 얻었다면, 사진에 있는 사람들이 매우 유사하게 보이기 때문이며 더욱 엄격한 얼굴 비교를 위해 낮은 허용치(tolerance value)가 필요합니다.

`--tolerance` 변수를 이용하여 이를 수행할 수 있습니다. 기본 허용치 값은 0.6이며 숫자가 낮으면 더욱 엄격한 얼굴 비교가 가능합니다:

```bash
$ face_recognition --tolerance 0.54 ./pictures_of_people_i_know/ ./unknown_pictures/

/unknown_pictures/unknown.jpg,Barack Obama
/face_recognition_test/unknown_pictures/unknown.jpg,unknown_person
```

허용치 설정을 조정하기 위해, 각 식별에서의 얼굴 거리를 알고 싶다면 `--show-distance true`를 통해 볼 수 있습니다:

```bash
$ face_recognition --show-distance true ./pictures_of_people_i_know/ ./unknown_pictures/

/unknown_pictures/unknown.jpg,Barack Obama,0.378542298956785
/face_recognition_test/unknown_pictures/unknown.jpg,unknown_person,None
```

##### 더 많은 예제들

파일 이름은 신경 쓰지 않고 각 사진에 있는 사람들의 이름만을 알고 싶다면 다음과 같이 할 수 있습니다:

```bash
$ face_recognition ./pictures_of_people_i_know/ ./unknown_pictures/ | cut -d ',' -f2

Barack Obama
unknown_person
```

##### 얼굴 인식 속도 향상

여러 개의 CPU 코어가 있는 컴퓨터를 사용한다면, 얼굴 인식을 동시에 수행할 수 있습니다. 예를 들면, 4개의 CPU 코어가 있는 환경에서는, 모든 CPU 코어를 병렬로 사용하여 같은 시간 동안 약 4배의 이미지들을 처리할 수 있습니다.

Python 3.4 이상을 사용하는 경우 `--cpus <number_of_cpu_cores_to_use>` 에 매개 변수(parameter)를 전달하십시오:

```bash
$ face_recognition --cpus 4 ./pictures_of_people_i_know/ ./unknown_pictures/
```

또한 `--cpus -1`을 전달하여 시스템의 모든 CPU 코어를 사용할 수도 있습니다.

#### Python 모듈

`face_recognition` 모듈을 불러와(import) 단 몇 줄의 코드만으로 얼굴 조작을 쉽게 할 수 있습니다. 이는 매우 간단합니다!

API 문서: [https://face-recognition.readthedocs.io](https://face-recognition.readthedocs.io/en/latest/face_recognition.html).

##### 이미지의 모든 얼굴을 자동으로 찾기

```python
import face_recognition

image = face_recognition.load_image_file("my_picture.jpg")
face_locations = face_recognition.face_locations(image)

# face_locations is now an array listing the co-ordinates of each face!
```

[이 예제](https://github.com/ageitgey/face_recognition/blob/master/examples/find_faces_in_picture.py) 를 사용하여 테스트 해 보십시오.

좀 더 정확한 딥 러닝 기반의 얼굴 탐지 모델을 채택할 수도 있습니다.

참고: 이 모델의 성능을 높이려면 (NVidia의 CUDA 라이브러리를 통한) GPU 가속이 필요합니다. 또한 `dlib`을 컴파일링할 때 CUDA 지원(support)을 활성화 할 수 있습니다.

```python
import face_recognition

image = face_recognition.load_image_file("my_picture.jpg")
face_locations = face_recognition.face_locations(image, model="cnn")

# face_locations is now an array listing the co-ordinates of each face!
```

[이 예제](https://github.com/ageitgey/face_recognition/blob/master/examples/find_faces_in_picture_cnn.py) 를 사용하여 테스트 해 보십시오.

이미지와 GPU가 둘 다 많은 경우, [얼굴을 일괄적으로 찾을](https://github.com/ageitgey/face_recognition/blob/master/examples/find_faces_in_batches.py) 수도 있습니다.

##### 이미지에서 자동으로 사람의 얼굴 특징 찾기

```python
import face_recognition

image = face_recognition.load_image_file("my_picture.jpg")
face_landmarks_list = face_recognition.face_landmarks(image)

# face_landmarks_list is now an array with the locations of each facial feature in each face.
# face_landmarks_list[0]['left_eye'] would be the location and outline of the first person's left eye.
```

[이 예제](https://github.com/ageitgey/face_recognition/blob/master/examples/find_facial_features_in_picture.py) 를 사용하여 테스트 해 보십시오.

##### 이미지에서 얼굴을 인식하고 누구인지 식별하기

```python
import face_recognition

picture_of_me = face_recognition.load_image_file("me.jpg")
my_face_encoding = face_recognition.face_encodings(picture_of_me)[0]

# my_face_encoding은 이제 어느 얼굴과도 비교할 수 있는 내가 가진 얼굴 특징의 보편적인 인코딩을 포함하게 되었습니다. 

unknown_picture = face_recognition.load_image_file("unknown.jpg")
unknown_face_encoding = face_recognition.face_encodings(unknown_picture)[0]

# 이제 `compare_faces`를 통해 두 얼굴이 같은 얼굴인지 비교할 수 있습니다!

results = face_recognition.compare_faces([my_face_encoding], unknown_face_encoding)

if results[0] == True:
    print("It's a picture of me!")
else:
    print("It's not a picture of me!")
```

[이 예제](https://github.com/ageitgey/face_recognition/blob/master/examples/recognize_faces_in_pictures.py) 를 사용하여 테스트 해 보십시오.

## Python 코드 예제

모든 예제는 [여기](https://github.com/ageitgey/face_recognition/tree/master/examples) 에 있습니다.


#### 얼굴 탐지

* [사진에서 얼굴 찾기](https://github.com/ageitgey/face_recognition/blob/master/examples/find_faces_in_picture.py)
* [사진에서 얼굴 찾기(딥 러닝 사용)](https://github.com/ageitgey/face_recognition/blob/master/examples/find_faces_in_picture_cnn.py)
* [이미지 모음에서 얼굴 찾기 w/ GPU (딥 러닝 사용)](https://github.com/ageitgey/face_recognition/blob/master/examples/find_faces_in_batches.py)
* [웹캠을 사용하여 라이브 비디오의 모든 얼굴을 흐리게 처리하기 (OpenCV가 설치되어 있어야 함)](https://github.com/ageitgey/face_recognition/blob/master/examples/blur_faces_on_webcam.py)

#### 얼굴의 특징

* [사진의 특정 얼굴 특징 확인하기](https://github.com/ageitgey/face_recognition/blob/master/examples/find_facial_features_in_picture.py)
* [Apply (horribly ugly) digital make-up](https://github.com/ageitgey/face_recognition/blob/master/examples/digital_makeup.py)

#### 얼굴 인식

* [알고있는 사람들의 사진을 기반으로 사진에서 알 수 없는 얼굴을 찾고 인식하기](https://github.com/ageitgey/face_recognition/blob/master/examples/recognize_faces_in_pictures.py)
* [사진 안의 각 사람들을 식별하고 주의에 상자를 그리기](https://github.com/ageitgey/face_recognition/blob/master/examples/identify_and_draw_boxes_on_faces.py)
* [얼굴 구분을 참/거짓 구분 대신 숫자로 비교하기](https://github.com/ageitgey/face_recognition/blob/master/examples/face_distance.py)
* [웹캠을 사용하여 라이브 비디오의 얼굴 인식하기 - 간단함 / 느린 버전 (OpenCV가 설치되어 있어야 함)](https://github.com/ageitgey/face_recognition/blob/master/examples/facerec_from_webcam.py)
* [웹캠을 사용하여 라이브 비디오의 얼굴 인식하기 - 빠른 버전 (OpenCV가 설치되어 있어야 함)](https://github.com/ageitgey/face_recognition/blob/master/examples/facerec_from_webcam_faster.py)
* [비디오 파일에서 얼굴을 인식하고 새 비디오 파일을 작성하기 (OpenCV가 설치되어 있어야 함)](https://github.com/ageitgey/face_recognition/blob/master/examples/facerec_from_video_file.py)
* [Raspberry Pi w/ camera에서의 얼굴 인식하기](https://github.com/ageitgey/face_recognition/blob/master/examples/facerec_on_raspberry_pi.py)
* [HTTP를 통해 얼굴을 인식하는 웹 서비스 실행하기 (Flask가 설치되어 있어야 함)](https://github.com/ageitgey/face_recognition/blob/master/examples/web_service_example.py)
* [K-nearest neighbors classifier를 통한 얼굴 인식하기](https://github.com/ageitgey/face_recognition/blob/master/examples/face_recognition_knn.py)

## 독립적인 실행 파일 만들기
`Python`이나 `face_recognition`을 설치할 필요 없이 실행할 수 있는 독립적인 실행형 실행 파일을 만들려면 [PyInstaller](https://github.com/pyinstaller/pyinstaller) 를 사용하면 됩니다. 그러나, 이 라이브러리로 작업하려면 어느 정도의 설정 커스텀이 필요합니다. 방법에 대해서는 [이 이슈](https://github.com/ageitgey/face_recognition/issues/357) 를 참조하십시오.

## `face_recognition`을 다루는 글 및 가이드

- 얼굴 인식이 어떻게 작동하는지에 관한 글: [딥 러닝을 통한 현대적 얼굴 인식](https://medium.com/@ageitgey/machine-learning-is-fun-part-4-modern-face-recognition-with-deep-learning-c3cffc121d78)
  - 알고리즘과 알고리즘이 일반적으로 어떻게 작동하는지
- Adrian Rosebrock의 [OpenCV, Python 및 딥 러닝을 통한 얼굴 인식](https://www.pyimagesearch.com/2018/06/18/face-recognition-with-opencv-python-and-deep-learning/)
  - 실제로 얼굴 인식을 사용하는 법
- Adrian Rosebrock의 [Raspberry Pi 얼굴 인식](https://www.pyimagesearch.com/2018/06/25/raspberry-pi-face-recognition/)
  - Raspberry Pi에서 어떻게 사용하는지
- Adrian Rosebrock의 [Python 얼굴 클러스터링](https://www.pyimagesearch.com/2018/07/09/face-clustering-with-python/) by Adrian Rosebrock
  - 비지도 학습을 사용하여 각 사진에 나타나는 사람을 기반으로 사진을 자동 클러스터하는 방법

## 얼굴 인식이 작동하는지

black box 라이브러리에 의존하는 대신 얼굴 위치와 인식이 어떻게 작동하는지 알고 싶으시다면 [이 글](https://medium.com/@ageitgey/machine-learning-is-fun-part-4-modern-face-recognition-with-deep-learning-c3cffc121d78) 을 읽으십시오.

## 주의 사항

* 얼굴 인식의 모델은 성인에 대한 데이터를 통해 훈련이 되었으며 따라서 어린이의 경우에는 잘 적용이 되지 않습니다. 0.6의 기본 임계 값을 사용한다면 어린이들의 얼굴을 구분하지 못하는 경향이 있습니다.
* 소수 민족마다 정확성이 다를 수 있습니다. 자세한 내용은 [이 위키 페이지](https://github.com/ageitgey/face_recognition/wiki/Face-Recognition-Accuracy-Problems#question-face-recognition-works-well-with-european-individuals-but-overall-accuracy-is-lower-with-asian-individuals) 를 참조하십시오.

## <a name="deployment">클라우드 호스트에 배포 (Heroku, AWS, 기타 등)</a>

`face_recognition`은 C++로 작성된 `dlib`에 의존하기 때문에, Heroku 또는 AWS와 같은 클라우드 호스팅 제공 업체에 이를 사용하는 앱을 배포하는 것은 까다로울 수 있습니다.

더 쉬운 작업을 위해, [Docker](https://www.docker.com/) container에서 `face_recognition`으로 빌드 된 앱을 실행하는 방법을 보여주는 이 repo의 Dockerfile 예제가 있습니다. 이를 통해 Docker 이미지를 지원하는 모든 서비스에 배포할 수 있어야합니다.

다음을 실행하여 Docker 이미지를 로컬로 시도 할 수 있습니다: `docker-compose up --build`

GPU (드라이버 >= 384.81) 및 [Nvidia-Docker](https://github.com/NVIDIA/nvidia-docker) 가 설치된 Linux 사용자는 GPU에서 예제를 실행할 수 있습니다: [docker-compose.yml](docker-compose.yml) 파일을 열고 `dockerfile : Dockerfile.gpu` 및 `runtime : nvidia` 행의 주석 처리를 제거합니다.

## 문제가 있으십니까?

문제가 발생하면 github 문제를 제기하기 전에 위키의 [일반적인 오류](https://github.com/ageitgey/face_recognition/wiki/Common-Errors) 섹션을 읽어보십시오.

## 감사의 말

* `dlib`를 만들고 이 라이브러리에 사용된 얼굴 인식 기능과 얼굴 인코딩 모델을 제공한 [Davis King](https://github.com/davisking) ([@nulhom](https://twitter.com/nulhom)) 에게 많은 감사를 드립니다. 얼굴 인코딩을 지원하는 ResNet에 대한 자세한 내용은 [블로그 게시물](http://blog.dlib.net/2017/02/high-quality-face-recognition-with-deep.html) 을 확인하십시오.
* numpy, scipy, scikit-image, pillow 등의 모든 멋진 파이썬 데이터 과학 라이브러리에서 일하는 모든 사람들에게 감사합니다. 이런 종류의 것들을 파이썬에서 쉽고 재미있게 만듭니다.
* [Cookiecutter](https://github.com/audreyr/cookiecutter)
와 [audreyr/cookiecutter-pypackage](https://github.com/audreyr/cookiecutter-pypackage)
 프로젝트 템플릿 덕분에 파이썬 프로젝트 패키징 방식이 더 괜찮아 졌습니다.
