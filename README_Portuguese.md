Reconheça e manipule rostos em Python ou pela linha de comando com
a biblioteca de reconhecimento facial mais simples do mundo.

Construído usando o reconhecimento facial de última geração da [dlib](http://dlib.net/),
com aprendizado profundo. O modelo tem uma precisão de 99,38% no benchmark
[Labeled Faces in the Wild](http://vis-www.cs.umass.edu/lfw/).

Isso também fornece uma ferramenta de linha de comando simples, `face_recognition`, que permite
que você faça reconhecimento facial em uma pasta de imagens a partir da linha de comando!

[![PyPI](https://img.shields.io/pypi/v/face_recognition.svg)](https://pypi.python.org/pypi/face_recognition)
[![Build Status](https://github.com/ageitgey/face_recognition/workflows/CI/badge.svg?branch=master&event=push)](https://github.com/ageitgey/face_recognition/actions?query=workflow%3ACI)
[![Documentation Status](https://readthedocs.org/projects/face-recognition/badge/?version=latest)](http://face-recognition.readthedocs.io/en/latest/?badge=latest)

## Recursos

#### Encontre rostos em imagens

Encontre todos os rostos que aparecem em uma imagem:

![](https://cloud.githubusercontent.com/assets/896692/23625227/42c65360-025d-11e7-94ea-b12f28cb34b4.png)

```python
import face_recognition
image = face_recognition.load_image_file("your_file.jpg")
face_locations = face_recognition.face_locations(image)
```

#### Encontre e manipule características faciais em fotos

Obtenha a localização e os contornos dos olhos, nariz, boca e queixo de cada pessoa.

![](https://cloud.githubusercontent.com/assets/896692/23625282/7f2d79dc-025d-11e7-8728-d8924596f8fa.png)

```python
import face_recognition
image = face_recognition.load_image_file("your_file.jpg")
face_landmarks_list = face_recognition.face_landmarks(image)
```

Encontrar características faciais é super útil para muitas coisas importantes. Mas você também pode usá-lo para coisas realmente bobas,
como aplicar [maquiagem digital](https://github.com/ageitgey/face_recognition/blob/master/examples/digital_makeup.py) (pense em "Meitu"):

![](https://cloud.githubusercontent.com/assets/896692/23625283/80638760-025d-11e7-80a2-1d2779f7ccab.png)

#### Identificar rostos em fotos

Reconhecer quem aparece em cada foto.

![](https://cloud.githubusercontent.com/assets/896692/23625229/45e049b6-025d-11e7-89cc-8a71cf89e713.png)

```python
import face_recognition
known_image = face_recognition.load_image_file("biden.jpg")
unknown_image = face_recognition.load_image_file("unknown.jpg")

biden_encoding = face_recognition.face_encodings(known_image)[0]
unknown_encoding = face_recognition.face_encodings(unknown_image)[0]

results = face_recognition.compare_faces([biden_encoding], unknown_encoding)
```


Você pode até usar esta biblioteca com outras bibliotecas Python para fazer reconhecimento facial em tempo real:

![](https://cloud.githubusercontent.com/assets/896692/24430398/36f0e3f0-13cb-11e7-8258-4d0c9ce1e419.gif)

Veja [este exemplo](https://github.com/ageitgey/face_recognition/blob/master/examples/facerec_from_webcam_faster.py) para o código.

## Demonstrações Online

Demonstração do notebook Jupyter compartilhada e contribuída pelo usuário (sem suporte oficial): [![Deepnote](https://beta.deepnote.org/buttons/try-in-a-jupyter-notebook.svg)](https://beta.deepnote.org/launch?template=face_recognition)

## Instalação

### Requisitos

* Python 3.3+ ou Python 2.7
* macOS ou Linux (Windows não é oficialmente suportado, mas pode funcionar)

### Opções de instalação:

#### Instalando no Mac ou Linux

Primeiro, certifique-se de que o dlib já esteja instalado com as vinculações do Python:

* [Como instalar o dlib a partir do código-fonte no macOS ou Ubuntu](https://gist.github.com/ageitgey/629d75c1baac34dfa5ca2a1928a7aeaf)

Em seguida, certifique-se de que o cmake esteja instalado:

```brew install cmake```

Finalmente, instale este módulo do pypi usando `pip3` (ou `pip2` para Python 2):

```bash
pip3 install face_recognition
```

Como alternativa, você pode tentar esta biblioteca com o [Docker](https://www.docker.com/), consulte [esta seção](#deployment).

Se estiver com problemas com a instalação, você também pode tentar uma
[VM pré-configurada](https://medium.com/@ageitgey/try-deep-learning-in-python-now-with-a-fully-pre-configured-vm-1d97d4c3e9b).

#### Instalando em uma placa Nvidia Jetson Nano

* [Instruções de instalação do Jetson Nano](https://medium.com/@ageitgey/build-a-hardware-based-face-recognition-system-for-150-with-the-nvidia-jetson-nano-and-python-a25cb8c891fd)
* Siga atentamente as instruções do artigo. Há um bug nas bibliotecas CUDA do Jetson Nano que fará com que essa biblioteca falhe silenciosamente se você não seguir as instruções do artigo para comentar uma linha na biblioteca dlib e recompilá-la.

#### Instalando no Raspberry Pi 2+

* [Instruções de instalação do Raspberry Pi 2+](https://gist.github.com/ageitgey/1ac8dbe8572f3f533df6269dab35df65)

#### Instalando no FreeBSD

```bash
pkg install graphics/py-face_recognition
```

#### Instalando no Windows

Embora o Windows não seja oficialmente suportado, usuários prestativos postaram instruções sobre como instalar esta biblioteca:

* [Guia de instalação do Windows 10 de @masoudr (dlib + face_recognition)](https://github.com/ageitgey/face_recognition/issues/175#issue-257710508)

#### Instalando uma imagem de máquina virtual pré-configurada

* [Baixe a VM pré-configurada imagem](https://medium.com/@ageitgey/try-deep-learning-in-python-now-with-a-fully-pre-configured-vm-1d97d4c3e9b) (para VMware Player ou VirtualBox).

## Uso

### Interface de Linha de Comando

Ao instalar o `face_recognition`, você obtém dois programas de linha de comando simples:

* `face_recognition` - Reconhece rostos em uma fotografia ou pasta cheia de fotografias.

* `face_detection` - Encontra rostos em uma fotografia ou pasta cheia de fotografias.

#### Ferramenta de linha de comando `face_recognition`

O comando `face_recognition` permite reconhecer rostos em uma fotografia ou pasta cheia de fotografias.

Primeiro, você precisa fornecer uma pasta com uma foto de cada pessoa que você
já conhece. Deve haver um arquivo de imagem para cada pessoa,
com os arquivos nomeados de acordo com quem está na foto:

![conhecido](https://cloud.githubusercontent.com/assets/896692/23582466/8324810e-00df-11e7-82cf-41515eba704d.png)

Em seguida, você precisa de uma segunda pasta com os arquivos que deseja identificar:

![desconhecido](https://cloud.githubusercontent.com/assets/896692/23582465/81f422f8-00df-11e7-8b0d-75364f641f58.png)

Então, basta executar o comando `face_recognition`, passando
a pasta de pessoas conhecidas e a pasta (ou imagem única) com pessoas desconhecidas
e ele informa quem está em cada Imagem:

```bash
$ reconhecimento_facial ./imagens_de_pessoas_que_conheço/ ./imagens_desconhecidas/

/imagens_desconhecidas/desconhecido.jpg,Barack Obama
/teste_de_reconhecimento_facial/imagens_desconhecidas/desconhecido.jpg,pessoa_desconhecida
```

Há uma linha na saída para cada rosto. Os dados são separados por vírgula
com o nome do arquivo e o nome da pessoa encontrada.

Uma `unknown_person` é um rosto na imagem que não corresponde a ninguém em
sua pasta de pessoas conhecidas.

#### Ferramenta de linha de comando `face_detection`

O comando `face_detection` permite encontrar a localização (coordenadas em pixels)
de quaisquer rostos em uma imagem.

Basta executar o comando `face_detection`, passando uma pasta de imagens
para verificar (ou uma única imagem):

```bash
$ face_detection ./folder_with_pictures/

examples/image1.jpg,65,215,169,112
examples/image2.jpg,62,394,211,244
examples/image2.jpg,95,941,244,792
```

Ele imprime uma linha para cada rosto detectado. As coordenadas
relatadas são as coordenadas superior, direita, inferior e esquerda do rosto (em pixels).

##### Ajustando Tolerância/Sensibilidade

Se você estiver obtendo várias correspondências para a mesma pessoa, pode ser que
as pessoas em suas fotos sejam muito semelhantes e um valor de tolerância menor
seja necessário para tornar as comparações de rostos mais rigorosas.

Você pode fazer isso com o parâmetro `--tolerance`. O valor padrão de tolerância
é 0,6 e números menores tornam as comparações faciais mais rigorosas:

```bash
$ face_recognition --tolerance 0.54 ./pictures_of_people_i_know/ ./unknown_pictures/

/unknown_pictures/unknown.jpg,Barack Obama
/face_recognition_test/unknown_pictures/unknown.jpg,unknown_person
```

Se quiser ver a distância facial calculada para cada correspondência para
ajustar a configuração de tolerância, você pode usar `--show-distance true`:

```bash
$ face_recognition --show-distance true ./pictures_of_people_i_know/ ./unknown_pictures/

/unknown_pictures/unknown.jpg,Barack Obama,0.378542298956785
/face_recognition_test/unknown_pictures/unknown.jpg,unknown_person,None
```

##### Mais Exemplos

Se você simplesmente quiser saber os nomes das pessoas em cada fotografia, mas não
se importar com os nomes dos arquivos, você pode fazer isso:

```bash
$ face_recognition ./pictures_of_people_i_know/ ./unknown_pictures/ | cut -d ',' -f2

Barack Obama
unknown_person
```

##### Acelerando o Reconhecimento Facial

O reconhecimento facial pode ser feito em paralelo se você tiver um computador com
vários núcleos de CPU. Por exemplo, se o seu sistema tiver 4 núcleos de CPU, você poderá
processar cerca de 4 vezes mais imagens no mesmo período de tempo usando
todos os núcleos da CPU em paralelo.

Se você estiver usando Python 3.4 ou mais recente, passe o parâmetro `--cpus <número_de_núcleos_de_cpu_a_usar>`:

```bash
$ face_recognition --cpus 4 ./pictures_of_people_i_know/ ./unknown_pictures/
```

Você também pode passar `--cpus -1` para usar todos os núcleos de CPU do seu sistema.

#### Módulo Python

Você pode importar o módulo `face_recognition` e manipular facilmente
rostos com apenas algumas linhas de código. É super fácil!

Documentação da API: [https://face-recognition.readthedocs.io](https://face-recognition.readthedocs.io/en/latest/face_recognition.html).

##### Encontre automaticamente todos os rostos em uma imagem

```python
import face_recognition

image = face_recognition.load_image_file("my_picture.jpg")
face_locations = face_recognition.face_locations(image)

# face_locations agora é um array que lista as coordenadas de cada rosto!
```

Veja [este exemplo](https://github.com/ageitgey/face_recognition/blob/master/examples/find_faces_in_picture.py)
para experimentar.

Você também pode optar por um modelo de detecção facial baseado em aprendizado profundo um pouco mais preciso.

Observação: a aceleração por GPU (via biblioteca CUDA da NVidia) é necessária para um bom
desempenho com este modelo. Você também deve habilitar o suporte a CUDA
ao compilar `dlib`.

```python
import face_recognition

image = face_recognition.load_image_file("minha_imagem.jpg")
face_locations = face_recognition.face_locations(image, model="cnn")

# face_locations agora é um array que lista as coordenadas de cada rosto!
```

Veja [este exemplo](https://github.com/ageitgey/face_recognition/blob/master/examples/find_faces_in_picture_cnn.py)
para experimentar.

Se você tiver muitas imagens e uma GPU, também poderá
[encontrar rostos em lotes](https://github.com/ageitgey/face_recognition/blob/master/examples/find_faces_in_batches.py).

##### Localizar automaticamente as características faciais de uma pessoa em uma imagem

```python
import face_recognition

image = face_recognition.load_image_file("minha_imagem.jpg")
face_landmarks_list = face_recognition.face_landmarks(imagem)

# face_landmarks_list agora é um array com as localizações de cada característica facial em cada rosto.
# face_landmarks_list[0]['left_eye'] seria a localização e o contorno do olho esquerdo da primeira pessoa.
```

Veja [este exemplo](https://github.com/ageitgey/face_recognition/blob/master/examples/find_facial_features_in_picture.py)
para experimentar.

##### Reconhecer rostos em imagens e identificá-los

```python
import face_recognition

picture_of_me = face_recognition.load_image_file("me.jpg")
my_face_encoding = face_recognition.face_encodings(picture_of_me)[0]

# my_face_encoding agora contém uma 'codificação' universal das minhas características faciais que pode ser comparada a qualquer outra imagem de rosto!

known_picture = face_recognition.load_image_file("unknown.jpg")
unknown_face_encoding = face_recognition.face_encodings(unknown_picture)[0]

# Agora podemos ver que as duas codificações faciais são da mesma pessoa com `compare_faces`!

results = face_recognition.compare_faces([my_face_encoding], unknown_face_encoding)

if results[0] == True:
print("É uma foto minha!")
else:
print("Não é uma foto minha!")
```

Veja [este exemplo](https://github.com/ageitgey/face_recognition/blob/master/examples/recognize_faces_in_pictures.py)
para testar.

## Exemplos de Código Python

Todos os exemplos estão disponíveis [aqui](https://github.com/ageitgey/face_recognition/tree/master/examples).

#### Detecção Facial

* [Encontre rostos em uma fotografia](https://github.com/ageitgey/face_recognition/blob/master/examples/find_faces_in_picture.py)
* [Encontre rostos em uma fotografia (usando aprendizado profundo)](https://github.com/ageitgey/face_recognition/blob/master/examples/find_faces_in_picture_cnn.py)
* [Encontre rostos em lotes de imagens com GPU (usando aprendizado profundo)](https://github.com/ageitgey/face_recognition/blob/master/examples/find_faces_in_batches.py)
* [Desfoque todos os rostos em um vídeo ao vivo usando sua webcam (requer OpenCV para ser instalado)](https://github.com/ageitgey/face_recognition/blob/master/examples/blur_faces_on_webcam.py)

#### Características Faciais

* [Identificar características faciais específicas em uma fotografia](https://github.com/ageitgey/face_recognition/blob/master/examples/find_facial_features_in_picture.py)
* [Aplicar maquiagem digital (horrivelmente feia)](https://github.com/ageitgey/face_recognition/blob/master/examples/digital_makeup.py)

#### Reconhecimento Facial

* [Encontrar e reconhecer rostos desconhecidos em uma fotografia com base em fotos de pessoas conhecidas](https://github.com/ageitgey/face_recognition/blob/master/examples/recognize_faces_in_pictures.py)
* [Identificar e desenhe caixas ao redor de cada pessoa em uma foto](https://github.com/ageitgey/face_recognition/blob/master/examples/identify_and_draw_boxes_on_faces.py)
* [Compare rostos por distância numérica em vez de apenas correspondências Verdadeiro/Falso](https://github.com/ageitgey/face_recognition/blob/master/examples/face_distance.py)
* [Reconheça rostos em vídeos ao vivo usando sua webcam - Versão simples/mais lenta (requer a instalação do OpenCV)](https://github.com/ageitgey/face_recognition/blob/master/examples/facerec_from_webcam.py)
* [Reconheça rostos em vídeos ao vivo usando sua webcam - Versão mais rápida (requer a instalação do OpenCV) instalado)](https://github.com/ageitgey/face_recognition/blob/master/examples/facerec_from_webcam_faster.py)
* [Reconhecer rostos em um arquivo de vídeo e gravar um novo arquivo de vídeo (requer a instalação do OpenCV)](https://github.com/ageitgey/face_recognition/blob/master/examples/facerec_from_video_file.py)
* [Reconhecer rostos em um Raspberry Pi com câmera](https://github.com/ageitgey/face_recognition/blob/master/examples/facerec_on_raspberry_pi.py)
* [Executar um serviço web para reconhecer rostos via HTTP (requer a instalação do Flask)](https://github.com/ageitgey/face_recognition/blob/master/examples/web_service_example.py)
* [Reconhecer rostos com um Classificador de K-vizinhos mais próximos](https://github.com/ageitgey/face_recognition/blob/master/examples/face_recognition_knn.py)
* [Treine várias imagens por pessoa e reconheça rostos usando uma SVM](https://github.com/ageitgey/face_recognition/blob/master/examples/face_recognition_svm.py)

## Criando um Executável Autônomo
Se você deseja criar um executável autônomo que possa ser executado sem a necessidade de instalar `python` ou `face_recognition`, você pode usar o [PyInstaller](https://github.com/pyinstaller/pyinstaller). No entanto, ele requer alguma configuração personalizada para funcionar com esta biblioteca. Consulte [este problema](https://github.com/ageitgey/face_recognition/issues/357) para saber como fazer isso.

## Artigos e Guias que abordam `reconhecimento_facial`

- Meu artigo sobre como o reconhecimento facial funciona: [Reconhecimento facial moderno com aprendizado profundo](https://medium.com/@ageitgey/machine-learning-is-fun-part-4-modern-face-recognition-with-deep-learning-c3cffc121d78)
- Aborda os algoritmos e como eles funcionam em geral
- [Reconhecimento facial com OpenCV, Python e aprendizado profundo](https://www.pyimagesearch.com/2018/06/18/face-recognition-with-opencv-python-and-deep-learning/) por Adrian Rosebrock
- Aborda como usar o reconhecimento facial na prática
- [Reconhecimento facial com Raspberry Pi](https://www.pyimagesearch.com/2018/06/25/raspberry-pi-face-recognition/) por Adrian Rosebrock
- Aborda como usar isso em um Raspberry Pi
- [Agrupamento de faces com Python](https://www.pyimagesearch.com/2018/07/09/face-clustering-with-python/) por Adrian Rosebrock
- Aborda como agrupar fotos automaticamente com base em quem aparece em cada foto usando aprendizado não supervisionado

## Como funciona o reconhecimento facial

Se você quiser aprender como a localização e o reconhecimento facial funcionam em vez de
depender de uma biblioteca de caixa preta, [leia meu artigo](https://medium.com/@ageitgey/machine-learning-is-fun-part-4-modern-face-recognition-with-deep-learning-c3cffc121d78).

## Advertências

* O modelo de reconhecimento facial é treinado em adultos e não funciona muito bem em crianças. Ele tende a confundir
as crianças facilmente usando o limite de comparação padrão de 0,6.
* A precisão pode variar entre os grupos étnicos. Consulte [esta página wiki](https://github.com/ageitgey/face_recognition/wiki/Face-Recognition-Accuracy-Problems#question-face-recognition-works-well-with-european-individuals-but-overall-accuracy-is-lower-with-asian-individuals) para obter mais detalhes.

## <a name="deployment">Implantação em Hosts em Nuvem (Heroku, AWS, etc.)</a>

Como `face_recognition` depende de `dlib`, que é escrito em C++, pode ser complicado implantar um aplicativo
usando-o em um provedor de hospedagem em nuvem como Heroku ou AWS.

Para facilitar, há um Dockerfile de exemplo neste repositório que mostra como executar um aplicativo criado com
`face_recognition` em um contêiner [Docker](https://www.docker.com/). Com isso, você poderá implementar
em qualquer serviço que suporte imagens Docker.

Você pode testar a imagem Docker localmente executando: `docker-compose up --build`

Há também [várias imagens Docker pré-compiladas.](docker/README.md)

Usuários de Linux com uma GPU (drivers >= 384.81) e [Nvidia-Docker](https://github.com/NVIDIA/nvidia-docker) instalada podem executar o exemplo na GPU: Abra o arquivo [docker-compose.yml](docker-compose.yml) e descomente as linhas `dockerfile: Dockerfile.gpu` e `runtime: nvidia`.

## Está com problemas?

Se tiver problemas, leia a seção [Erros Comuns](https://github.com/ageitgey/face_recognition/wiki/Common-Errors) do wiki antes de registrar um problema no GitHub.

## Obrigado

* Muito, muito obrigado a [Davis King](https://github.com/davisking) ([@nulhom](https://twitter.com/nulhom))
por criar a dlib e por fornecer os modelos treinados de detecção de características faciais e codificação facial
usados ​​nesta biblioteca. Para mais informações sobre o ResNet que alimenta as codificações faciais, confira
sua [postagem no blog](http://blog.dlib.net/2017/02/high-quality-face-recognition-with-deep.html).
* Obrigado a todos que trabalham em todas as incríveis bibliotecas de ciência de dados em Python, como numpy, scipy, scikit-image,
pillow, etc., etc., que tornam esse tipo de coisa tão fácil e divertida em Python.
* Agradecimentos a [Cookiecutter](https://github.com/audreyr/cookiecutter) e ao modelo de projeto
[audreyr/cookiecutter-pypackage](https://github.com/audreyr/cookiecutter-pypackage)
por tornar o empacotamento de projetos Python muito mais tolerável.
