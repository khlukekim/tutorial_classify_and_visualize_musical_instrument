# 음원 분류 및 2차원 시각화

이 튜토리얼에서는 인공신경망을 이용해 다양한 음원의 종류를 파악하는 분류기를 만들어 보고, 이를 이용해 각 음원의 특징을 2차원 상에 나타내는 시각화를 해 보겠습니다.

이 튜토리얼은 다음과 같은 구조로 이루어져 있습니다.

== toc ==

## 준비

이 튜토리얼을 완료하기 위해선 다음과 같은 파이썬 라이브러리가 필요합니다.

- numpy: 수치 자료를 고속으로 다루기 위한 라이브러리입니다.
- librosa: 음원 파일을 분석하는 데 필요한 라이브러리입니다.
- tensorflow: 인공신경망을 이용해 학습시키는 데 필요한 라이브러리입니다.
- scikit-learn: tSNE를 이용해 자료를 2차원에 시각화하는데 필요한 라이브러리입니다.

== 설치 과정? ==

## 자료 준비와 전처리

인공신경망을 학습시키기 위해선 좋은 품질의 자료가 필요합니다. 마침 구글의 마젠타 프로젝트에서 [NSynth](https://magenta.tensorflow.org/datasets/nsynth)라는 악기 데이터셋을 제공합니다.

### 데이터셋 다운받기

NSynth 데이터셋은 tfrecord, json/wav 두 가지 종류로 다운로드할 수 있습니다. 이번 튜토리얼에서는 wav파일을 직접 전처리하는 과정을 연습해볼 것이므로 json/wav 형태의 자료를 다운로드합니다. 또한, NSynth 데이터셋은 세 부분으로 나뉘어 있습니다. 이번에는 세 부분 모두 필요하므로 전부 다운로드해서 압축을 풀도록 합시다.

각 데이터셋의 의미는 다음과 같습니다.
- Train: 모델을 학습시킬 때 사용하는 부분입니다.
- Valid: 최적의 조건을 찾기 위해 여러 조건에서 모델을 학습하게 되는데, 이 때 이 부분을 이용해 각 모델의 성능을 비교합니다.
- Test: Valid를 이용해 찾은 최고의 모델을 최종적으로 테스트하는데 필요한 부분입니다.

NSynth 데이터셋은 다양한 가상악기에서 높이와 세기를 달리하며 한 음씩 녹음한 데이터셋입니다. 우리가 분류하고 싶은 것은 악기의 종류(Instrument Families)입니다. [NSynth 데이터셋의 설명](https://magenta.tensorflow.org/datasets/nsynth#instrument-families)에 따르면 11개의 종류가 있고, 각 음원은 단 하나의 종류에만 해당한다고 합니다.

세 파일의 압축을 풀면 다음과 같은 파일 구조가 나오게 됩니다.

```
- train
  | audio
  | examples.json
- valid
  | audio
  | examples.json
- test
  | audio
  | examples.json
```

압축 해제 방식 등의 이유로 폴더 구조가 다르다면 위와 같이 맞춰줍니다. `audio` 폴더 안에는 `wav` 파일들이 있고, 그 위에는 `examples.json` 파일이 있습니다. `wav`파일은 압축되지 않은 음원 파일이고, `json`은 해당 음원 파일에 관한 추가적인 정보를 답고 있는 파일입니다. `wav` 파일의 이름은 `bass_synthetic_035-025-030.wav`와 같은 식으로 `음원종류_생성방식_악기번호_높이_세기.wav`로 이루어져 있습니다. 우리는 음원 종류만 필요하므로 `json` 파일을 따로 분석할 필요는 없습니다.

### 데이터셋 정보 정리하기

데이터가 준비되었으면 먼저 각 파일의 경로와 악기 종류를 정리해둔 파일을 만들어 보겠습니다. `example.json`에도 우리가 필요한 정보가 정리되어 있지만 모든 데이터셋이 그런 것을 제공하지는 않으므로 연습해보도록 하겠습니다.

먼저 악기 종류를 정리해서 저장해 둡니다.

`labels.txt` 파일을 만들고 다음과 같이 내용을 작성한 후 저장합니다.

```bass
brass
flute
guitar
keyboard
mallet
organ
reed
string
synth_lead
vocal
```

레이블은 일반적으로 어떤 학습 모델이 찾아야 할 답(이 경우에는 악기의 종류)을 말합니다.

`gather_information.py` 파일을 만듭니다. 이 파일에서는 다음과 같은 작업들을 하겠습니다.

- audio 폴더에서 `wav` 파일 목록을 얻어오기
- 각 파일 이름에서 앞 부분을 잘라서 어떤 종류의 악기인지 파악하기
- train, valid, test별로 음원 파일 이름과 악기 종류를 모아서 저장하기

그럼 실제로 파일을 작성해 보겠습니다. 다음 코드를 참고해서 파일을 작성해 주세요.

```
import os # 파일 목록을 구할 때 필요한 패키지

def gather_information(part): # part 인자에는 'train', 'valid', 'test' 등을 넣어줍니다.
  label_list = open('labels.txt').read().strip().split('\n') # labels.txt 파일을 읽어서 각 줄을 기준으로 나눕니다.
  files = [] # wav 파일의 목록
  labels = [] # 각 파일의 label
  all_files = os.listdir(part + '/audio') # 각 part 아래 'audio' 안에 있는 파일의 목록을 불러옵니다.
  for f in all_files: # all_files에 있는 각 파일들에 대해서
    if f[-4:] == '.wav': # 파일이 '.wav'로 끝나면
      files.append(f[:-4]) # files 목록에 추가하고,
      label = f.split('_')[0] # 파일의 가장 첫 단어를 잘라냅니다.
      if label == 'synth': # synth_lead의 경우는 두 단어로 이루어져 있으므로
        label = 'synth_lead' # 첫 단어가 synth인 경우에는 label 이름을 synth_lead로 바꿔줍니다.
      labels.append(label_list.index(label)) # 이제 label_list에서 label의 위치를 찾아서 labels에 추가합니다.
  file_out = open(part + '_samples.txt', 'w') # part+'_samples.txt' 에 파일 목록을 적어줍니다.
  for f in files:
    file_out.write(f + '\n')
  file_out.close()
  label_out = open(part + '_labels.txt', 'w') # label도 저장합니다.
  for l in labels:
    label_out.write(str(l) + '\n')
  label_out.close()

if __name__ == '__main__' :
  gather_information('train') # 위 함수를 'train', 'valid', 'test'에 대해 실행합니다.
  gather_information('valid')
  gather_information('test')
```

파일의 각 부분을 좀 더 자세히 보겠습니다.

```def gather_information(part):```

정보를 모으는 기능을 독립된 함수로 만들었습니다. 'train', 'valid', 'test'에 대해 동일한 작업을 반복해야 하므로 이렇게 동일한 작업을 함수로 만들면 코드가 간결하고 이해하기 쉬워집니다. part에는 'train', 'valid', 'test' 중 하나가 들어올 수 있습니다. 만약 폴더명이 다르다면 해당 폴더명을 넣어도 됩니다.

```labels_list = open('labels.txt').read().strip().split('\n')```

우리는 악기의 종류를 'bass', 'brass' 처럼 문자로 기억하지만, 인공지능 모델을 학습시키는 등 연산 작업을 할 때는 숫자로 변환시켜서 사용합니다. 예를 들면 'bass'는 `0`, 'brass'는 `1`같은 식으로 사용합니다. `labels_list`는 우리가 사용할 악기의 종류를 담고 있는 리스트입니다. 'labels.txt'의 내용을 읽어서, 줄바꿈('\n') 기준으로 잘라내어 리스트를 만듭니다.(`strip()` 함수는 문자열 양 끝의 불필요한 공백문자를 잘라내 줍니다.) 그 결과는 다음과 같은 리스트가 됩니다.

```['bass', 'brass', 'flute', 'guitar', 'keyboard', 'mallet', 'organ', 'reed', 'string', 'synth_lead', 'vocal']```

이제 이 리스트에서 각 악기 종류의 위치가 해당 악기 종류를 나타내는 숫자가 됩니다.

```all_files = os.listdir(part + '/audio')```
`os.listdir`은 주어진 경로 아래에 있는 파일 목록을 반환합니다. 우리는 'train/audio'와 같이 `part + '/audio'` 아래에 음원 파일들이 있으므로 all_files에는 음원 파일들의 목록이 저장됩니다.

```for f in all_files:
  if f[-4:] == '.wav':```

`all_files`에 들어있는 각 파일에 대해서 정보를 처리하고 파일 목록과 레이블 목록에 추가합니다. 그런데, 'audio' 디렉토리 아래에 있는 파일 중 '.wav'파일이 아닌 파일도 있을 수 있으므로 (NSynth 데이터셋 자체에는 없지만 시스템에서 숨겨진 파일을 생성할 수 있습니다.) 파일 이름의 맨 뒤 네 글자가 '.wav'로 끝나는 경우에만 작업을 진행하도록 합니다.

```files.append(f[:-4])```

files에 현재 파일을 추가합니다. 이 때, 다음 작업의 편의를 위해 뒤의 네 글자('.wav')를 제외하고 저장합니다.

```label = f.split('_')[0]```

`split` 함수는 문자열을 인자를 기준으로 나눈 리스트를 반환합니다. 예를 들어 다음과 같은 경우,

```words = 'bass_electronic_044-046-075'.split('_')```

`words`에는 `['bass', 'electronic', '044-046-075']`가 저장됩니다. 즉 `f.split('_')[0]`은 파일명을 '_'를 기준으로 잘랐을 때의 첫 번째 단어가 됩니다. 대부분의 경우 이 첫 단어가 악기의 종류를 나타냅니다.

```
if label == 'synth':
  label = 'synth_lead'
```

하지만 레이블 중 'synth_lead'는 두 단어로 이루어져 첫 단어만 잘랐을 때는 'synth'가 됩니다. 이런 경우 'synth'를 'synth_lead'로 바꿔줍니다.

```labels.append(label_list.index(label))```

`index()` 함수를 이용해 `label_list`에서 `label`이 몇 번째에 있는지를 알아낼 수 있습니다. 위에서 확인한 대로 'bass'의 경우 `0`, 'brass'의 경우 `1`이 반환되어 `labels`에 추가되게 됩니다.

```
file_out = open(part + '_samples.txt', 'w')
for f in files:
  file_out.write(f + '\n')
file_out.close()
```

`samples`와 `labels`를 파일에 저장합니다. 이 때, 한 줄에 한 샘플이 들어가도록 샘플 파일 명에 줄바꿈('\n')을 붙여줍니다.

이제 `gather_information.py` 파일을 파이썬으로 실행하면 6개의 `txt` 파일이 생긴 것을 확인할 수 있습니다.

### 스펙트럼 추출하기

이제 'wav'파일을 읽고 스펙트럼을 추출해 보도록 하겠습니다.

`extract_spectrum.py` 파일을 만듭니다. 이 파일에서는 다음과 같은 일들을 할 것입니다.

- 오디오 파일 목록 불러오기
- 각 파일을 읽어서 스펙트럼으로 변환
- 정규화를 위한 통계 누적
- 변환된 파일을 저장하기

파일의 내용은 다음과 같습니다.

```
import numpy # 수치 연산에 이용
import librosa # 음원 파일을 읽고 분석하는 데 이용
import os # 디렉토리 생성 등 시스템 관련 작업
import os.path # 특정 경로가 존재하는지 파악하기 위해 필요

sequence_length = 251
feature_dimension = 513

def extract_spectrum(part):
  sample_files = open(part + '_samples.txt').read().strip().split('\n') # 샘플 목록을 읽어옵니다.
  if part == 'train': # 'train'인 경우에는 평균과 표준편차를 구해야 합니다.
    data_sum = numpy.zeros((sequence_length, feature_dimension)) # 합계를 저장할 변수를 만듭니다.
    data_squared_sum = numpy.zeros((sequence_length, feature_dimension)) # 제곱의 합을 저장할 변수입니다.
  if not os.path.exists(part+'/spectrum/'): # 'spectrum' 디렉토리가 존재하지 않으면 만들어 줍니다.
    os.mkdir(part+'/spectrum/')
  for f in sample_files:
    print('%d/%d: %s'%(sample_files.index(f), len(sample_files), f)) # 현재 진행상황을 출력합니다.
    y, sr = librosa.load(part+'/audio/'+f+'.wav', sr=16000) # librosa를 이용해 샘플 파일을 읽습니다.
    D = librosa.stft(y, n_fft=1024, hop_length=256).T # short-time Fourier transform을 합니다.
    mag, phase = librosa.magphase(D) # phase 정보를 제외하고, 세기만 얻습니다.
    S = numpy.log(1 + mag * 1000) # 로그형태로 변환합니다.
    if part == 'train': # 'train'인 경우 합계와 제곱의 합을 누적합니다.
      data_sum += S
      data_squared_sum += S ** 2
    numpy.save(part+'/spectrum/'+f+'.npy', S) # 현재 샘플의 스펙트럼을 저장합니다.
  if part == 'train': # 모든 파일의 변환이 끝난 후에, 'train'인 경우 평균과 표준편차를 저장합니다.
    data_mean = data_sum / len(sample_files)
    data_std = data_squared_sum / len(sample_files) - data_mean ** 2
    numpy.save('data_mean.npy', data_mean)
    numpy.save('data_std.npy', data_std)

if __name__ == '__main__':
  for part in ['train', 'valid', 'test']:
    extract_spectrum(part)

```

스펙트럼은 어떤 신호를 소리의 높이에 따라 분해했을 때, 각 높이에 해당하는 성분이 얼마나 강한지 나타냅니다. 이 때, 신호의 높이를 주파수라고 하고, 낮은 소리는 낮은 주파수 성분이 강하게, 높은 소리는 높은 주파수의 성분이 강하게 나옵니다. 이를 통해 어떤 소리의 특징을 보다 알기 쉽게 표현할 수 있습니다. 스펙트럼을 짧은 시간마다 반복해서 추출한 것을 STFT(short-time Fourier transform)이라고 합니다.

많은 경우에 음원 자체(raw wave)를 이용하는 것보다 스펙트럼과 같이 분석된 정보를 이용하는 것이 기계학습에 더 효율이 좋다고 알려져 있습니다. 이 연습에서는 매 스펙트럼마다 1024 개의 샘플(기록)을 사용하고, 다음 스펙트럼은 256 샘플만큼 뒤에서 다시 추출합니다. 즉, 모든 스펙트럼은 자신과 인접한 스펙트럼과 768개의 샘플을 공유하게 됩니다.

샘플링 레이트(samping rate)는 소리를 녹음할 때 얼마나 자주 기록하는지를 의미합니다. NSynth 데이터셋의 경우 16,000 Hz로 일초에 16,000번 기록했다는 의미입니다. 전체 샘플 길이는 4초이니 각 샘플은 총 16,000 * 4 = 64,000개의 샘플을 가지고 있습니다.

전체 64,000개의 샘플을 256개씩 넘어가면서 스펙트럼을 추출하기 때문에 방식에 따라 250개 내외의 스펙트럼이 나오게 됩니다. 여기서 사용한 `librosa.stft`의 경우 총 251개가 나옵니다.

```
sequence_length = 251
feature_dimension = 513
```

스펙트럼에 1024개의 샘플을 넣었기 때문에 결과로 나오는 한 스펙트럼은 513개의 값을 가집니다. 한 파일에서 STFT를 추출하면 (251, 513)의 크기를 가진 행렬이 나오게 됩니다. 위에서는 미리 두 숫자를 정의했습니다.

```
if part == 'train': # 'train'인 경우에는 평균과 표준편차를 구해야 합니다.
    data_sum = numpy.zeros((sequence_length, feature_dimension))
    data_squared_sum = numpy.zeros((sequence_length, feature_dimension))
```

'train' 셋의 경우 정규화(normalization)를 위해 평균(mean)과 표준편차(standard deviation, std)를 구해야 합니다. 정규화는 자료의 전체 혹은 일부의 평균과 표준편차를 일정한 값으로 조정해 주는 것으로 일반적으로 평균은 0으로, 표준편차는 1로 변환해줍니다. 자료가 정규화되면 학습에 사용하는 다양한 함수들이 더 효과적인 범위에서 작동하게 됩니다.

모든 자료를 동시에 로드할 수 있다면 `numpy.mean(), numpy.std()` 함수를 통해 간단히 평균과 표준편차를 구할 수 있지만
이 경우는 자료가 매우 크므로 직접 통계적인 계산을 해야 합니다. 평균은 `전체 데이터의 합 / 전체 데이터의 수`로 구할 수 있고, 표준편차는 `전체 데이터의 제곱의 평균 - 전체 데이터의 평균의 제곱`으로 구할 수 있습니다. 전체 데이터의 수는 이미 알고 있으므로 합과 제곱의 합을 저장할 변수를 만들어 줍니다.

```
print('%d/%d: %s'%(sample_files.index(f), len(sample_files), f))
```

프로그램이 실행되는 중에 진행 상황에 관해 적절한 정보를 주도록 만들면 어떤 문제가 생겼을 때 쉽게 대처할 수도 있고, 실행 시간이 매우 긴 프로그램의 경우 남은 시간을 추측할 수도 있습니다. 위의 코드는 예를 들어 현재 어떤 파일을 처리하고 있고, 전체의 몇번째 파일인지를 출력해줍니다. 만약 특정 파일을 처리하다 에러가 났다면 마지막으로 처리하던 파일의 이름을 알 수 있습니다.

```
y, sr = librosa.load(part+'/audio/'+f+'.wav', sr=16000)
```

`librosa.load()` 함수는 음원 파일을 읽어 샘플을 리스트로 반환해 줍니다. 샘플링 레이트는 위에서 언급한 대로 16,000을 사용합니다.

```
D = librosa.stft(y, n_fft=1024, hop_length=256).T
```

`librosa.stft()` 함수를 이용해 스펙트럼을 얻습니다. `librosa`는 `(feature_dim, sequence_length)` 형태로 반환하므로 `.T`를 이용해 `(sequence_length, feature_dim)`의 형태로 뒤집어 줍니다.

```
mag, phase = librosa.magphase(D)
S = numpy.log(1 + mag * 1000)
```





파일 작성 후 파이썬을 이용해 실행하면 각 부분 폴더 아래에 'spectrum' 폴더가 생성되고, 각 음원파일의 스펙트럼이 'npy' 파일로 저장됩니다.

## 모델 생성하고 학습시키기

그럼 이제 모델을 만들고, 수집된 데이터를 이용해서 학습을 시켜보겠습니다.

`train.py` 파일을 만들고, 상단에 다음과 같은 항목들을 작성해 주세요.

```
import os
import tensorflow as tf
import numpy
import random

n_labels = 11
batch_size = 3
sequence_length = 200
feature_dimension = 512
```

### 데이터 준비하기

```
def prepare_data():
  train_samples = open('train_samples.txt').read().strip().split('\n')
  train_labels = [int(label) in label for open('train_labels.txt').read().strip().split('\n')]

  valid_samples = open('valid_samples.txt').read().strip().split('\n')
  valid_labels = [int(label) in label for open('valid_labels.txt').read().strip().split('\n')]

  test_samples = open('test_samples.txt').read().strip().split('\n')
  test_labels = [int(label) in label for open('test_labels.txt').read().strip().split('\n')]

  data_mean = numpy.load('mean.npy')
  data_std = numpy.load('std.npy')

  return train_samples, train_labels, valid_samples, valid_labels, test_samples, test_labels, data_mean, data_std

```

### 데이터 읽어오기

```
def get_random_sample(part, samples, labels, mean, std):
  i = random.randrange(len(samples))
  spectrum = numpy.load(part+'/spectrum/'+samples[i]+'.npy')
  spectrum = (spectrum - mean) / (std + 0.0001)
  return spectrum, labels[i]

def get_random_batch(part, samples, labels, mean, std):
  X = numpy.zeros((batch_size, sequence_length, feature_dimension, 1))
  Y = numpy.zeros((batch_size,))
  for b in batch_size:
    s, l = get_random_sample(part, samples, labels, mean, std)
    X[b, :, :, 0] = s[:sequence_length, :feature_dimension]
    Y[b] = l
  return X, Y
```

### 모델 정의하기
```
def get_model(X, is_training):
  conv1 = tf.contrib.layers.conv2d(X, 64, (3, sequence_length), normalizer_fn=tf.contrib.layers.batch_norm, normalizer_param={is_training:is_training})
  pool1 = tf.contrib.layers.max_pool2d(conv1, (3, 1), stride=3)
  conv2 = tf.contrib.layers.conv2d(pool1, 64, (3, sequence_length), normalizer_fn=tf.contrib.layers.batch_norm, normalizer_param={is_training:is_training})
  pool2 = tf.contrib.layers.max_pool2d(conv2, (3, 1), stride=3)
  conv3 = tf.contrib.layers.conv2d(pool2, 128, (3, sequence_length), normalizer_fn=tf.contrib.layers.batch_norm, normalizer_param={is_training:is_training})
  pool3 = tf.contrib.layers.max_pool2d(conv3, (3, 1), stride=3)
  conv4 = tf.contrib.layers.conv2d(pool3, 128, (3, sequence_length), normalizer_fn=tf.contrib.layers.batch_norm, normalizer_param={is_training:is_training})
  pool4 = tf.contrib.layers.max_pool2d(conv4, (3, 1), stride=3)
  flatten = tf.contrib.layers.flatten(pool4)
  fc1 = tf.contrib.layers.fully_connected(flatten, 256)
  fc2 = tf.contrib.layers.fully_connected(fc1, 256)
  fc3 = tf.contrib.layers.fully_connected(fc2, n_labels)
  return fc3

def get_loss(logits, labels):
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
  return tf.reduce_mean(cross_entropy)
```

### 학습시키기
```
def train():
  train_samples, train_labels, valid_samples, valid_labels, test_samples, test_labels, data_mean, data_std = prepare_data()
  with tf.Graph().as_default():
    X = tf.placeholder(tf.float32, shape(batch_size, sequence_length, feature_dimension, 1))
    Y = tf.placeholder(tf.int32, shape=(batch_size,))
    phase_train = tf.placeholder(tf.bool)
    logits = get_model(X, phase_train)
    loss = get_loss(logits, Y)
    evaluation = tf.reduce_sum(tf.cast(tf.nn.in_top_k(logits, Y, 1), tf.int32))
    optimizer = tf.train.AdadeltaOptimizer(0.01)
    train_op = optimizer.minimize(loss)
    sess = tf.Session()
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    accumulated_loss =m
    for step in range(10000):
      x, y = get_random_batch('train', train_samples, train_labels, data_mean, data_std)
      _, loss_value = sess.run([train_op, loss], feed_dict={phase_train:True, X:x, Y:y})
      accumulated_loss += loss_value
      if (step + 1) % 10 == 0:
        print('step %d, loss = %.5f'%(step+1, accumulated_loss / 10))
        accumulated_loss = 0
      if (step + 1) % 100 == 0:
        correct = 0;
        for i in range(10):
          x, y = get_random_batch('valid', valid_samples, valid_labels, data_mean, data_std)
          corr = sess.run([evaluation], feed_dict={phase_train:False, X:x, Y:y})
          correct += corr[0]
        print('step %d, valid accuracy = %.2f'%(step+1, 100 * correct / 10 / batch_size))
```


## t-SNE
## 시각화