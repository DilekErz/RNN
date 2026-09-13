# RNN ile Türkçe Duygu Analizi

Bu proje, yazılım geliştirme stajım kapsamında **Recurrent Neural Network (RNN)** yapısını öğrenmek ve uygulamak amacıyla geliştirilmiştir.

Projede TensorFlow/Keras kullanılarak Türkçe film yorumlarının **pozitif** veya **negatif** olarak sınıflandırılması amaçlanmıştır.

Model oluşturulurken metin verileri önce sayısal hale getirilmiş, cümle uzunlukları eşitlenmiş ve ardından **Embedding**, **SimpleRNN** ve **Dense** katmanlarından oluşan bir sinir ağı modeli geliştirilmiştir.

---

## Projenin Amacı

Bu çalışmanın temel amacı, RNN yapısının sıralı veriler ve özellikle metin verileri üzerinde nasıl çalıştığını uygulamalı olarak incelemektir.

Model iki farklı sınıfı tahmin etmektedir:

- `0` → Negatif
- `1` → Pozitif

Örnek negatif yorumlar:

```text
Dilek dedi ki Bu film çok kötü
Bu film çok kötü
Hiç beğenmedim
Film çok sıkıcıydı
Oyunculuklar çok kötüydü
Bu film berbattı
```

Örnek pozitif yorumlar:

```text
Dilek dedi ki Bu film çok güzel
Bu film çok güzel
Çok beğendim
Film harikaydı
Oyunculuklar çok başarılıydı
Bu film mükemmeldi
```

---

## Kullanılan Teknolojiler

Projede aşağıdaki teknolojiler kullanılmıştır:

- Python
- NumPy
- TensorFlow
- Keras
- SimpleRNN
- Embedding
- Tokenizer
- Padding
- NLP
- Sentiment Analysis

---

## RNN Nedir?

**Recurrent Neural Network (RNN)**, özellikle sıralı verilerin işlenmesi için kullanılan bir yapay sinir ağı mimarisidir.

Normal ileri beslemeli sinir ağlarından farklı olarak RNN, önceki adımlarda elde ettiği bilgileri bir sonraki adıma aktarabilir.

Bu özellik sayesinde:

- Metin
- Zaman serileri
- Ses verileri
- Ardışık sensör verileri

gibi sıralamanın önemli olduğu problemlerde kullanılabilir.

Bu projede RNN, cümledeki kelimeleri sırayla inceleyerek film yorumunun olumlu veya olumsuz olduğunu öğrenmektedir.

---

## Veri Seti

Projede küçük ve manuel olarak oluşturulmuş Türkçe bir film yorumu veri seti kullanılmıştır.

Toplamda:

- 11 negatif yorum
- 11 pozitif yorum

olmak üzere **22 cümle** bulunmaktadır.

Etiketleme işlemi şu şekilde yapılmıştır:

```python
labels = np.array([
    0,0,0,0,0,0,0,0,0,0,0,
    1,1,1,1,1,1,1,1,1,1,1
])
```

Burada:

```text
0 = Negatif
1 = Pozitif
```

---

## Tokenization

Sinir ağları doğrudan kelimeler üzerinde işlem yapamadığı için metinlerin önce sayısal verilere dönüştürülmesi gerekir.

Bu amaçla Keras `Tokenizer` kullanılmıştır.

```python
tokenizer = Tokenizer(
    num_words=50,
    oov_token="<OOV>"
)
```

Ardından tokenizer veri setindeki cümlelere göre eğitilmiştir:

```python
tokenizer.fit_on_texts(sentences)
```

Her kelimeye bir indeks numarası atanmıştır.

Örneğin:

```text
film = 2
çok = 3
bu = 4
güzel = 5
kötü = 6
```

Bu numaralar kelimelerin matematiksel anlamları değildir. Sadece kelimeleri birbirinden ayırmak için kullanılan indekslerdir.

---

## Cümlelerin Sayısal Hale Dönüştürülmesi

Tokenizer oluşturulduktan sonra bütün cümleler sayı dizilerine dönüştürülmüştür.

```python
sequences = tokenizer.texts_to_sequences(sentences)
```

Örneğin:

```text
Bu film çok güzel
```

cümlesi şu tarz bir yapıya dönüşebilir:

```text
[4, 2, 3, 5]
```

Bu sayılar tokenizer tarafından kelimelere verilen indeksleri temsil etmektedir.

---

## Padding

Cümlelerin uzunlukları birbirinden farklı olduğu için sinir ağına verilmeden önce aynı uzunluğa getirilmiştir.

Bunun için:

```python
padded_sequences = pad_sequences(
    sequences,
    padding="post"
)
```

kullanılmıştır.

Örneğin:

```text
Hiç beğenmedim
```

sayısal olarak:

```text
[10, 11]
```

şeklinde ise padding işleminden sonra:

```text
[10, 11, 0, 0, 0, 0, 0]
```

şeklinde olabilir.

`padding="post"` kullanıldığı için eksik değerler dizinin sonuna eklenmektedir.

---

## Embedding Katmanı

Modelin ilk katmanı `Embedding` katmanıdır.

```python
Embedding(
    input_dim=50,
    output_dim=8,
    mask_zero=True
)
```

Kelime indeksleri doğrudan RNN'e verilmek yerine Embedding katmanı tarafından öğrenilebilir vektörlere dönüştürülmektedir.

Örneğin:

```text
kötü
 ↓
6
 ↓
Embedding
 ↓
[0.12, -0.31, 0.44, ...]
```

Bu projede her kelime **8 boyutlu bir vektör** ile temsil edilmektedir.

### `mask_zero=True`

Padding sırasında eklenen `0` değerlerinin gerçek bir kelime olarak değerlendirilmemesini sağlar.

---

## SimpleRNN Katmanı

Modelin temel RNN katmanı:

```python
SimpleRNN(16)
```

şeklinde oluşturulmuştur.

Buradaki `16`, RNN'in **16 boyutlu hidden state** kullanacağını belirtmektedir.

RNN cümledeki kelimeleri sırayla işlerken önceki kelimelerden elde ettiği bilgiyi sonraki kelimelere aktarır.

Örneğin:

```text
Bu
 ↓
film
 ↓
çok
 ↓
güzel
 ↓
RNN
```

RNN yalnızca son kelimeye değil, önceki kelimelerden gelen bilgilere de göre değerlendirme yapmaktadır.

---

## Model Mimarisi

Projede kullanılan model:

```python
model_rnn = Sequential([
    Embedding(
        input_dim=50,
        output_dim=8,
        mask_zero=True
    ),

    SimpleRNN(16),

    Dense(
        1,
        activation="sigmoid"
    )
])
```

Model toplam üç temel katmandan oluşmaktadır:

```text
Metin
  ↓
Tokenizer
  ↓
Padding
  ↓
Embedding
  ↓
SimpleRNN
  ↓
Dense
  ↓
Pozitif / Negatif
```

---

## Dense ve Sigmoid Katmanı

Son katmanda:

```python
Dense(1, activation="sigmoid")
```

kullanılmıştır.

Sigmoid aktivasyon fonksiyonu model çıktısını **0 ile 1 arasında** bir değere dönüştürmektedir.

Örneğin:

```text
0.15
```

çıktısı negatif sınıfa yakınken,

```text
0.91
```

çıktısı pozitif sınıfa yakındır.

Projede sınıflandırma sınırı:

```python
0.5
```

olarak belirlenmiştir.

```python
if tahmin >= 0.5:
    print("Sonuç: POZİTİF")
else:
    print("Sonuç: NEGATİF")
```

---


### Binary Crossentropy

İki sınıflı sınıflandırma problemlerinde kullanılan hata fonksiyonudur.

Modelin tahmini ile gerçek etiket arasındaki farkı hesaplamaktadır.

### Adam Optimizer

Hesaplanan hataya göre model ağırlıklarının güncellenmesini sağlar.

### Accuracy

Modelin doğru tahmin ettiği örneklerin oranını gösterir.

---

## Model Eğitimi

Model:

```python
model_rnn.fit(
    padded_sequences,
    labels,
    epochs=50
)
```

ile eğitilmiştir.

`epochs=50`, modelin eğitim veri setini toplam 50 kez görmesi anlamına gelmektedir.

Eğitim sırasında genel olarak beklenen davranış:

```text
Accuracy ↑
Loss ↓
```

şeklindedir.

---

## Yeni Cümlelerle Tahmin

Model eğitildikten sonra eğitim sırasında doğrudan kullanılmayan farklı cümlelerle test edilmiştir.

```python
test_cumleleri = [
    "Bu film çok güzel",
    "Bu film güzel",
    "Bu film çok kötü",
    "Bu film kötü"
]
```

Yeni bir cümle önce tokenizer ile sayısal hale getirilmektedir:

```python
sequence = tokenizer.texts_to_sequences([cumle])
```

Daha sonra eğitim verileri ile aynı uzunluğa getirilir:

```python
padded = pad_sequences(
    sequence,
    maxlen=max_length,
    padding="post"
)
```

Son olarak modele tahmin yaptırılır:

```python
tahmin = model_rnn.predict(
    padded,
    verbose=0
)[0][0]
```

---

## Tahmin Mantığı

Modelin ürettiği olasılık değerine göre sınıflandırma yapılmaktadır.

```python
if tahmin >= 0.5:
    print("Sonuç: POZİTİF")
else:
    print("Sonuç: NEGATİF")
```

Örnek:

```text
Cümle: Bu film çok güzel
Tahmin değeri: 0.93
Sonuç: POZİTİF
```

```text
Cümle: Bu film çok kötü
Tahmin değeri: 0.08
Sonuç: NEGATİF
```

Tahmin değerleri her eğitim çalıştırmasında modelin başlangıç ağırlıklarına bağlı olarak değişiklik gösterebilir.

---

## Model Sonuçları

SimpleRNN modeli **50 epoch** boyunca eğitilmiştir. Eğitim sürecinde modelin doğruluk oranının arttığı, loss değerinin ise azaldığı gözlemlenmiştir.

### Eğitim Sonucu

50. epoch sonunda elde edilen eğitim sonuçları:

```text
Accuracy: 1.0000
Loss: 0.5200
```

Model, kullanılan **22 cümlelik eğitim veri seti üzerinde %100 eğitim doğruluğuna** ulaşmıştır.

> **Not:** Bu değer eğitim verileri üzerinden elde edilen doğruluk oranıdır. Projede ayrı bir test veri seti kullanılmamış, model eğitildikten sonra örnek cümleler üzerinden tahminler gerçekleştirilmiştir.

---

### Örnek Tahmin Sonuçları

Eğitim tamamlandıktan sonra model farklı cümleler üzerinde çalıştırılmıştır.

| Test Cümlesi | Tahmin Değeri | Sınıflandırma |
|---|---:|---|
| `Bu film çok güzel` | 0.5894 | **POZİTİF** |
| `Bu film güzel` | 0.5279 | **POZİTİF** |
| `Bu film çok kötü` | 0.4476 | **NEGATİF** |
| `Bu film kötü` | 0.3868 | **NEGATİF** |

Modelde **0.5** sınıflandırma eşiği kullanılmıştır:

```text
Tahmin >= 0.5  →  POZİTİF
Tahmin < 0.5   →  NEGATİF
```

Test edilen dört örnek cümle de beklenen sınıfa atanmıştır.

---

## Proje Dosyası

```text
RNN/
│
├── RNN_UYG.py
└── README.md
```

`RNN_UYG.py` dosyası içerisinde:

- Veri setinin oluşturulması
- Etiketlerin hazırlanması
- Tokenization
- Sequence oluşturma
- Padding
- Embedding
- SimpleRNN modelinin oluşturulması
- Model eğitimi
- Yeni cümlelerle tahmin

işlemleri gerçekleştirilmektedir.

---

## Kurulum

Projeyi çalıştırmak için gerekli kütüphaneler:

```bash
pip install numpy tensorflow
```

---

## Çalıştırma

Projeyi klonladıktan sonra:

```bash
python RNN_UYG.py
```

komutu ile çalıştırabilirsiniz.

---

## Staj Kapsamında Kazanımlar

Bu uygulama ile staj kapsamında aşağıdaki konular uygulamalı olarak incelenmiştir:

- Recurrent Neural Network mantığı
- Sıralı veri işleme
- Doğal Dil İşleme
- Türkçe metin sınıflandırma
- Sentiment Analysis
- Tokenization
- Word Index
- Sequence dönüşümü
- Padding
- Embedding
- Hidden State
- SimpleRNN
- Binary Classification
- Sigmoid aktivasyon fonksiyonu
- Binary Crossentropy
- Adam optimizer
- Model eğitimi
- Yeni veriler üzerinde tahmin

---

## Sonuç

Bu proje kapsamında **SimpleRNN kullanılarak temel bir Türkçe duygu analizi uygulaması geliştirilmiştir.**

Metin verilerinin doğrudan sinir ağına verilemeyeceği için önce Tokenizer yardımıyla sayısal verilere dönüştürülmesi, ardından Padding ile eşit uzunlukta girişler oluşturulması incelenmiştir.

Embedding katmanı ile kelimeler öğrenilebilir vektörlerle temsil edilmiş ve SimpleRNN katmanı kullanılarak cümle içerisindeki sıralı bilgiler işlenmiştir.

Son olarak sigmoid aktivasyon fonksiyonuna sahip Dense katmanı ile film yorumlarının **pozitif veya negatif** olarak sınıflandırılması gerçekleştirilmiştir.

Bu çalışma sayesinde RNN mimarisinin temel çalışma mantığı ve doğal dil işleme problemlerinde nasıl kullanılabileceği uygulamalı olarak öğrenilmiştir.

---

## Geliştirici
**Dilek Ayça Ersöz**
