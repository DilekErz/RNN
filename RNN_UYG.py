# DİLEKERSÖZ NOTE:

import numpy as np 

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense 
#bizim asıl RNN katmanımız Dense → en sonunda pozitif mi negatif mi? kararını verecek
from tensorflow.keras.preprocessing.text import Tokenizer #kelimelere numara verecek
from tensorflow.keras.preprocessing.sequence import pad_sequences #cümlelerin uzunluklarını eşitleyecek


sentences = [
    "Dilek dedi ki Bu film çok kötü",
     "Bu film çok kötü",
    "Hiç beğenmedim",
    "Film çok sıkıcıydı",
    "Oyunculuklar çok kötüydü",
"Bu film berbattı",
    "Film hiç güzel değildi",
    "Çok kötü bir filmdi",
    "Senaryo çok kötüydü",
    "Bu filmi sevmedim",
    "Film çok başarısızdı",


    "Dilek dedi ki Bu film çok güzel",
    "Bu film çok güzel",
    "Çok beğendim",
    "Film harikaydı",
    "Oyunculuklar çok başarılıydı",
    "Bu film mükemmeldi",
    "Film çok güzeldi",
    "Çok güzel bir filmdi",
    "Senaryo çok başarılıydı",
    "Bu filmi çok sevdim",
    "Film gerçekten harikaydı"
]

# 0 = Negatif, 1 = Pozitif
# Tokenizer önce kelimelere indeks numaraları veriyor
labels = np.array([
    0,0,0,0,0,0,0,0,0,0,0,
    1,1,1,1,1,1,1,1,1,1,1
])

tokenizer = Tokenizer(num_words=50, oov_token="<OOV>")

tokenizer.fit_on_texts(sentences)

print(tokenizer.word_index )
print("\nKELİME İNDEKSLERİ:")

for kelime, numara in tokenizer.word_index.items():
    print(kelime, "=", numara)


sequences = tokenizer.texts_to_sequences(sentences) #  burda ben kelimelere numara vermiştim Şimdi bu numaraları kullanarak bütün cümleleri sayılara dönüştüreyim

print("\nCÜMLELERİN SAYISAL HALİ:")

for cumle, sequence in zip(sentences, sequences):
    print(cumle)
    print(sequence)
    print()

# Burada önemli bir şeyi fark et : Dilek dedi ki Bu film çok kötü→ 7 elemanCümlelerin uzunlukları farklı İşte şimdi padding yapmamızın sebebi bu

padded_sequences = pad_sequences(sequences, padding="post") #padding="post":emek Eksik olan yerlere 0 ekle ama sona ekle
#Örneğin en uzun cümlemiz 7 kelimeyse:
# Hiç beğenmedim
# [10, 11]
# şuna dönüşür:
# [10, 11, 0, 0, 0, 0, 0]

print("\nPADDING SONRASI:")

for cumle, padded in zip(sentences, padded_sequences):
    print(cumle)
    print(padded)
    print()


    # RNN MODELİNİ OLUŞTURUYORUZ:
max_length = padded_sequences.shape[1]
model_rnn = Sequential([ #Katmanların sırayla çalışacağını söylüyor Ama 4, 5, 6 gibi sayılar sadece kelime numaraları. Mesela kötü = 9 olması kötü kelimesinin matematiksel anlamının 9 olduğu anlamına gelmiyor Embedding bu kelime numaralarının her birini 8 sayılık öğrenilebilir bir vektöre dönüştürüyor
    
    Embedding(input_dim=50, output_dim=8, mask_zero=True), #mask_zero=True demek:Padding için eklediğim 0'ları kelime olarak değerlendirme
    SimpleRNN(16), #Her kelimeyi işlerken önceki kelimelerden elde ettiği hidden state (gizli durum) bilgisini de kullanacak 16 RNN birimi / 16 boyutlu hidden state kullanacağımız anlamına geliyo
    Dense(1, activation="sigmoid")#Sigmoid nedeniyle sonuç 0–1 arasında olacak
])
# input_dim=50 → sözlük için 50 indekslik alan.
# output_dim=8 → her kelimeyi 8 boyutlu bir vektörle temsil et
# kötü
#  ↓
#  9
#  ↓
# Embedding
#  ↓
# [0.12, -0.31, 0.44, ...]
#        8 değer
model_rnn.build(input_shape=(None, max_length))
# None → Kaç tane cümle geleceği değişebilir
# 7    → Her cümle 7 sayıdan oluşacak
model_rnn.summary()

model_rnn.compile( #modelin nasıl öğreneceğini ayarlıyor
    loss="binary_crossentropy", #pozitif/negatif ikili sınıflandırma için hata fonksiyonu YANİ ASLINDA Modelin tahmini ile gerçek cevap arasındaki hatayı hesaplıyor
    optimizer="adam", #Bu hataya bakıp modeldeki ağırlıkları nasıl değiştireceğine karar veriyor
    metrics=["accuracy"]#Kaç tanesini doğru bildiğini göster
)

model_rnn.fit(
    #eğitimi başlatıyor
    padded_sequences, #bizim giriş verimiz
    labels, #labels ise doğru cevaplarımız
    epochs=50# 50 kez baştan sona gör demek
)
# İSTEDİĞİİMİZ HENEL DAVRANIŞ Accuracy ↑, Loss ↓

#  modeli eğittik şimdi Şimdi eğitimde olmayan bir cümle yazıp modele soralım: bu film güzel Model bunu pozitif mi negatif mi bulacak

# YENİ BİR CÜMLE İLE TAHMİN TESTİ YAP:

test_cumleleri = [
    "Bu film çok güzel",
    "Bu film güzel",
    "Bu film çok kötü",
    "Bu film kötü"
]

for cumle in test_cumleleri:

    # Cümleyi sayılara dönüştür
    sequence = tokenizer.texts_to_sequences([cumle])

    # Uzunluğu eğitim verileriyle aynı yap
    padded = pad_sequences(
        sequence,
        maxlen=max_length,
        padding="post"
    )

    # RNN modeline tahmin yaptır
    tahmin = model_rnn.predict(padded, verbose=0)[0][0]

    print("\nCümle:", cumle)
    print("Tahmin değeri:", tahmin)

    if tahmin >= 0.5:
        print("Sonuç: POZİTİF")
    else:
        print("Sonuç: NEGATİF")


