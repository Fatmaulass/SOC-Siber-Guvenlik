#🛡️Makine Öğrenmesi ile Ağ Trafiği Analizi

Bu sistem, modern ağ güvenliği ihtiyaçlarını karşılamak üzere, klasik "imza tabanlı" (sadece bilinen virüsleri tanıyan) sistemlerin ötesine geçer. Veri odaklı bir yaklaşım kullanarak ağ trafiğindeki anormal davranışları (anomali) yakalar.

1) Veri Seti ve Ön İşleme (Data Engineering)
Proje, PCAP dosyalarından elde edilen 49 farklı özniteliğe sahip UNSW-NB15 veri setini temel alır.
* Sınıf Dengeleme (SMOTE): Siber güvenlik verilerinde "normal" trafik miktarı "saldırı" miktarından çok daha fazladır. Bu dengesizlik modelin yanılmasına sebep olur. Projede SMOTE (Sentetik Azınlık Aşırı Örnekleme Tekniği) kullanılarak, azınlıkta kalan saldırı türleri yapay olarak artırılmış ve modelin nadir saldırıları yakalama hassasiyeti (recall) yükseltilmiştir.
* Öznitelik Mühendisliği: Gereksiz kolonlar (id, timestamp vb.) temizlenmiş, kategorik veriler One-Hot Encoding ile sayısal forma getirilmiş ve tüm veriler StandardScaler ile normalize edilmiştir.

2) Model Mimarisi: Neden XGBoost?
Sistemde tercih edilen XGBoost (Extreme Gradient Boosting), karar ağaçları tabanlı bir topluluk (ensemble) öğrenme algoritmasıdır.
* Yüksek Performans: Yapılandırılmış verilerde derin öğrenme modellerine kıyasla daha hızlıdır ve daha az işlemci gücüyle daha yüksek doğruluk sunar.
* Optimizasyon: Projede GridSearchCV kullanılarak; öğrenme hızı (learning_rate), ağaç sayısı (n_estimators) ve derinlik (max_depth) gibi parametreler en iyi performans için otomatik olarak ayarlanmıştır.

3) İki Aşamalı Tehdit Analizi
Sistem iki farklı modelin entegre çalışmasıyla karar verir:
* Aşama 1 (Binary): Trafik anlık olarak "Normal" mi yoksa "Saldırı" mı diye ayrılır. Bu aşamada %90 doğruluk (accuracy) ve 0.92 F1-skoru elde edilmiştir.
* Aşama 2 (Multi-Class): Eğer bir saldırı tespit edilirse, sistem devreye girer ve bunun hangi kategoride olduğunu (DoS, Fuzzers, Exploits, Backdoor, Reconnaissance vb.) belirler.

4) Açıklanabilir Yapay Zeka (SHAP Entegrasyonu)
Yapay zekanın neden "bu bir saldırıdır" dediğini bilmek bir güvenlik uzmanı için hayatidir.
* Kara Kutu Sorununu Aşma: Projede kullanılan SHAP analizi, modelin kararına hangi özelliğin ne kadar katkı sağladığını gösterir.
* Kritik Parametreler: Analiz sonucunda sbytes (kaynak byte miktarı) ve sttl (yaşam süresi) gibi değerlerin saldırı tespitinde en belirleyici değişkenler olduğu kanıtlanmıştır.

5) Canlı SOC Dashboard (Streamlit)
Analistlerin kullanımı için geliştirilen arayüz şunları sağlar:
* Zaman Çizelgesi: Saldırı yoğunluğunun zaman içindeki değişimini takip etme.
* Dağılım Analizi: Hangi saldırı türünün ağda ne kadar yer kapladığını gösteren dinamik grafikler.
* Log ve Raporlama: Tespit edilen tehditleri IP bazlı olarak listeleme ve CSV olarak dışa aktarma.

🚀 Kurulum ve Çalıştırma Rehberi
1. Hazırlık
Depoyu klonlayın ve gerekli kütüphaneleri yükleyin:
* git clone https://github.com/Fatmaulass/SOC-Siber-Guvenlik.git
* cd SOC-Siber-Guvenlik
* pip install -r requirements.txt

2. Modellerin Eğitilmesi
Modeller (.pkl dosyaları) repo boyutunu optimize etmek için dahil edilmemiştir. Uygulamayı çalıştırmadan önce modelleri oluşturmak için eğitim scriptlerini şu sırayla çalıştırın. Bu işlem sonucunda gerekli .pkl dosyaları otomatik olarak oluşturulacaktır.
* Binary Sınıflandırma Modeli (Normal/Saldırı):   
python model_egitme_gridsearch.py
* Multiclass Sınıflandırma Modeli (Saldırı Türleri):   
python model_egitme_tur_gridsearchcv.py

3. Arayüzü Başlatma
Modeller oluştuktan sonra SOC Dashboard'u şu komutla başlatabilirsiniz:
* streamlit run app.py
