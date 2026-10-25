# ChatBot: Çoklu Sağlayıcı Destekli, Düşünce Akışlı ve Canlı Web Aramalı LLM Platformu

Bu proje; Node.js, npm paket yöneticisi veya Webpack/Vite gibi karmaşık JavaScript derleme zincirlerine ihtiyaç duymayan, doğrudan FastAPI arka ucu ve saf JavaScript (Vanilla JS) istemcisiyle geliştirilmiş tam teşekküllü, hafif ve yüksek performanslı bir yapay zeka arayüzüdür.

Sistem; Google Gemini, Groq ve OpenRouter servislerini tek bir çatı altında birleştirir. Modellerin akıl yürütme (reasoning / thinking) adımlarını gerçek zamanlı yakalayıp bağımsız bir düşünce akordeonunda sunar, DuckDuckGo üzerinden canlı web araştırması yaparak cevapları güncel kaynaklarla destekler ve yerel PDF ile görsel analizi yetenekleri sağlar.

---

## 1. Sistem Gereksinimleri ve Ön Koşullar

Bu uygulamanın kararlı, tam performanslı ve tüm asenkron iş parçacıklarıyla hatasız çalışabilmesi için aşağıdaki sistem ortamı şarttır:

* **İşletim Sistemi:** Windows 10 Pro veya üstü (Windows 10 Pro, Windows 11 Pro / Enterprise / Education). Windows Home sürümlerinde bazı ağ soket izinleri ve gelişmiş süreç yönetimleri kısıtlanabileceğinden Pro ve üzeri sürümler gereklidir.
* **Python Sürümü:** Python 3.15 (patch uygulanmış güncel sürüm: 3.15.0+). Uygulama, modern asenkron standartları (`asyncio.to_thread`, gelişmiş tip ipuçları ve optimize edilmiş bellek yönetimi) baz alacak şekilde yapılandırılmıştır.
* **Donanım:** Minimum 4 GB RAM ve standart internet bağlantısı (büyük modeller ve multimodal içerikler doğrudan bulut API'leri üzerinden işlendiği için yerel GPU gerektirmez).

---

## 2. Mimari Tasarım ve Mühendislik Felsefesi

Modern yapay zeka istemcilerinde genellikle React, Next.js veya Vue gibi hantal framework'ler tercih edilir. Ancak bu durum, yalnızca metin ve akış gösteren bir arayüz için yüzlerce megabaytlık `node_modules` bağımlılığı, yavaş derleme süreleri ve gereksiz bellek tüketimi yaratır.

ChatBot projesinde benimsenen temel mühendislik kararları şunlardır:

* **Sıfır Derleme İhtiyacı (Zero-Build Frontend):** Arayüz tek bir `templates/index.html` dosyasında yaşar. CSS ve modern ES6 JavaScript harici hiçbir paketleyici olmadan doğrudan tarayıcı motoru tarafından çalıştırılır. Sayfa boyutu son derece küçüktür, saniyenin altında açılır ve istemci kaynaklarını tüketmez.
* **Asenkron Python Backend:** Sunucu katmanında `FastAPI` ve `httpx.AsyncClient` kullanılarak tamamen engellemesiz (non-blocking) bir asenkron mimari kurulmuştur.
* **Server-Sent Events (SSE):** Çift yönlü WebSocket yerine, sunucudan istemciye tek yönlü veri akışı sağlayan `text/event-stream` protokolü tercih edilmiştir. Bu protokol bağlantı koptuğunda doğal olarak temizlenir, proxy arkasında sorunsuz çalışır ve ağ yükünü minimuma indirir.
* **Sıfır İstemci Sızıntısı:** API anahtarları istemci tarafında veya tarayıcı deposunda asla tutulmaz. Bütün sağlayıcı kimlik doğrulaması sunucunun yerel ortamında (`.env`) gerçekleşir.

---

## 3. Kod Tabanının Derinlemesine Analizi

```
chatbot_project/
├── app.py                # FastAPI çekirdeği, model yönlendirme, SSE akışı, araçlar
├── requirements.txt      # Python kütüphane bağımlılıkları
├── .env                  # Yerel API anahtarları ve port yapılandırması
└── templates/
    └── index.html        # Saf HTML5, CSS3 ve ES6 JavaScript tek sayfa arayüzü (SPA)
```

### 3.1. Backend Katmanı (`app.py`)

Backend dosyası, harici yapay zeka servisleri ile kullanıcı arayüzü arasındaki tüm veri akışını, araç çağrılarını ve akış ayrıştırmasını yönetir.

#### 1. Çoklu Sağlayıcı ve Model Yönetimi
`PROVIDERS` sözlüğü sistemin merkezi model kaydıdır:
* **Groq API:** Llama 3.3 70B, Llama 3.1 8B, Llama 4 Scout 17B (görsel destekli), GPT-OSS (20B/120B) ve Qwen 3 32B modellerini barındırır. Modellerin özelliğine göre `reasoning_effort` parametresi ile akıl yürütme seviyesi (low/medium/high) dinamik aktarılır.
* **Gemini API:** Gemini 3 (3.8, 3.7, 3.6, 3.5 Flash, 3.1 Flash-Lite) ve Gemini 2.5 (Pro, Flash) serisini kapsar. Google REST v1beta uç noktası (`streamGenerateContent?alt=sse`) üzerinden doğrudan çağrılır. Gemini 3 için `thinkingLevel`, Gemini 2.5 için ise `thinkingBudget` (1024, 8192, 24576 token) otomatik yapılandırılır.
* **OpenRouter API:** DeepSeek R1 (ücretsiz yerel akıl yürütmeli) ve Qwen 2.5 Coder 32B modellerini OpenAI uyumlu REST standardında sisteme bağlar.

#### 2. Düşünce Akışı Ayrıştırıcı (`ThinkSplitter`)
Bazı modeller akıl yürütme verisini doğrudan API parametresiyle ayrı bir delta olarak dönerken, bazıları metin gövdesi içerisine `<think>...</think>` etiketleri yerleştirerek üretir.

`ThinkSplitter` sınıfı, token akışı sırasında etiketlerin parçalanarak gelmesi durumunu (`_partial_suffix`) hesaplayan durum makineli (state-machine) bir tamponlama algoritmasıdır:
* Örneğin `<th` parçası bir chunk içinde, `ink>` parçası sonraki chunk içinde gelse dahi bunu kaçırmaz.
* Etiket açıldığı andan kapandığı ana kadar gelen tüm tokenları `reasoning` kanalına yönlendirir.
* Etiket dışındaki temiz yanıtı `content` kanalına aktarır. Böylece modelin iç monoloğu ile nihai yanıt birbirinden kusursuz şekilde izole edilir.

#### 3. Gerçek Zamanlı Canlı Web Araştırması
Kullanıcı web aramasını aktif ettiğinde şu işlem adımları yürütülür:
* Son kullanıcı mesajından arama sorgusu ayıklanır.
* DuckDuckGo motoru (`ddgs`) asenkron bir iş parçacığında (`asyncio.to_thread`) çalıştırılarak ilk 4 web kaynağı taranır.
* Taranan bağlantılardan sayfa metinleri çekilir (sayfa başına maksimum 4.000 karakter).
* İstemciye canlı SSE olayları fırlatılır:
  * `{"type": "search", "phase": "start", "query": "..."}`
  * `{"type": "search", "phase": "visit", "url": "..."}`
  * `{"type": "search", "phase": "done", "count": 4}`
* Toplanan sayfa özetleri, kaynak URL'leri ile birlikte sistem istemine (System Prompt) bağlam olarak enjekte edilir ve modele bilgileri bu kaynaklara dayandırması talimatı verilir.

#### 4. Belge ve Görsel İşleme Motoru
* **PDF Metin Çıkarımı (`/api/extract`):** Yüklenen PDF dosyaları diske kaydedilmeden doğrudan bellek akışında (`io.BytesIO`) `pypdf.PdfReader` aracılığıyla okunur. Maksimum 15 MB dosya boyutu ve 60.000 karakter güvenlik sınırı uygulanır. Çıkarılan metin o anki kullanıcı mesajına iliştirilir.
* **Görsel Doğrulama ve Güvenlik:** Görseller istemci tarafından Base64 Data URL formatında iletilir. Sunucu tarafında MIME tipi kontrolü (`image/jpeg`, `image/png`, `image/webp`, `image/gif`) yapılır ve maksimum 6 milyon Base64 karakteri (yaklaşık 4 MB) sınır denetimi uygulanır.

#### 5. Dinamik Token Bütçesi Genişletme
Akıl yürütme (reasoning) açıldığında, düşünce tokenları modelin toplam çıktı limitinden harcar. Modelin cevabı yarıda kesmesini önlemek için, düşünme modu açıkken `max_tokens` değeri Gemini için otomatik olarak en az 16.384'e, diğer modeller için en az 8.192'ye yükseltilir.

---

### 3.2. Frontend Katmanı (`templates/index.html`)

Arayüz, koyu tema odaklı, responsive ve son derece akıcı bir kullanıcı deneyimi sunar.

#### 1. Yerel Durum Yönetimi ve Depolama Optimizasyonu
* Tüm sohbet geçmişi ve kullanıcı ayarları tarayıcının `localStorage` alanında saklanır.
* **Tarayıcı Kotası Güvenliği (`stripTransient`):** Base64 kodlu görseller tarayıcının yerel depolama kotasını (genellikle 5 MB) doldurmasın diye, sohbet geçmişi kaydedilirken görseller nesneden ayıklanır.
* **Bağlam Penceresi Sınırı (`buildHistory`):** Modelin bağlam sınırını aşmamak ve gereksiz token tüketimini engellemek için, modele yalnızca kullanıcının belirlediği son N adet mesaj (`context_messages`, varsayılan 20) gönderilir.

#### 2. Akıllı Kod Blokları ve İndirme Sistemi
Markdown çıktısı `marked.js` ile derlenir ve `highlight.js` ile renklendirilir:
* Kod bloklarının üzerine özel bir çubuk yerleştirilir; programlama dili otomatik tespit edilir.
* **Panoya Kopyala:** Kod bloğunu tek tıkla panoya kopyalar ve kopyalandı onayı verir.
* **Dile Göre Kod İndir:** Tespit edilen programlama diline göre doğru dosya uzantısını (`.py`, `.js`, `.html`, `.css`, `.sh`, `.sql`, `.rs`, `.go` vb.) otomatik atayarak kodu bilgisayara dosya olarak indirir.
* **Mesajı İndir:** Eğer asistanın yanıtı tek bir kod bloğundan ibaretse doğrudan o dilin uzantısıyla, normal metinse `.md` uzantısıyla dışa aktarılır.

#### 3. Kalıcı Hafıza (Memory) ve Sistem İstemi
Ayarlar menüsünde kullanıcıya iki katmanlı yönlendirme sunulur:
1. **Sistem Mesajı (System Prompt):** Modelin genel davranış kuralları ve üslubu.
2. **Kalıcı Hafıza (Memory):** Kullanıcının mesleği, tercih ettiği teknolojiler veya konuşma tarzı gibi her sohbette hatırlanması gereken notlar. Bu notlar sohbetler silinse dahi korunur ve her istekte sistem istemine arka planda dahil edilir.

#### 4. İstemci Tarafı Görsel Sıkıştırma
Görseller sunucuya gönderilmeden önce istemci tarafında bir HTML5 Canvas üzerinde işlenir. En uzun kenar 1280 piksele düşürülür ve JPEG 0.85 kalitesinde sıkıştırılır. Böylece mobil ağlarda dahi yükleme gecikmesi yaşanmaz.

---

## 4. Kurulum ve Çalıştırma Rehberi

### 1. Depoyu Hazırlayın
```powershell
git clone <depo-adresi>
cd chatbot_project
```

### 2. Python Sanal Ortamını Kurun ve Başlatın
Windows PowerShell üzerinde:
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Bağımlılıkları Yükleyin
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. API Anahtarlarını Tanımlayın
Proje kök dizininde bulunan `.env` dosyasını açıp kullanmak istediğiniz sağlayıcıların anahtarlarını girin:
```env
GROQ_API_KEY=gsk_...
GEMINI_API_KEY=AIzaSy...
OPENROUTER_API_KEY=sk-or-v1-...
```

*Not: Hepsini girmek zorunda değilsiniz. Yalnızca anahtarını girdiğiniz sağlayıcılar arayüzde aktif hale gelir.*

### 5. Sunucuyu Başlatın
```powershell
python app.py
```
veya alternatif Uvicorn komutuyla:
```powershell
uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```

Uygulama hazır olduğunda tarayıcınızdan `http://127.0.0.1:8000` adresine giderek sohbet etmeye başlayabilirsiniz.

---

## 5. API Uç Noktaları Referansı

### `GET /`
* **İşlev:** Ana tek sayfa sohbet arayüzünü (`index.html`) döner.

### `GET /api/config`
* **İşlev:** Tanımlı sağlayıcıların, model listelerinin ve `.env` dosyasında hangi sağlayıcıların tanımlı olduğunun durumunu döner.
* **Güvenlik:** API anahtarlarının kendisi asla dönülmez; yalnızca `configured: true/false` durumu paylaşılır.

### `POST /api/extract`
* **İşlev:** Yüklenen PDF belgesinden metin çıkarır.
* **İçerik Türü:** `multipart/form-data`
* **Örnek Yanıt:**
  ```json
  {
    "name": "dokuman.pdf",
    "pages": 8,
    "text": "Belgeden okunan metin içeriği...",
    "truncated": false
  }
  ```

### `POST /api/chat`
* **İşlev:** Modele bağlamı iletir ve Server-Sent Events (SSE) akışını başlatır.
* **İstek Gövdesi:**
  ```json
  {
    "provider": "gemini",
    "model": "gemini-3.8-flash",
    "temperature": 0.7,
    "top_p": 0.95,
    "max_tokens": 4096,
    "system_prompt": "Sen yardımcı bir asistansın.",
    "reasoning": "medium",
    "web_search": true,
    "messages": [
      {
        "role": "user",
        "content": "Kuantum bilgisayarlar hakkında güncel durum nedir?",
        "images": []
      }
    ]
  }
  ```
* **SSE Akış Olayları:**
  * `search` (`start`, `visit`, `done`, `error`)
  * `reasoning` (düşünce delta parçaları)
  * `content` (görünür metin delta parçaları)
  * `done` (akış tamamlandı)
  * `error` (hata mesajı)

---

## 6. Model ve Yetenek Tablosu

| Sağlayıcı | Model Adı | Model ID | Görsel Analizi | Düşünme (Reasoning) |
|---|---|---|:---:|:---:|
| **Groq** | Llama 3.3 70B | `llama-3.3-70b-versatile` | Yok | Sistem İstemiyle |
| **Groq** | Llama 3.1 8B | `llama-3.1-8b-instant` | Yok | Sistem İstemiyle |
| **Groq** | Llama 4 Scout 17B | `meta-llama/llama-4-scout-17b-16e-instruct` | Var | Sistem İstemiyle |
| **Groq** | GPT-OSS 120B | `openai/gpt-oss-120b` | Yok | Yerel (`reasoning_effort`) |
| **Groq** | Qwen 3 32B | `qwen/qwen3-32b` | Yok | Yerel |
| **Gemini** | Gemini 3.8 Flash | `gemini-3.8-flash` | Var | Yerel (`thinkingLevel`) |
| **Gemini** | Gemini 3.7 Flash | `gemini-3.7-flash` | Var | Yerel (`thinkingLevel`) |
| **Gemini** | Gemini 2.5 Pro | `gemini-2.5-pro` | Var | Yerel (`thinkingBudget`) |
| **Gemini** | Gemini 2.5 Flash | `gemini-2.5-flash` | Var | Yerel (`thinkingBudget`) |
| **OpenRouter** | DeepSeek R1 (Free) | `deepseek/deepseek-r1:free` | Yok | Yerel (`effort`) |
| **OpenRouter** | Qwen 2.5 Coder 32B | `qwen/qwen-2.5-coder-32b-instruct:free` | Yok | Sistem İstemiyle |

---

## 7. Güvenlik ve Gizlilik Prensipleri

1. **Anahtar İzolasyonu:** API anahtarları istemci tarafına asla geçmez. Arayüz sadece anahtarın sunucuda tanımlı olup olmadığını bilir.
2. **Geçici Belge Belleği:** Yüklenen PDF dosyaları ve görseller diske yazılmaz; bellek üzerinde anlık işlenip istek sonlandığında bellekten temizlenir.
3. **Depolama Koruması:** `localStorage` şişmesini önlemek amacıyla sohbet geçmişine Base64 görseller eklenmez.

---

## 8. Lisans

Bu proje açık kaynak olarak geliştirilmiştir. Kişisel ve ticari projelerinizde serbestçe kullanabilir, çatallayabilir ve katkıda bulunabilirsiniz.
