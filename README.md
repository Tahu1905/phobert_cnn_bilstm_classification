 Vietnamese AI-Text Detector: Hybrid PhoBERT-CNN-BiLSTM Architecture
(https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)](https://pytorch.org/) (https://img.shields.io/badge/Model-PhoBERT-blue?style=for-the-badge&logo=huggingface&logoColor=white)](https://github.com/VinAIResearch/PhoBERT) ()]()

Hệ thống phát hiện văn bản do AI tạo sinh (ChatGPT, Llama, Gemini) dành riêng cho tiếng Việt, sử dụng kiến trúc lai ghép tiên tiến giữa Transformer, CNN và BiLSTM.

📖 Giới Thiệu (Introduction)
Trong bối cảnh bùng nổ của Generative AI, việc phân biệt nội dung do con người viết và máy tạo ra trở thành một thách thức lớn. Dự án này giới thiệu một phương pháp tiếp cận State-of-the-Art (SOTA) cho tiếng Việt, kết hợp sức mạnh hiểu ngữ nghĩa của PhoBERT với khả năng trích xuất đặc trưng cục bộ của CNN và khả năng nắm bắt chuỗi thời gian của BiLSTM.

Mô hình được thiết kế để phát hiện các dấu hiệu tinh vi của văn bản máy: sự lặp lại cấu trúc (structural repetition), độ trôi chảy bất thường (unnatural fluency) và các mẫu thống kê (statistical patterns) mà mắt thường khó nhận biết.

🧠 Kiến Trúc Hệ Thống (System Architecture)
Mô hình PhoBERT-CNN-BiLSTM hoạt động dựa trên luồng xử lý dữ liệu phức hợp:

Input Processing: Văn bản được chuẩn hóa (Unicode NFC), làm sạch (loại bỏ công thức toán, trích dẫn) và phân đoạn từ (Word Segmentation) bằng VnCoreNLP.

PhoBERT Embedding Fusion:

Thay vì chỉ sử dụng lớp cuối cùng, chúng tôi nối (concatenate) 4 lớp ẩn cuối cùng của PhoBERT.

Tạo ra vector biểu diễn siêu giàu thông tin với kích thước 3072 chiều (768 x 4).

Parallel Feature Extraction:

Nhánh CNN: Sử dụng các bộ lọc kích thước `` để bắt các mẫu n-gram cục bộ.

Nhánh BiLSTM: Quét toàn bộ chuỗi văn bản theo hai chiều để nắm bắt ngữ cảnh toàn cục.

Fusion & Classification:

Kết hợp đặc trưng từ hai nhánh.

Đi qua các lớp Fully Connected với Dropout để đưa ra dự đoán xác suất (Human vs AI).

📊 Dữ Liệu & Hiệu Năng (Dataset & Performance)
Dự án sử dụng bộ dữ liệu ViDetect và các nguồn dữ liệu khoa học nội bộ (train_new.tsv) để huấn luyện.

Kết quả thực nghiệm cho thấy mô hình vượt trội hơn so với việc chỉ sử dụng PhoBERT hoặc BiLSTM đơn lẻ từ 2-5%.

🛠️ Cài Đặt & Sử Dụng (Installation & Usage)
Yêu cầu hệ thống
Python 3.7+

CUDA (GPU) được khuyến nghị để huấn luyện.

RAM: 16GB+ (do vector nhúng kích thước lớn).
## 🚀 Hướng dẫn thực hiện

### Bước 1: Clone Repository
```bash
git clone https://github.com/Tahu1905/phobert_cnn_bilstm_classification.git
cd phobert_cnn_bilstm_classification
```
### Bước 2: Cài đặt thư viện
```bash
pip install -r requirements.txt
#Cài đặt VnCoreNLP (Bắt buộc)
pip install py_vncorenlp
```
### Bước 3: Chuẩn bị dữ liệu
```text
label	sentence
0	Chiều dài nước nhảy là một đặc trưng quan trọng... (Human text)
1	Trong kỷ nguyên số, trí tuệ nhân tạo... (AI text)

Đặt file dữ liệu train_new.tsv vào thư mục gốc. Định dạng file:
```
### Bước 4: Huấn luyện
```bash
python main.py

Chạy lệnh sau để bắt đầu quá trình training. Hệ thống sẽ tự động tải PhoBERT pre-trained weights.
Logs sẽ được lưu tại ./log (xem bằng TensorBoard).

Model tốt nhất sẽ được lưu tại saved_dict/best_model.pth.
```
### Bước 5: Kiểm thử (Inference)
Sử dụng script đánh giá để kiểm tra trên văn bản mới:

🔬 Phân Tích Kỹ Thuật (Technical Insights)
Tại sao lại nối 4 lớp PhoBERT?
Các lớp trên cùng của Transformer thường thiên về nhiệm vụ pre-training (MLM), trong khi các lớp dưới chứa thông tin ngữ pháp. Việc nối 4 lớp giúp mô hình downstream (CNN-BiLSTM) truy cập được cả thông tin ngữ pháp và ngữ nghĩa, tăng độ bền vững (robustness) cho mô hình.

Tại sao dùng CNN kết hợp BiLSTM?
CNN cực tốt trong việc phát hiện các từ khóa hoặc cụm từ "lạ" mà AI hay dùng (ví dụ: các từ sáo rỗng, lặp lại).

BiLSTM đảm bảo rằng văn bản phải có tính mạch lạc về thời gian. AI đôi khi viết rất trôi chảy từng câu nhưng tổng thể đoạn văn lại thiếu logic chặt chẽ, điều mà BiLSTM có thể phát hiện qua các trạng thái ẩn.

### 📜 Trích Dẫn (Citation)
Nếu bạn sử dụng mã nguồn hoặc ý tưởng từ dự án này, vui lòng trích dẫn:

### 6. Kết Luận và Định Hướng Tương Lai
Báo cáo này đã trình bày một giải pháp toàn diện cho vấn đề phát hiện văn bản AI trong tiếng Việt. Bằng cách kết hợp sức mạnh của PhoBERT với kiến trúc CNN-BiLSTM, hệ thống không chỉ đạt được độ chính xác cao mà còn thể hiện sự bền vững trước các loại văn bản đầu vào đa dạng.

Tuy nhiên, cuộc đua giữa AI tạo sinh và AI phát hiện là một cuộc đua không hồi kết. Các hướng phát triển trong tương lai bao gồm:

Adversarial Training: Huấn luyện mô hình với các văn bản AI đã bị làm nhiễu (paraphrased) để tăng khả năng chống chịu trước các kỹ thuật lách luật.

Model Distillation: Nén mô hình khổng lồ này thành phiên bản nhẹ hơn (ví dụ: sử dụng PhoBERT-tiny hoặc DistilBERT) để có thể chạy trên các thiết bị biên (Edge devices) hoặc trình duyệt web.

Explainable AI (XAI): Tích hợp cơ chế Attention Map để trực quan hóa lý do tại sao mô hình phán đoán một đoạn văn là do AI viết (ví dụ: tô màu các từ/cụm từ đáng ngờ), giúp tăng tính thuyết phục đối với người dùng cuối.

Hệ thống này không chỉ là một công cụ kỹ thuật mà còn là một bước tiến quan trọng trong việc bảo vệ sự trong sáng và chân thực của không gian thông tin tiếng Việt.

Tác giả báo cáo: Ngày: 02/02/2026 Phiên bản: 1.0.0
