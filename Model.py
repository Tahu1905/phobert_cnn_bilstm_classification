from transformers import AutoModel, AutoTokenizer
import torch
import  torch.nn as nn
import torch.nn.functional as F
import os
import  re
import unicodedata
import py_vncorenlp
from vncorenlp import VnCoreNLP
class PhoBert:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.phobert = AutoModel.from_pretrained("vinai/phobert-base",output_hidden_states=True).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
        current_dir = os.path.dirname(os.path.abspath(__file__))
        rdr_path = os.path.join(current_dir, 'vncorenlp')

        # 2. Kiểm tra xem folder có tồn tại không
        if not os.path.exists(rdr_path):
            print(f"LỖI: Không tìm thấy thư mục VnCoreNLP tại: {rdr_path}")
            # Fallback: Nếu không có thì mới thử tải lại (nhưng khuyên bạn nên tải tay)
            try:
                py_vncorenlp.download_model(save_dir=rdr_path)
            except:
                print("Không thể tự tải VnCoreNLP. Vui lòng tải thủ công!")
                raise

        # 3. Khởi tạo trực tiếp
        try:
            self.rdr = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir=rdr_path)
        except Exception as e:
            print("Lỗi khởi tạo VnCoreNLP. Hãy kiểm tra xem máy đã cài JAVA chưa?")
            raise e

        self.max_len = 256
        self.cls_token = torch.tensor([self.tokenizer.cls_token_id], device=self.device)
        self.sep_token = torch.tensor([self.tokenizer.sep_token_id], device=self.device)

    def preprocess(self, text):

        if not isinstance(text, str): return ""

        # 1. Chuẩn hóa Unicode (NFC)
        text = unicodedata.normalize('NFC', text)


        text = re.sub(r'\$.*?\$', '', text)  # Xóa $...$
        text = re.sub(r'\[\d+\]', '', text)  # Xóa [1; 2]

        # 3. Xóa khoảng trắng thừa
        text = re.sub(r'\s+', ' ', text).strip()

        # 4. Tách từ (Word Segmentation) - BẮT BUỘC
        sentences = self.rdr.word_segment(text)
        return " ".join(sentences)

    def extract_feature(self,text,overlap =50):
        text_segment = self.preprocess(text)
        tokens = self.tokenizer(text_segment, add_special_tokens=False, return_tensors="pt")
        input_ids = tokens["input_ids"][0]

        chunk_size = self.max_len - 2
        stride = chunk_size - overlap


        all_embeddings = []
        self.phobert.eval()
        seq_len = len(input_ids)


        for i in range(0, seq_len, stride):
            chunk = input_ids[i: i + chunk_size]
            if len(chunk) < 2 and i > 0:
                break

            chunk = chunk.to(self.device)
            input_chunk = torch.cat([self.cls_token, chunk, self.sep_token]).unsqueeze(0)  # [1, chunk_len+2]


            attention_mask = torch.ones_like(input_chunk).to(self.device)

            with torch.no_grad():
                output = self.phobert(input_ids=input_chunk, attention_mask=attention_mask)


            hidden_states = output.hidden_states
            last_4 = torch.cat([hidden_states[i] for i in [-1, -2, -3, -4]], dim=-1)

            core_embed = last_4[:, 1:-1, :]


            if i == 0:

                all_embeddings.append(core_embed)
            else:

                if core_embed.size(1) > overlap:
                    new_info = core_embed[:, overlap:, :]
                    all_embeddings.append(new_info)
                else:
                    pass

        if all_embeddings:
            full_seq = torch.cat(all_embeddings, dim=1)

            if full_seq.size(1) > seq_len:
                full_seq = full_seq[:, :seq_len, :]

            return full_seq  # Shape: [1, seq_len, 3072]
        else:
            return torch.zeros(1, 1, 3072).to(self.device)


class CNN_BiLSTM(nn.Module):
    def __init__(self, config):
        super(CNN_BiLSTM, self).__init__()

        # Parameter
        self.embed_dim = config.embed_dim
        self.hidden_dim = config.hidden_size
        self.num_layers = config.num_layers
        self.class_num = config.num_classes
        self.kernel_num = config.num_filters
        self.kernel_sizes = config.filter_sizes
        self.dropout_val = config.dropout

        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=self.embed_dim,
                      out_channels=self.kernel_num,
                      kernel_size=K)
            for K in self.kernel_sizes
        ])

        # --- PHẦN 2: BiLSTM ---
        # Input size = embed_dim (3072)
        self.bilstm = nn.LSTM(input_size=self.embed_dim,
                              hidden_size=self.hidden_dim,
                              num_layers=self.num_layers,
                              dropout=config.dropout_LSTM if self.num_layers > 1 else 0,
                              bidirectional=True,
                              batch_first=True)  # Luôn dùng batch_first=True cho dễ quản lý

        # --- PHẦN 3: Fully Connected (Phân loại) ---
        # Kích thước đầu vào của lớp Linear = (Số Filter CNN * Số loại kích thước) + (LSTM Hidden * 2 chiều)
        L = (len(self.kernel_sizes) * self.kernel_num) + (self.hidden_dim * 2)

        self.hidden2label1 = nn.Linear(L, L // 2)
        self.hidden2label2 = nn.Linear(L // 2, self.class_num)

        self.dropout = nn.Dropout(self.dropout_val)

    def forward(self, x):
        #CNN
        cnn_in = x.permute(0, 2, 1)
        # Chạy qua từng bộ lọc CNN + ReLU + MaxPool
        cnn_out = [F.relu(conv(cnn_in)) for conv in self.convs]
        # Max Pooling (Lấy đặc trưng lớn nhất)
        cnn_out = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in cnn_out]
        # Ghép các đặc trưng CNN lại: Shape [Batch, Total_Kernels]
        cnn_cat = torch.cat(cnn_out, 1)
        cnn_cat = self.dropout(cnn_cat)

        # --- BiLSTM ---
        lstm_out, _ = self.bilstm(x)
        # Global Max Pooling cho LSTM (Thay vì lấy hidden cuối cùng)
        # Giúp bắt tín hiệu tốt hơn cho văn bản dài
        lstm_out = lstm_out.permute(0, 2, 1)  # Đảo để Max Pool theo chiều thời gian
        lstm_pool = F.max_pool1d(lstm_out, lstm_out.size(2)).squeeze(2)
        # --- CONCAT ---
        # Ghép CNN và LSTM lại với nhau
        cnn_bilstm_out = torch.cat((cnn_cat, lstm_pool), 1)
        # --- PHÂN LOẠI ---
        out = self.hidden2label1(F.relu(cnn_bilstm_out))
        logit = self.hidden2label2(F.relu(out))

        return logit

