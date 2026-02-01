import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from Model import *
from utils import *
from train_eval import train, init_network


def prepare_data_in_ram(config):
    """
    Hàm này chạy 1 lần: Đọc file -> Extract Feature -> Trả về Tensor
    """
    print(">>> 1. Đang đọc và xử lý dữ liệu (Extract Features)...")
    phobert = PhoBert()

    # Đọc file TSV (Hãy chắc chắn file 'train_new.tsv' nằm cùng thư mục)
    try:
        df = pd.read_csv('train_new.tsv', sep='\t', encoding='utf-8-sig').dropna()
    except FileNotFoundError:
        print("❌ Lỗi: Không tìm thấy file 'train_new.tsv'. Hãy kiểm tra lại đường dẫn!")
        exit()

    label_map = {'Human': 0, 'AI': 1, 0: 0, 1: 1, '0': 0, '1': 1}
    all_features = []
    all_labels = []

    print(f"🔄 Đang trích xuất đặc trưng cho {len(df)} mẫu...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        text = str(row[0]).strip()
        label_raw = row[1]
        if label_raw not in label_map: continue
        label = label_map[label_raw]

        try:
            # Extract và chuyển về CPU ngay
            feature = phobert.extract_feature(text).squeeze(0).cpu()
            all_features.append(feature)
            all_labels.append(label)
        except Exception as e:
            print(f"Lỗi dòng {idx}: {e}")

    return all_features, all_labels

def main():
    config = Config()

    # Setup Seed
    np.random.seed(1)
    torch.manual_seed(1)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(1)

    # --- BƯỚC 1: Xử lý dữ liệu thô thành Vector (Nặng nhất) ---
    # List các tensor: [Tensor1, Tensor2,...]
    X_list, y_list = prepare_data_in_ram(config)

    # --- BƯỚC 2: Chia dữ liệu (Stratified Split) ---
    print(">>> 2. Đang chia tập dữ liệu (Stratified Split)...")

    # Vì X_list là list các tensor có độ dài khác nhau, ta chia index
    indices = np.arange(len(X_list))

    # Tách Train (80%) - Temp (20%)
    train_idx, temp_idx, y_train, y_temp = train_test_split(
        indices, y_list, test_size=0.2, stratify=y_list, random_state=42
    )

    # Tách Dev (10%) - Test (10%)
    dev_idx, test_idx, y_dev, y_test = train_test_split(
        temp_idx, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )

    print(f"   + Train: {len(train_idx)} | Dev: {len(dev_idx)} | Test: {len(test_idx)}")


    from torch.nn.utils.rnn import pad_sequence



    # Helper function tạo dataset từ index
    def create_dataset(indices, all_X, all_Y):
        data = []
        for i in indices:
            data.append((all_X[i], all_Y[i]))
        return data  # Trả về list tuple để đưa vào DataLoader

    train_data = create_dataset(train_idx, X_list, y_list)
    dev_data = create_dataset(dev_idx, X_list, y_list)
    test_data = create_dataset(test_idx, X_list, y_list)

    train_iter = DataLoader(train_data, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)
    dev_iter = DataLoader(dev_data, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)
    test_iter = DataLoader(test_data, batch_size=config.batch_size, shuffle=False,
                           collate_fn=collate_fn)  # Sửa test_dataset -> test_data

    # --- BƯỚC 4: Train ---
    print(">>> 4. Khởi tạo Model & Train...")
    model = CNN_BiLSTM(config).to(config.device)
    init_network(model)

    train(config, model, train_iter, dev_iter, test_iter)


if __name__ == "__main__":
    main()