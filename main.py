import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
import multiprocessing

from Model import *
from utils import *
from train_eval import train, init_network

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. Tạo đường dẫn tuyệt đối đến file dữ liệu (Bất chấp đang đứng ở đâu)
DATA_FILE = os.path.join(BASE_DIR, 'train_new.tsv')

# (Debug) In ra để kiểm tra xem nó đang trỏ đi đâu
print(f"[DEBUG] Looking for data at: {DATA_FILE}")

# Kiểm tra nóng ngay lập tức
if not os.path.exists(DATA_FILE):
    print(f"[ERROR] Python vẫn KHÔNG THẤY file tại: {DATA_FILE}")
    print(f"[DEBUG] Các file đang có trong thư mục {BASE_DIR} là:")
    print(os.listdir(BASE_DIR))  # In ra danh sách file để bạn so sánh tên
    sys.exit(1)


def print_log(message, level="INFO"):
    colors = {
        "INFO": "\033[94m",  # Blue
        "SUCCESS": "\033[92m",  # Green
        "WARNING": "\033[93m",  # Yellow
        "ERROR": "\033[91m",  # Red
        "RESET": "\033[0m"
    }
    lvl = f"[{level}]"
    print(f"{colors.get(level, '')}{lvl:<9} {message}{colors['RESET']}")


def print_header(title):
    print(f"\n{'=' * 60}")
    print(f"{title.center(60)}")
    print(f"{'=' * 60}")


# --- DATA PIPELINE ---
def prepare_data_in_ram(config):
    print_header("PHASE 1: DATA PREPARATION (IN-MEMORY)")

    # 1. Init Model
    print_log("Initializing PhoBERT for Feature Extraction...", "INFO")
    try:
        phobert = PhoBert()
    except Exception as e:
        print_log(f"Failed to load PhoBERT: {e}", "ERROR")
        sys.exit(1)

    # 2. Load Data
    print_log(f"Reading dataset: {DATA_FILE}", "INFO")
    try:
        # Đọc file, đảm bảo kiểu dữ liệu string
        df = pd.read_csv(DATA_FILE, sep='\t', encoding='utf-8-sig', dtype=str).dropna()
        # Chuẩn hóa tên cột (xóa khoảng trắng thừa trong tên cột nếu có)
        df.columns = [c.strip() for c in df.columns]

        print_log(f"Loaded {len(df)} samples.", "SUCCESS")
        print_log(f"Columns found: {list(df.columns)}", "INFO")

        # Kiểm tra xem có đúng cột không
        if 'sentence' not in df.columns or 'label' not in df.columns:
            print_log(f"❌ File thiếu cột 'sentence' hoặc 'label'. Cột hiện tại: {list(df.columns)}", "ERROR")
            sys.exit(1)

    except FileNotFoundError:
        print_log(f"File '{DATA_FILE}' not found!", "ERROR")
        sys.exit(1)

    # 3. Extraction Loop
    # Map nhãn: Chấp nhận cả string '0','1' và số 0,1
    label_map = {'Human': 0, 'AI': 1, '0': 0, '1': 1, 0: 0, 1: 1}
    all_features = []
    all_labels = []

    print_log("Starting Feature Extraction...", "INFO")

    error_count = 0
    success_count = 0

    # Dùng tqdm
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting", unit="sample", ncols=100):
        try:
            # --- SỬA QUAN TRỌNG: Dùng tên cột thay vì chỉ số ---
            text = str(row['sentence']).strip()
            label_raw = row['label']

            # Kiểm tra nhãn hợp lệ
            if label_raw not in label_map:
                # Thử convert sang int xem sao (nếu label là 0.0)
                try:
                    label_int = int(float(label_raw))
                    if label_int in label_map:
                        label_raw = label_int
                    else:
                        continue
                except:
                    continue

            label = label_map[label_raw]

            # Extract & Move to CPU
            feature = phobert.extract_feature(text).squeeze(0).cpu()

            all_features.append(feature)
            all_labels.append(label)
            success_count += 1

        except Exception as e:
            error_count += 1
            # Chỉ in 3 lỗi đầu tiên để debug
            if error_count <= 3:
                print(f"\n❌ Lỗi dòng {idx}: {type(e).__name__} - {e}")

    print_log(f"Extraction Done. Success: {success_count} | Failed: {error_count}", "SUCCESS")

    if success_count == 0:
        print_log("CRITICAL: Không trích xuất được mẫu nào!", "ERROR")
        print_log("👉 Kiểm tra lại tên cột trong file TSV (phải là 'label' và 'sentence').", "ERROR")
        print_log("👉 Kiểm tra lại VnCoreNLP (Java) đã chạy được chưa.", "ERROR")
        sys.exit(1)

    return all_features, all_labels


def main():
    # 1. Config & Setup
    config = Config()

    config.batch_size = 32

    # Setup Seed
    np.random.seed(1)
    torch.manual_seed(1)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(1)
        print_log(f"GPU Detected: {torch.cuda.get_device_name(0)}", "SUCCESS")
    else:
        print_log("WARNING: No GPU detected! Training will be slow.", "WARNING")

    # 2. Prepare Data
    X_list, y_list = prepare_data_in_ram(config)

    # 3. Splitting
    print_header("PHASE 2: DATA SPLITTING")
    indices = np.arange(len(X_list))

    train_idx, temp_idx, y_train, y_temp = train_test_split(
        indices, y_list, test_size=0.2, stratify=y_list, random_state=42
    )
    dev_idx, test_idx, y_dev, y_test = train_test_split(
        temp_idx, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )

    print(f"{'Set':<10} | {'Count':<10} | {'Ratio'}")
    print("-" * 30)
    print(f"{'Train':<10} | {len(train_idx):<10} | 80%")
    print(f"{'Val':<10} | {len(dev_idx):<10} | 10%")
    print(f"{'Test':<10} | {len(test_idx):<10} | 10%")

    # 4. DataLoader Setup (Optimized for Cloud)
    def create_dataset(indices, all_X, all_Y):
        return [(all_X[i], all_Y[i]) for i in indices]

    train_data = create_dataset(train_idx, X_list, y_list)
    dev_data = create_dataset(dev_idx, X_list, y_list)
    test_data = create_dataset(test_idx, X_list, y_list)

    train_iter = DataLoader(train_data, batch_size=config.batch_size, shuffle=True,
                            collate_fn=collate_fn, num_workers=2, pin_memory=True)
    dev_iter = DataLoader(dev_data, batch_size=config.batch_size, shuffle=False,
                          collate_fn=collate_fn, num_workers=2, pin_memory=True)
    test_iter = DataLoader(test_data, batch_size=config.batch_size, shuffle=False,
                           collate_fn=collate_fn, num_workers=2, pin_memory=True)

    # 5. Training
    print_header("PHASE 3: TRAINING START")
    model = CNN_BiLSTM(config).to(config.device)
    init_network(model)

    print_log(f"Model: {config.model_name}", "INFO")
    print_log(f"Params: Batch={config.batch_size}, LR={config.learning_rate}", "INFO")

    train(config, model, train_iter, dev_iter, test_iter)


if __name__ == "__main__":
    main()