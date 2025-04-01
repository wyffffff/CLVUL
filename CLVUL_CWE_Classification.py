import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import numpy as np
import logging
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import random
import os

# ------------------------------ 配置与随机种子 ------------------------------
logging.basicConfig(level=logging.INFO)

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

set_seed()

# ------------------------------ 超参数设置 ------------------------------
MODEL_NAME = "unixcoder-base-nine"  # 使用的预训练模型名称
BATCH_SIZE = 16                     # 根据显存情况调整
LEARNING_RATE = 1e-5
NUM_EPOCHS = 3
MAX_LENGTH = 512                    # 根据显存情况调整
TEMPERATURE = 0.07
MODEL_SAVE_PATH = "best_model.pt"   # 模型保存路径

# ------------------------------ 标签（CWE）定义 ------------------------------
# 仅保留前4个 CWE 条目进行训练
training_cwe = [
      {'id': 476, 'description': "NULL Pointer Dereference"},
    {'id': 119, 'description': "Buffer Overflow"},
    {'id': 787, 'description': "Out-of-bounds Write"},
    {'id': 416, 'description': "Use After Free"},
    {'id': 125, 'description': "Out-of-bounds Read"},
    {'id': 20, 'description': "Improper Input Validation"},
    {'id': 401, 'description': "Memory Leak"},
    {'id': 200, 'description': "Information Exposure"},
    {'id': 399, 'description': "Resource Management Errors"},
    {'id': 264, 'description': "Permissions, Privileges, and Access Controls"},
    {'id': 22, 'description': "Path Traversal"},
    {'id': 190, 'description': "Integer Overflow or Wraparound"}
]
# 此处 all_cwe 直接使用 training_cwe，因为仅训练这4个类别
all_cwe = training_cwe.copy()

# 构造标签描述字典和标签到索引的映射（顺序保持一致）
LABEL_DESCRIPTIONS_ALL = {cwe['id']: cwe['description'] for cwe in training_cwe}
all_label_ids = [cwe['id'] for cwe in all_cwe]
all_label_to_index = {label_id: idx for idx, label_id in enumerate(all_label_ids)}
all_index_to_label = {idx: label_id for idx, label_id in enumerate(all_label_ids)}
logging.info(f"标签到索引映射: {all_label_to_index}")

# ------------------------------ 数据集类定义 ------------------------------
class CodeDefectDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length, selected_cwe, label_descriptions, label_to_index):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.selected_cwe = selected_cwe
        self.label_descriptions = label_descriptions
        self.label_to_index = label_to_index

        dataframe = dataframe.copy()
        if dataframe['CWE_ID'].dtype == object:
            # 如果是字符串形式，比如 "CWE-119"，去除 "CWE-" 前缀后转换成数值
            dataframe['CWE_ID'] = dataframe['CWE_ID'].str.replace('CWE-', '', regex=False)
            dataframe['CWE_ID'] = pd.to_numeric(dataframe['CWE_ID'], errors='coerce')
            dataframe = dataframe[pd.notnull(dataframe['CWE_ID'])].copy()
            dataframe['CWE_ID'] = dataframe['CWE_ID'].astype(int)

        # 仅保留选定的 CWE 数据（只训练前4个类别）
        filtered_df = dataframe[dataframe['CWE_ID'].isin(selected_cwe)].reset_index(drop=True)
        self.labels = filtered_df['CWE_ID'].values  # 标签
        self.codes = filtered_df['Func'].values     # 代码文本

        logging.info(f"选定的 CWE 类别样本数: {len(self.labels)}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        code = str(self.codes[idx])
        cwe_id = int(self.labels[idx])
        label_index = self.label_to_index[cwe_id]

        encoding = self.tokenizer(
            code,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),         # [max_length]
            'attention_mask': encoding['attention_mask'].squeeze(0),   # [max_length]
            'labels': torch.tensor(label_index, dtype=torch.long)
        }

# ------------------------------ 损失函数定义 ------------------------------
class LabelContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(LabelContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, code_embeddings, label_embeddings, labels):
        # 计算相似度矩阵，形状为 (batch_size, num_classes)
        logits = torch.matmul(code_embeddings, label_embeddings.t()) / self.temperature
        loss = self.cross_entropy(logits, labels)
        return loss

# ------------------------------ 模型定义 ------------------------------
class UnixcoderForClassification(nn.Module):
    def __init__(self, model_name, num_labels, temperature, label_descriptions=None):
        """
        如果 label_descriptions 不为 None，则使用描述文本初始化标签嵌入。
        """
        super(UnixcoderForClassification, self).__init__()
        self.unicoder = AutoModel.from_pretrained(model_name)
        self.temperature = temperature
        hidden_size = self.unicoder.config.hidden_size

        if label_descriptions is not None:
            # 使用预训练模型和相同的分词器对描述文本进行编码
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            description_encoding = tokenizer(
                label_descriptions,
                add_special_tokens=True,
                max_length=MAX_LENGTH // 4,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            with torch.no_grad():
                description_outputs = self.unicoder(
                    input_ids=description_encoding['input_ids'],
                    attention_mask=description_encoding['attention_mask']
                )
            # 提取每个描述的 [CLS] 表示并归一化
            label_emb = description_outputs.last_hidden_state[:, 0, :]
            label_emb = nn.functional.normalize(label_emb, dim=1)
            self.label_embeddings = nn.Parameter(label_emb)
        else:
            # 如果没有描述，则随机初始化
            self.label_embeddings = nn.Parameter(torch.randn(num_labels, hidden_size))
            nn.init.xavier_uniform_(self.label_embeddings)

    def forward(self, input_ids, attention_mask):
        outputs = self.unicoder(input_ids=input_ids, attention_mask=attention_mask)
        # 使用 [CLS] 位置的输出作为代码嵌入，并归一化
        code_embeddings = outputs.last_hidden_state[:, 0, :]
        code_embeddings = nn.functional.normalize(code_embeddings, dim=1)
        return code_embeddings

# ------------------------------ 数据集划分 8:1:1 ------------------------------
# 假设 CSV 文件中包含 'CWE_ID' 和 'Func' 两列
dataframe = pd.read_csv('CWEID_Function_vul10.4.csv')
total_samples = len(dataframe)
logging.info(f"总样本数: {total_samples}")

# 先分出 10% 作为测试集
test_split_ratio = 0.1
train_val_df, test_df = train_test_split(
    dataframe,
    test_size=test_split_ratio,
    random_state=42,
    shuffle=True
)
logging.info(f"训练+验证集大小: {len(train_val_df)}")
logging.info(f"测试集大小: {len(test_df)}")

# 创建训练+验证数据集对象（仅选取前4个 CWE_ID）
train_val_dataset = CodeDefectDataset(
    train_val_df,
    tokenizer=AutoTokenizer.from_pretrained(MODEL_NAME),
    max_length=MAX_LENGTH,
    selected_cwe=all_label_ids,  # all_label_ids 中仅包含前4个类别
    label_descriptions=LABEL_DESCRIPTIONS_ALL,
    label_to_index=all_label_to_index
)

# 根据 train_val_dataset 的实际长度计算划分比例
dataset_length = len(train_val_dataset)
train_size = int((8/9) * dataset_length)
val_size = dataset_length - train_size

train_dataset, val_dataset = random_split(
    train_val_dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)
logging.info(f"训练集大小: {len(train_dataset)}")
logging.info(f"验证集大小: {len(val_dataset)}")

# 创建测试数据集对象（测试集也只包含前4个类别）
test_dataset = CodeDefectDataset(
    test_df,
    tokenizer=AutoTokenizer.from_pretrained(MODEL_NAME),
    max_length=MAX_LENGTH,
    selected_cwe=all_label_ids,
    label_descriptions=LABEL_DESCRIPTIONS_ALL,
    label_to_index=all_label_to_index
)
logging.info(f"测试集大小: {len(test_dataset)}")

if len(train_dataset) == 0:
    logging.error("训练集为空，请检查数据和 CWE 过滤条件。")
    exit(1)

# ------------------------------ DataLoader 创建 ------------------------------
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ------------------------------ 模型初始化 ------------------------------
# 根据 all_label_ids 顺序构造描述列表（前4个类别）
label_descriptions_list = [LABEL_DESCRIPTIONS_ALL[label_id] for label_id in all_label_ids]
num_labels = len(all_label_ids)
model = UnixcoderForClassification(MODEL_NAME, num_labels, TEMPERATURE, label_descriptions=label_descriptions_list)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
criterion = LabelContrastiveLoss(temperature=TEMPERATURE)

# ------------------------------ 训练与评估函数 ------------------------------
def train_model(model, train_loader, val_loader, optimizer, criterion, device, num_epochs, model_save_path):
    best_val_accuracy = 0.0
    best_model_state = None

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            code_embeddings = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(code_embeddings, model.label_embeddings, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        logging.info(f"Epoch {epoch+1}/{num_epochs} - Average Training Loss: {avg_train_loss:.4f}")

        val_accuracy = evaluate_model(model, val_loader, device, mode='validation')
        logging.info(f"Epoch {epoch+1}/{num_epochs} - Validation Accuracy: {val_accuracy:.4f}")

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model_state = model.state_dict()
            torch.save(best_model_state, model_save_path)
            logging.info(f"保存最佳模型，验证准确率: {best_val_accuracy:.4f}")

    if best_model_state:
        model.load_state_dict(best_model_state)
        logging.info(f"加载最佳模型，验证准确率: {best_val_accuracy:.4f}")

def evaluate_model(model, dataloader, device, mode='validation'):
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        label_embeddings = nn.functional.normalize(model.label_embeddings, dim=1)
        for batch in tqdm(dataloader, desc=f"Evaluating on {mode.capitalize()} Set"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].cpu().numpy()

            code_embeddings = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = torch.matmul(code_embeddings, label_embeddings.t()) / model.temperature
            predictions = torch.argmax(logits, dim=1).cpu().numpy()

            y_true.extend([all_index_to_label[idx] for idx in labels])
            y_pred.extend([all_index_to_label[idx] for idx in predictions])

    accuracy = accuracy_score(y_true, y_pred)

    if mode == 'test':
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=all_label_ids, zero_division=0
        )
        conf_mat = confusion_matrix(y_true, y_pred, labels=all_label_ids)
        per_class_accuracy = {}
        for idx, label_id in enumerate(all_label_ids):
            TP = conf_mat[idx, idx]
            support = conf_mat[idx].sum()
            accuracy_i = TP / support if support > 0 else 0.0
            per_class_accuracy[label_id] = accuracy_i

        metrics_df = pd.DataFrame({
            'CWE ID': all_label_ids,
            'Description': [LABEL_DESCRIPTIONS_ALL[label_id] for label_id in all_label_ids],
            'Accuracy': [per_class_accuracy[label_id] for label_id in all_label_ids],
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1
        })
        overall_precision, overall_recall, overall_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )
        overall_metrics_df = pd.DataFrame({
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
            'Value': [accuracy, overall_precision, overall_recall, overall_f1]
        })

        print(f'\nTest Accuracy: {accuracy:.4f}')
        print(f'Weighted Precision: {overall_precision:.4f}')
        print(f'Weighted Recall: {overall_recall:.4f}')
        print(f'Weighted F1-Score: {overall_f1:.4f}\n')
        print("Per-class Evaluation Metrics:")
        print(metrics_df.to_string(index=False))
        print("\nOverall Evaluation Metrics:")
        print(overall_metrics_df.to_string(index=False))
        print("\nClassification Report:")
        target_names = [LABEL_DESCRIPTIONS_ALL[label_id] for label_id in all_label_ids]
        report = classification_report(
            y_true, y_pred, labels=all_label_ids, zero_division=0,
            target_names=target_names
        )
        print(report)
    elif mode == 'validation':
        return accuracy

# ------------------------------ 训练与测试 ------------------------------
train_model(model, train_dataloader, val_dataloader, optimizer, criterion, device, NUM_EPOCHS, MODEL_SAVE_PATH)
evaluate_model(model, test_dataloader, device, mode='test')
