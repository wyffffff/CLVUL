import os
import re
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split, Subset, ConcatDataset
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import numpy as np
import logging
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import random
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau

# ============================
# 配置和超参数设置
# ============================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

set_seed()

# 超参数
MODEL_NAME = "unixcoder-base-nine"  # 请根据需要调整
BATCH_SIZE_PHASE1 = 32  # 第一阶段批次大小
BATCH_SIZE_PHASE2 = 32  # 第二阶段批次大小
BATCH_SIZE_PHASE3 = 32  # 第三阶段批次大小
LEARNING_RATE = 1e-5
NUM_EPOCHS = 5
MAX_LENGTH = 512
TEMPERATURE = 0.07
HIDDEN_SIZE = 256
ACCUMULATION_STEPS = 1
LAMBDA_EWC = 0.1
PATIENCE = 3

# 模型保存路径
MODEL_SAVE_PATH_PHASE1 = "best_model_phase1.pt"
MODEL_SAVE_PATH_PHASE2 = "best_model_phase2.pt"
MODEL_SAVE_PATH_PHASE3 = "best_model_phase3.pt"

# ============================
# CWE 类别定义
# ============================
training_cwe_phase1 = [
    {'id': 476, 'description': "NULL Pointer Dereference"},
    {'id': 119, 'description': "Buffer Overflow"},
    {'id': 787, 'description': "Out-of-bounds Write"},
    {'id': 416, 'description': "Use After Free"}
]

additional_cwe_phase2 = [
    {'id': 125, 'description': "Out-of-bounds Read"},
    {'id': 20,  'description': "Improper Input Validation"},
    {'id': 401, 'description': "Missing Release of Memory after Effective Lifetime"},
    {'id': 200, 'description': "Exposure of Sensitive Information to an Unauthorized Actor"}
]

additional_cwe_phase3 = [
    {'id': 399, 'description': "Resource Management Errors"},
    {'id': 264, 'description': "Permissions, Privileges, and Access Control"},
    {'id': 190, 'description': "Integer Overflow or Wraparound"},
    {'id': 362, 'description': "Concurrent Execution using Shared Resource with Improper Synchronization"}
]

# 合并所有阶段的 CWE
all_cwe_phase1 = training_cwe_phase1
all_cwe_phase2 = training_cwe_phase1 + additional_cwe_phase2
all_cwe_phase3 = training_cwe_phase1 + additional_cwe_phase2 + additional_cwe_phase3

# 创建标签描述字典
LABEL_DESCRIPTIONS_PHASE1 = {cwe['id']: cwe['description'] for cwe in training_cwe_phase1}
LABEL_DESCRIPTIONS_PHASE2 = {cwe['id']: cwe['description'] for cwe in all_cwe_phase2}
LABEL_DESCRIPTIONS_PHASE3 = {cwe['id']: cwe['description'] for cwe in all_cwe_phase3}

# 获取标签 ID 列表和标签到索引的映射
def get_label_mappings(cwe_list):
    label_ids = [cwe['id'] for cwe in cwe_list]
    label_id_to_index = {label_id: idx for idx, label_id in enumerate(label_ids)}
    index_to_label_id = {idx: label_id for idx, label_id in enumerate(label_ids)}
    return label_ids, label_id_to_index, index_to_label_id

label_ids_phase1, label_id_to_index_phase1, index_to_label_id_phase1 = get_label_mappings(all_cwe_phase1)
label_ids_phase2, label_id_to_index_phase2, index_to_label_id_phase2 = get_label_mappings(all_cwe_phase2)
label_ids_phase3, label_id_to_index_phase3, index_to_label_id_phase3 = get_label_mappings(all_cwe_phase3)

logging.info(f"Phase1 Labels: {label_ids_phase1}")
logging.info(f"Phase2 Labels: {label_ids_phase2}")
logging.info(f"Phase3 Labels: {label_ids_phase3}")

# ============================
# CombinedDataset 类定义
# ============================
class CombinedDataset(Dataset):
    def __init__(self, datasets):
        self.datasets = datasets
        self.data = []
        self.labels = []
        for dataset in datasets:
            if isinstance(dataset, Subset):
                # 从 Subset 中提取数据
                for idx in dataset.indices:
                    item = dataset.dataset[idx]
                    self.data.append(item)
                    self.labels.append(item['label_id'])
            else:
                for idx in range(len(dataset)):
                    item = dataset[idx]
                    self.data.append(item)
                    self.labels.append(item['label_id'])
        logging.info(f"CombinedDataset total samples: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# ============================
# 数据集类定义
# ============================
class CodeDefectDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length, selected_cwe, label_descriptions, label_id_to_index, augment=False):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.selected_cwe = selected_cwe
        self.label_descriptions = label_descriptions
        self.label_id_to_index = label_id_to_index
        self.augment = augment

        dataframe = dataframe.copy()
        if 'CWE_ID' not in dataframe.columns or 'Func' not in dataframe.columns:
            raise ValueError("Dataframe must contain 'CWE_ID' and 'Func' columns.")
        
        if dataframe['CWE_ID'].dtype == object:
            # 假设 CWE_ID 以 'CWE-xxx' 的形式存在
            dataframe['CWE_ID'] = dataframe['CWE_ID'].str.replace('CWE-', '', regex=False)
            dataframe['CWE_ID'] = pd.to_numeric(dataframe['CWE_ID'], errors='coerce')
            dataframe = dataframe[pd.notnull(dataframe['CWE_ID'])].copy()
            dataframe['CWE_ID'] = dataframe['CWE_ID'].astype(int)

        filtered_df = dataframe[dataframe['CWE_ID'].isin(selected_cwe)].reset_index(drop=True)
        self.labels = filtered_df['CWE_ID'].values
        self.codes = filtered_df['Func'].values

        logging.info(f"Dataset selected CWE categories sample count: {len(self.labels)}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        code = str(self.codes[idx])
        cwe_id = int(self.labels[idx])

        label_description = self.label_descriptions[cwe_id]
        label_index = self.label_id_to_index[cwe_id]

        code_encoding = self.tokenizer(
            code,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'code_input_ids': code_encoding['input_ids'].squeeze(0),
            'code_attention_mask': code_encoding['attention_mask'].squeeze(0),
            'label_id': label_index
        }

# ============================
# 模型类定义
# ============================
class CodeTextEmbeddingModel(nn.Module):
    def __init__(self, model_name, temperature, hidden_size):
        super(CodeTextEmbeddingModel, self).__init__()
        self.code_encoder = AutoModel.from_pretrained(model_name)
        self.text_encoder = AutoModel.from_pretrained(model_name)
        self.temperature = temperature

        self.code_projection = nn.Sequential(
            nn.Linear(self.code_encoder.config.hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )

        self.text_projection = nn.Sequential(
            nn.Linear(self.text_encoder.config.hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )

    def forward(self, code_input_ids, code_attention_mask):
        code_outputs = self.code_encoder(input_ids=code_input_ids, attention_mask=code_attention_mask)
        code_embeddings = code_outputs.last_hidden_state[:, 0, :]  # CLS token
        code_embeddings = self.code_projection(code_embeddings)
        code_embeddings = nn.functional.normalize(code_embeddings, dim=1)
        return code_embeddings

    def encode_text(self, text_input_ids, text_attention_mask):
        text_outputs = self.text_encoder(input_ids=text_input_ids, attention_mask=text_attention_mask)
        text_embeddings = text_outputs.last_hidden_state[:, 0, :]  # CLS token
        text_embeddings = self.text_projection(text_embeddings)
        text_embeddings = nn.functional.normalize(text_embeddings, dim=1)
        return text_embeddings

# ============================
# EWC 类定义
# ============================
class EWC:
    def __init__(self, model, dataloader, device, label_embeddings_dict, lambda_ewc=1):
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.lambda_ewc = lambda_ewc
        self.label_embeddings_dict = label_embeddings_dict
        self.params = {n: p.clone().detach() for n, p in self.model.named_parameters() if p.requires_grad}
        self.fisher = self.compute_fisher()

    def compute_fisher(self):
        fisher = {}
        self.model.eval()
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                fisher[n] = torch.zeros_like(p)
        
        loss_fn = nn.CrossEntropyLoss()
        
        # 获取所有标签的嵌入向量
        all_label_ids = list(self.label_embeddings_dict.keys())
        all_label_embeddings = torch.stack([self.label_embeddings_dict[label_id] for label_id in all_label_ids]).to(self.device)  # [num_labels, hidden_size]
        all_label_embeddings = nn.functional.normalize(all_label_embeddings, dim=1)  # 归一化
        
        for data in tqdm(self.dataloader, desc="Computing Fisher Information"):
            self.model.zero_grad()
            code_input_ids = data['code_input_ids'].to(self.device)
            code_attention_mask = data['code_attention_mask'].to(self.device)
            labels = data['label_id'].to(self.device)  # [batch_size]

            # 生成函数嵌入
            code_embeddings = self.model(code_input_ids, code_attention_mask)  # [batch_size, hidden_size]
            code_embeddings = nn.functional.normalize(code_embeddings, dim=1)  # 归一化

            # 计算相似度作为logits
            logits = torch.matmul(code_embeddings, all_label_embeddings.T)  # [batch_size, num_labels]

            loss = loss_fn(logits, labels)
            loss.backward()

            for n, p in self.model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher[n] += p.grad.data ** 2

        for n in fisher:
            fisher[n] = fisher[n] / len(self.dataloader.dataset)
        return fisher

    def penalty(self, model):
        loss = 0
        for n, p in model.named_parameters():
            if p.requires_grad:
                loss += (self.fisher[n] * (p - self.params[n]) ** 2).sum()
        return self.lambda_ewc * loss

# ============================
# 标签嵌入准备函数
# ============================
def prepare_label_embeddings_dict(tokenizer, model, device, labels, label_descriptions, cache_path=None):
    label_embeddings = {}
    if cache_path and os.path.exists(cache_path):
        cached_embeddings = torch.load(cache_path)
        for label_id, embedding in zip(labels, cached_embeddings):
            label_embeddings[label_id] = embedding.to(device)
        logging.info(f"加载缓存的标签嵌入: {cache_path}")
    else:
        for label_id in labels:
            description = label_descriptions[label_id]
            encoding = tokenizer(
                description,
                add_special_tokens=True,
                max_length=MAX_LENGTH // 4,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)

            with torch.no_grad():
                embedding = model.encode_text(input_ids, attention_mask)
            label_embeddings[label_id] = embedding.squeeze(0)  # 移除批次维度
        if cache_path:
            all_embeddings = torch.stack([label_embeddings[label_id] for label_id in labels])
            torch.save(all_embeddings, cache_path)
            logging.info(f"保存标签嵌入到缓存: {cache_path}")
    return label_embeddings

def prepare_all_label_embeddings_tensor(tokenizer, model, device, labels, label_descriptions, cache_path=None):
    if cache_path and os.path.exists(cache_path):
        label_embeddings = torch.load(cache_path).to(device)
        logging.info(f"加载缓存的所有标签嵌入: {cache_path}")
    else:
        all_label_descriptions = [label_descriptions[label_id] for label_id in labels]
        label_encoding = tokenizer(
            all_label_descriptions,
            add_special_tokens=True,
            max_length=MAX_LENGTH // 4,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        label_input_ids = label_encoding['input_ids'].to(device)
        label_attention_mask = label_encoding['attention_mask'].to(device)

        with torch.no_grad():
            label_embeddings = model.encode_text(label_input_ids, label_attention_mask)  # [num_labels, hidden_size]
        if cache_path:
            torch.save(label_embeddings, cache_path)
            logging.info(f"保存所有标签嵌入到缓存: {cache_path}")
    return label_embeddings  # [num_labels, hidden_size]

# ============================
# 辅助函数：加载部分 state_dict
# ============================
def load_partial_state_dict(model, checkpoint_path, exclude_layers=None):
    """
    Load a partial state_dict into the model, excluding specified layers.

    Args:
        model (nn.Module): Target model.
        checkpoint_path (str): Path to the checkpoint file.
        exclude_layers (list of str): List of layer name prefixes to exclude.
    """
    if exclude_layers is None:
        exclude_layers = []
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    filtered_checkpoint = {k: v for k, v in checkpoint.items() if not any(k.startswith(layer) for layer in exclude_layers)}
    model.load_state_dict(filtered_checkpoint, strict=False)
    logging.info(f"Loaded weights from {checkpoint_path}, excluding layers: {exclude_layers}")

# ============================
# 评估函数定义
# ============================
def evaluate_with_similarity(model, dataloader, device, label_embeddings_dict, all_label_ids, index_to_label_id, label_descriptions):
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating with Similarity"):
            code_input_ids = batch['code_input_ids'].to(device)
            code_attention_mask = batch['code_attention_mask'].to(device)
            labels = batch['label_id'].cpu().numpy()  # 标签索引

            # 生成函数嵌入
            code_embeddings = model(code_input_ids, code_attention_mask)  # [batch_size, hidden_size]

            # 准备标签嵌入
            label_embeddings = torch.stack([label_embeddings_dict[label_id] for label_id in all_label_ids]).to(device)  # [num_labels, hidden_size]
            label_embeddings = nn.functional.normalize(label_embeddings, dim=1)  # 归一化

            # 计算相似度（余弦相似度）
            similarity = torch.matmul(code_embeddings, label_embeddings.T)  # [batch_size, num_labels]

            # 获取最大相似度的索引
            _, predicted_indices = similarity.max(dim=1)
            predicted_labels = [all_label_ids[idx.item()] for idx in predicted_indices]

            # 将标签索引映射为实际的 CWE ID
            label_ids_true = [all_label_ids[idx] for idx in labels]

            y_true.extend(label_ids_true)
            y_pred.extend(predicted_labels)

    # 计算总体准确率
    accuracy = accuracy_score(y_true, y_pred)

    # 计算混淆矩阵
    try:
        conf_mat = confusion_matrix(y_true, y_pred, labels=all_label_ids)
    except ValueError as e:
        logging.error(f"Confusion matrix error: {e}")
        logging.error(f"y_true labels: {set(y_true)}")
        logging.error(f"all_label_ids: {set(all_label_ids)}")
        raise e

    # 计算每个类别的准确率（TP / (TP + FN)）
    per_class_accuracy = {}
    for idx, label_id in enumerate(all_label_ids):
        TP = conf_mat[idx, idx]
        support = conf_mat[idx].sum()
        if support > 0:
            accuracy_i = TP / support
        else:
            accuracy_i = 0.0
        per_class_accuracy[label_id] = accuracy_i

    # 计算每个类别的精确率、召回率和F1分数
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=all_label_ids, zero_division=0
    )

    # 创建一个 DataFrame 来存储每个类别的评估指标
    metrics_df = pd.DataFrame({
        'CWE ID': all_label_ids,
        'Description': [label_descriptions[label_id] for label_id in all_label_ids],
        'Accuracy': [per_class_accuracy[label_id] for label_id in all_label_ids],
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1
    })

    # 打印总体指标
    print(f'\nTest Accuracy: {accuracy:.4f}\n')

    # 打印每个类别的指标
    print("Per-class Evaluation Metrics:")
    print(metrics_df.to_string(index=False))

    # 打印分类报告
    print("\nClassification Report:")
    report = classification_report(
        y_true, y_pred, zero_division=0,
        target_names=[label_descriptions[label_id] for label_id in all_label_ids]
    )
    print(report)

    return accuracy

# ============================
# 训练函数定义
# ============================
def train_with_similarity(model, train_dataloader, val_dataloader, optimizer, scheduler, device, num_epochs, label_embeddings_dict, all_label_ids, index_to_label_id, label_descriptions, model_save_path, ewc=None, scaler=None):
    best_accuracy = 0.0
    best_model_state = None
    counter = 0  # 早停计数器

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for i, batch in enumerate(tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}/{num_epochs}")):
            code_input_ids = batch['code_input_ids'].to(device)
            code_attention_mask = batch['code_attention_mask'].to(device)
            labels = batch['label_id'].to(device)  # [batch_size]

            optimizer.zero_grad()

            with autocast():
                # 生成函数嵌入
                code_embeddings = model(code_input_ids, code_attention_mask)  # [batch_size, hidden_size]
                code_embeddings = nn.functional.normalize(code_embeddings, dim=1)  # 归一化

                # 获取对应的标签嵌入
                label_ids = [index_to_label_id[idx.item()] for idx in labels]
                text_embeddings = torch.stack([label_embeddings_dict[label_id] for label_id in label_ids]).to(device)  # [batch_size, hidden_size]
                text_embeddings = nn.functional.normalize(text_embeddings, dim=1)  # 归一化

                # 计算相似度（余弦相似度）
                similarity = torch.matmul(code_embeddings, text_embeddings.T)  # [batch_size, batch_size]

                # 创建标签索引
                target = torch.arange(similarity.size(0)).to(device)

                # 计算损失（交叉熵）
                loss_fn = nn.CrossEntropyLoss()
                loss = loss_fn(similarity, target)

                if ewc:
                    loss += ewc.penalty(model)

                loss = loss / ACCUMULATION_STEPS  # 梯度累积

            scaler.scale(loss).backward()

            if (i + 1) % ACCUMULATION_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            total_loss += loss.item() * ACCUMULATION_STEPS

        avg_loss = total_loss / len(train_dataloader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

        # 评估验证集
        accuracy = evaluate_with_similarity(model, val_dataloader, device, label_embeddings_dict, all_label_ids, index_to_label_id, label_descriptions)
        print(f'Epoch [{epoch+1}/{num_epochs}], Validation Accuracy: {accuracy:.4f}')

        # 更新学习率调度器
        if scheduler:
            scheduler.step(accuracy)

        # 早停和模型保存
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_state = model.state_dict()
            torch.save(best_model_state, model_save_path)
            print(f'最佳模型已保存，准确率: {best_accuracy:.4f}')
            counter = 0
        else:
            counter += 1
            if counter >= PATIENCE:
                print("早停触发，停止训练")
                break

    if best_model_state:
        model.load_state_dict(best_model_state)
        print(f'最佳模型已加载，准确率: {best_accuracy:.4f}')

# ============================
# 数据预处理函数
# ============================
def preprocess_dataframe(dataframe):
    valid_rows = []
    for idx, row in dataframe.iterrows():
        code_str = str(row['Func'])
        try:
            if code_str.strip() == "":
                raise ValueError("Empty code")
            # 可以添加更多C++代码有效性的检查
            valid_rows.append(row)
        except Exception as e:
            logging.warning(f"Invalid code at index {idx}, skipping. Reason: {e}")
    return pd.DataFrame(valid_rows)

# ============================
# 冻结模型参数函数
# ============================
def freeze_model(model, freeze_until_layer=6):
    """
    冻结模型的部分层参数。

    Args:
        model (nn.Module): 需要冻结参数的模型。
        freeze_until_layer (int): 冻结直到的层编号（不包括该编号）。
    """
    # 定义匹配层编号的正则表达式模式
    pattern = r'.*layer\.(\d+)\..*'

    # 冻结 code_encoder 中的参数
    for name, param in model.code_encoder.named_parameters():
        match = re.match(pattern, name)
        if match:
            layer_num = int(match.group(1))
            if layer_num < freeze_until_layer:
                param.requires_grad = False
                logging.debug(f"Frozen parameter: {name}")

    # 冻结 text_encoder 中的参数
    for name, param in model.text_encoder.named_parameters():
        match = re.match(pattern, name)
        if match:
            layer_num = int(match.group(1))
            if layer_num < freeze_until_layer:
                param.requires_grad = False
                logging.debug(f"Frozen parameter: {name}")

# ============================
# 创建数据加载器函数（修改后的版本）
# ============================
def create_dataloaders(dataframe, tokenizer, max_length, current_cwe, previous_cwe=None, previous_ratio=0.1, label_descriptions=None, label_id_to_index=None, batch_size=32, augment=False):
    """
    创建训练集和验证集的 DataLoader，包含当前阶段的CWE类别数据和之前阶段的CWE类别数据的指定比例。
    确保当前阶段的数据也被按比例划分到训练集和验证集中。
    
    Args:
        dataframe (pd.DataFrame): 原始数据集。
        tokenizer: 分词器。
        max_length (int): 最大序列长度。
        current_cwe (list): 当前阶段的CWE ID 列表。
        previous_cwe (list, optional): 之前阶段的CWE ID 列表。默认为 None。
        previous_ratio (float, optional): 之前阶段数据在训练集和验证集中所占的比例。默认为 0.1。
        label_descriptions (dict): 标签描述字典。
        label_id_to_index (dict): 标签ID到索引的映射。
        batch_size (int, optional): 批次大小。默认为 32。
        augment (bool, optional): 是否进行数据增强。默认为 False。

    Returns:
        train_dataloader (DataLoader): 训练集的数据加载器。
        val_dataloader (DataLoader): 验证集的数据加载器。
    """
    # 当前阶段的数据集
    current_dataset_full = CodeDefectDataset(
        dataframe,
        tokenizer,
        max_length,
        current_cwe,
        label_descriptions,
        label_id_to_index,
        augment=augment
    )

    # 按比例划分当前阶段的数据
    current_train_size = int(0.8 * len(current_dataset_full))
    current_val_size = len(current_dataset_full) - current_train_size
    current_train, current_val = random_split(
        current_dataset_full,
        [current_train_size, current_val_size],
        generator=torch.Generator().manual_seed(42)
    )

    datasets_train = [current_train]
    datasets_val = [current_val]

    # 如果有前一阶段的数据，采样指定比例并加入
    if previous_cwe and previous_ratio > 0:
        previous_dataset_full = CodeDefectDataset(
            dataframe,
            tokenizer,
            max_length,
            previous_cwe,
            label_descriptions,
            label_id_to_index,
            augment=False
        )
        # 计算采样数量
        sample_size = int(len(previous_dataset_full) * previous_ratio)
        if sample_size > 0:
            # 随机采样 sample_size 个索引
            sampled_indices = random.sample(range(len(previous_dataset_full)), sample_size)
            sampled_subset = Subset(previous_dataset_full, sampled_indices)

            # 将采样后的子集拆分为训练集和验证集
            train_size_prev = int(0.8 * sample_size)
            val_size_prev = sample_size - train_size_prev
            previous_train, previous_val = random_split(
                sampled_subset,
                [train_size_prev, val_size_prev],
                generator=torch.Generator().manual_seed(42)
            )
            datasets_train.append(previous_train)
            datasets_val.append(previous_val)
            logging.info(f"Added {sample_size} samples from previous CWE IDs to training and validation sets.")
        else:
            logging.warning("Previous ratio too small, no previous data added.")

    # 合并数据集
    train_dataset = CombinedDataset(datasets_train)
    val_dataset = CombinedDataset(datasets_val)

    logging.info(f"Training set size: {len(train_dataset)}")
    logging.info(f"Validation set size: {len(val_dataset)}")

    # 创建标准的 DataLoader
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True  # 打乱训练集
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False  # 验证集不打乱
    )

    return train_dataloader, val_dataloader

# ============================
# 主训练流程
# ============================
def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    dataframe = pd.read_csv('CWEID_Function_vul10.4.csv')  # 请确保 CSV 文件路径正确

    # 预处理数据集，过滤无效代码
    dataframe = preprocess_dataframe(dataframe)

    # 划分数据集为训练+验证 和 测试 两部分
    test_split_ratio = 0.1
    total_size = len(dataframe)
    test_size = int(test_split_ratio * total_size)
    train_val_size = total_size - test_size

    train_val_df, test_df = train_test_split(
        dataframe,
        test_size=test_split_ratio,
        random_state=42,
        shuffle=True
    )

    logging.info(f"总数据集大小: {total_size}")
    logging.info(f"训练+验证集大小: {len(train_val_df)}")
    logging.info(f"测试集大小: {len(test_df)}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ###########################
    # 第一阶段训练 Phase1
    ###########################
    logging.info("开始第一阶段训练 Phase1")

    # 创建训练集和验证集的 dataloader
    train_dataloader_phase1, val_dataloader_phase1 = create_dataloaders(
        dataframe=train_val_df,
        tokenizer=tokenizer,
        max_length=MAX_LENGTH,
        current_cwe=[cwe['id'] for cwe in training_cwe_phase1],
        previous_cwe=None,
        previous_ratio=0.0,
        label_descriptions=LABEL_DESCRIPTIONS_PHASE1,
        label_id_to_index=label_id_to_index_phase1,
        batch_size=BATCH_SIZE_PHASE1,
        augment=False  # 禁用数据增强
    )

    # 创建测试集数据集（仅第一阶段的 CWE）
    test_dataset_phase1 = CodeDefectDataset(
        test_df,
        tokenizer,
        MAX_LENGTH,
        [cwe['id'] for cwe in training_cwe_phase1],
        LABEL_DESCRIPTIONS_PHASE1,
        label_id_to_index_phase1,
        augment=False
    )

    test_dataloader_phase1 = DataLoader(test_dataset_phase1, batch_size=BATCH_SIZE_PHASE1, shuffle=False)
    logging.info(f"测试集大小（Phase1）: {len(test_dataset_phase1)}")

    # 初始化模型 Phase1
    model_phase1 = CodeTextEmbeddingModel(MODEL_NAME, TEMPERATURE, HIDDEN_SIZE).to(device)
    optimizer_phase1 = optim.AdamW(model_phase1.parameters(), lr=LEARNING_RATE)
    scheduler_phase1 = ReduceLROnPlateau(optimizer_phase1, mode='max', factor=0.1, patience=2, verbose=True)

    # 准备训练标签嵌入（Phase1）
    label_embeddings_cache_phase1 = 'label_embeddings_phase1.pt'
    train_label_embeddings_dict_phase1 = prepare_label_embeddings_dict(
        tokenizer, model_phase1, device, label_ids_phase1, LABEL_DESCRIPTIONS_PHASE1, cache_path=label_embeddings_cache_phase1
    )

    # 确保所有 CWE ID 都存在于 label_embeddings_dict 中
    missing_labels_phase1 = set(label_ids_phase1) - set(train_label_embeddings_dict_phase1.keys())
    if missing_labels_phase1:
        logging.error(f"Missing label embeddings for Phase1 CWE IDs: {missing_labels_phase1}")
        raise KeyError(f"Missing label embeddings for Phase1 CWE IDs: {missing_labels_phase1}")

    # 准备测试标签嵌入（Phase1）
    label_embeddings_all_cache_phase1 = 'all_label_embeddings_phase1.pt'
    all_label_embeddings_phase1 = prepare_all_label_embeddings_tensor(
        tokenizer, model_phase1, device, label_ids_phase1, LABEL_DESCRIPTIONS_PHASE1, cache_path=label_embeddings_all_cache_phase1
    )

    # 初始化混合精度训练
    scaler_phase1 = GradScaler()

    # 第一阶段不需要EWC
    ewc_phase1 = None

    # 训练模型（Phase1）
    train_with_similarity(
        model=model_phase1,
        train_dataloader=train_dataloader_phase1,
        val_dataloader=val_dataloader_phase1,
        optimizer=optimizer_phase1,
        scheduler=scheduler_phase1,
        device=device,
        num_epochs=NUM_EPOCHS,
        label_embeddings_dict=train_label_embeddings_dict_phase1,
        all_label_ids=label_ids_phase1,
        index_to_label_id=index_to_label_id_phase1,
        label_descriptions=LABEL_DESCRIPTIONS_PHASE1,
        model_save_path=MODEL_SAVE_PATH_PHASE1,
        ewc=ewc_phase1,
        scaler=scaler_phase1
    )

    # 评估模型（Phase1）
    evaluate_with_similarity(
        model_phase1,
        test_dataloader_phase1,
        device,
        train_label_embeddings_dict_phase1,
        label_ids_phase1,
        index_to_label_id_phase1,
        LABEL_DESCRIPTIONS_PHASE1
    )

    ###########################
    # 第二阶段训练 Phase2
    ###########################
    logging.info("开始第二阶段训练 Phase2")

    # 选择 Phase2 的CWE类别
    current_cwe_phase2 = [cwe['id'] for cwe in additional_cwe_phase2]
    previous_cwe_phase2 = [cwe['id'] for cwe in training_cwe_phase1]

    # 创建训练集和验证集的 dataloader（Phase2的CWE + Phase1的10%数据）
    train_dataloader_phase2, val_dataloader_phase2 = create_dataloaders(
        dataframe=train_val_df,
        tokenizer=tokenizer,
        max_length=MAX_LENGTH,
        current_cwe=current_cwe_phase2,
        previous_cwe=previous_cwe_phase2,
        previous_ratio=0.1,
        label_descriptions=LABEL_DESCRIPTIONS_PHASE2,
        label_id_to_index=label_id_to_index_phase2,
        batch_size=BATCH_SIZE_PHASE2,
        augment=False  # 禁用数据增强
    )

    # 创建测试集数据集（Phase1和Phase2的CWE）
    test_dataset_phase2 = CodeDefectDataset(
        test_df,
        tokenizer,
        MAX_LENGTH,
        [cwe['id'] for cwe in all_cwe_phase2],
        LABEL_DESCRIPTIONS_PHASE2,
        label_id_to_index_phase2,
        augment=False
    )

    test_dataloader_phase2 = DataLoader(test_dataset_phase2, batch_size=BATCH_SIZE_PHASE2, shuffle=False)
    logging.info(f"测试集大小（Phase2）: {len(test_dataset_phase2)}")

    # 初始化模型 Phase2，类别数为8
    model_phase2 = CodeTextEmbeddingModel(MODEL_NAME, TEMPERATURE, HIDDEN_SIZE).to(device)

    # 加载 Phase1 的权重
    load_partial_state_dict(model_phase2, MODEL_SAVE_PATH_PHASE1, exclude_layers=[])

    # 冻结部分模型参数（例如冻结前6层编码器）
    freeze_model(model_phase2, freeze_until_layer=6)

    optimizer_phase2 = optim.AdamW(filter(lambda p: p.requires_grad, model_phase2.parameters()), lr=LEARNING_RATE)
    scheduler_phase2 = ReduceLROnPlateau(optimizer_phase2, mode='max', factor=0.1, patience=2, verbose=True)

    # 准备训练标签嵌入（Phase2）
    label_embeddings_cache_phase2 = 'label_embeddings_phase2.pt'
    train_label_embeddings_dict_phase2 = prepare_label_embeddings_dict(
        tokenizer, model_phase2, device, label_ids_phase2, LABEL_DESCRIPTIONS_PHASE2, cache_path=label_embeddings_cache_phase2
    )

    # 确保所有 CWE ID 都存在于 label_embeddings_dict 中
    missing_labels_phase2 = set(label_ids_phase2) - set(train_label_embeddings_dict_phase2.keys())
    if missing_labels_phase2:
        logging.error(f"Missing label embeddings for Phase2 CWE IDs: {missing_labels_phase2}")
        raise KeyError(f"Missing label embeddings for Phase2 CWE IDs: {missing_labels_phase2}")

    # 准备测试标签嵌入（Phase2）
    label_embeddings_all_cache_phase2 = 'all_label_embeddings_phase2.pt'
    all_label_embeddings_phase2 = prepare_all_label_embeddings_tensor(
        tokenizer, model_phase2, device, label_ids_phase2, LABEL_DESCRIPTIONS_PHASE2, cache_path=label_embeddings_all_cache_phase2
    )

    # 初始化混合精度训练
    scaler_phase2 = GradScaler()

    # 初始化 EWC（保留第一阶段的知识）
    previous_dataset_phase2_for_ewc = CodeDefectDataset(
        train_val_df,
        tokenizer,
        MAX_LENGTH,
        previous_cwe_phase2,
        LABEL_DESCRIPTIONS_PHASE1,
        label_id_to_index_phase1,
        augment=False
    )
    ewc_phase2 = EWC(
        model_phase2, 
        DataLoader(previous_dataset_phase2_for_ewc, batch_size=1, shuffle=True),
        device,
        label_embeddings_dict=train_label_embeddings_dict_phase1,
        lambda_ewc=LAMBDA_EWC
    )

    # 训练模型（Phase2）
    train_with_similarity(
        model=model_phase2,
        train_dataloader=train_dataloader_phase2,
        val_dataloader=val_dataloader_phase2,
        optimizer=optimizer_phase2,
        scheduler=scheduler_phase2,
        device=device,
        num_epochs=NUM_EPOCHS,
        label_embeddings_dict=train_label_embeddings_dict_phase2,
        all_label_ids=label_ids_phase2,
        index_to_label_id=index_to_label_id_phase2,
        label_descriptions=LABEL_DESCRIPTIONS_PHASE2,
        model_save_path=MODEL_SAVE_PATH_PHASE2,
        ewc=ewc_phase2,
        scaler=scaler_phase2
    )

    # 评估模型（Phase2）
    evaluate_with_similarity(
        model_phase2,
        test_dataloader_phase2,
        device,
        train_label_embeddings_dict_phase2,
        label_ids_phase2,
        index_to_label_id_phase2,
        LABEL_DESCRIPTIONS_PHASE2
    )

    ###########################
    # 第三阶段训练 Phase3
    ###########################
    logging.info("开始第三阶段训练 Phase3")

    # 选择 Phase3 的CWE类别
    current_cwe_phase3 = [cwe['id'] for cwe in additional_cwe_phase3]
    previous_cwe_phase3 = [cwe['id'] for cwe in all_cwe_phase2]

    # 创建训练集和验证集的 dataloader（Phase3的CWE + Phase1和Phase2的10%数据）
    train_dataloader_phase3, val_dataloader_phase3 = create_dataloaders(
        dataframe=train_val_df,
        tokenizer=tokenizer,
        max_length=MAX_LENGTH,
        current_cwe=current_cwe_phase3,
        previous_cwe=previous_cwe_phase3,
        previous_ratio=0.1,
        label_descriptions=LABEL_DESCRIPTIONS_PHASE3,
        label_id_to_index=label_id_to_index_phase3,
        batch_size=BATCH_SIZE_PHASE3,
        augment=False  # 禁用数据增强
    )

    # 创建测试集数据集（Phase1、Phase2和Phase3的CWE）
    test_dataset_phase3 = CodeDefectDataset(
        test_df,
        tokenizer,
        MAX_LENGTH,
        [cwe['id'] for cwe in all_cwe_phase3],
        LABEL_DESCRIPTIONS_PHASE3,
        label_id_to_index_phase3,
        augment=False
    )

    test_dataloader_phase3 = DataLoader(test_dataset_phase3, batch_size=BATCH_SIZE_PHASE3, shuffle=False)
    logging.info(f"测试集大小（Phase3）: {len(test_dataset_phase3)}")

    # 初始化模型 Phase3，类别数为12
    model_phase3 = CodeTextEmbeddingModel(MODEL_NAME, TEMPERATURE, HIDDEN_SIZE).to(device)

    # 加载 Phase2 的权重
    load_partial_state_dict(model_phase3, MODEL_SAVE_PATH_PHASE2, exclude_layers=[])

    # 冻结部分模型参数（例如冻结前6层编码器）
    freeze_model(model_phase3, freeze_until_layer=6)

    optimizer_phase3 = optim.AdamW(filter(lambda p: p.requires_grad, model_phase3.parameters()), lr=LEARNING_RATE)
    scheduler_phase3 = ReduceLROnPlateau(optimizer_phase3, mode='max', factor=0.1, patience=2, verbose=True)

    # 准备训练标签嵌入（Phase3）
    label_embeddings_cache_phase3 = 'label_embeddings_phase3.pt'
    train_label_embeddings_dict_phase3 = prepare_label_embeddings_dict(
        tokenizer, model_phase3, device, label_ids_phase3, LABEL_DESCRIPTIONS_PHASE3, cache_path=label_embeddings_cache_phase3
    )

    # 确保所有 CWE ID 都存在于 label_embeddings_dict 中
    missing_labels_phase3 = set(label_ids_phase3) - set(train_label_embeddings_dict_phase3.keys())
    if missing_labels_phase3:
        logging.error(f"Missing label embeddings for Phase3 CWE IDs: {missing_labels_phase3}")
        raise KeyError(f"Missing label embeddings for Phase3 CWE IDs: {missing_labels_phase3}")

    # 准备测试标签嵌入（Phase3）
    label_embeddings_all_cache_phase3 = 'all_label_embeddings_phase3.pt'
    all_label_embeddings_phase3 = prepare_all_label_embeddings_tensor(
        tokenizer, model_phase3, device, label_ids_phase3, LABEL_DESCRIPTIONS_PHASE3, cache_path=label_embeddings_all_cache_phase3
    )

    # 初始化混合精度训练
    scaler_phase3 = GradScaler()

    # 初始化 EWC（保留前两个阶段的知识）
    concatenated_dataset_phase3 = CombinedDataset([
        train_dataloader_phase1.dataset,
        train_dataloader_phase2.dataset
    ])
    ewc_phase3 = EWC(
        model_phase3, 
        DataLoader(concatenated_dataset_phase3, batch_size=1, shuffle=True),
        device,
        label_embeddings_dict={**train_label_embeddings_dict_phase1, **train_label_embeddings_dict_phase2},
        lambda_ewc=LAMBDA_EWC
    )

    # 训练模型（Phase3）
    train_with_similarity(
        model=model_phase3,
        train_dataloader=train_dataloader_phase3,
        val_dataloader=val_dataloader_phase3,
        optimizer=optimizer_phase3,
        scheduler=scheduler_phase3,
        device=device,
        num_epochs=NUM_EPOCHS,
        label_embeddings_dict=train_label_embeddings_dict_phase3,
        all_label_ids=label_ids_phase3,
        index_to_label_id=index_to_label_id_phase3,
        label_descriptions=LABEL_DESCRIPTIONS_PHASE3,
        model_save_path=MODEL_SAVE_PATH_PHASE3,
        ewc=ewc_phase3,
        scaler=scaler_phase3
    )

    # 评估模型（Phase3）
    evaluate_with_similarity(
        model_phase3,
        test_dataloader_phase3,
        device,
        train_label_embeddings_dict_phase3,
        label_ids_phase3,
        index_to_label_id_phase3,
        LABEL_DESCRIPTIONS_PHASE3
    )

# ============================
# 运行主函数
# ============================
if __name__ == "__main__":
    main()
