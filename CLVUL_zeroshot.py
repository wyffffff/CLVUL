import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split, Sampler
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import numpy as np
import logging
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
import random
from collections import defaultdict
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
import pickle
import os
from itertools import cycle  # 导入 cycle 函数

# 设置日志配置
logging.basicConfig(level=logging.INFO)

# 设置随机种子
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

set_seed()

# 超参数设置
CODE_MODEL_NAME = "unixcoder-base-nine"  # 代码编码器，替换为正确的 UniXcoder 模型名称
TEXT_MODEL_NAME = "bert-base-uncased"  # 文本编码器也使用 UniXcoder
BATCH_SIZE = 8  # 从6更改为8，以匹配新增的8个 CWE 类别
LEARNING_RATE = 1e-5
NUM_EPOCHS = 3
MAX_LENGTH = 512  # 根据显存情况调整，UnixCoder通常使用较短的长度
TEMPERATURE = 0.07
HIDDEN_SIZE = 256
MODEL_SAVE_PATH = "best_model.pth"
COEFFICIENTS_SAVE_PATH = "zero_shot_coefficients.pkl"
K = 2  # 基准向量的数量

# 前八个 CWE 类别用于训练
training_cwe = [
    {'id': 476, 'description': "NULL Pointer Dereference"},
    {'id': 119, 'description': "Buffer Overflow"},
    {'id': 787, 'description': "Out-of-bounds Write"},
    {'id': 416, 'description': "Use After Free"},
    {'id': 401, 'description': "Memory Leak"},
    {'id': 200, 'description': "Information Exposure"},
    {'id': 399, 'description': "Resource Management Errors"},
    {'id': 264, 'description': "Permissions, Privileges, and Access Controls"}
]

# 保持零样本 CWE 不变（可根据需要调整）
zero_shot_cwe = [
    {'id': 125, 'description': "Out-of-bounds Read"},
    {'id': 20, 'description': "Improper Input Validation"}
]

# 合并所有 CWE 类别（仅用于定义标签映射）
all_cwe = training_cwe + zero_shot_cwe

# 创建标签描述字典
LABEL_DESCRIPTIONS_TRAIN = {cwe['id']: cwe['description'] for cwe in training_cwe}
LABEL_DESCRIPTIONS_ALL = {cwe['id']: cwe['description'] for cwe in all_cwe}

# 获取训练集标签 ID 列表和标签到索引的映射
train_label_ids = [cwe['id'] for cwe in training_cwe]
train_label_id_to_index = {label_id: idx for idx, label_id in enumerate(train_label_ids)}
train_index_to_label_id = {idx: label_id for idx, label_id in enumerate(train_label_ids)}

# 获取所有标签 ID 列表和标签到索引的映射（用于测试集，包括 Zero-Shot）
all_label_ids = [cwe['id'] for cwe in all_cwe]
all_label_id_to_index = {label_id: idx for idx, label_id in enumerate(all_label_ids)}
all_index_to_label_id = {idx: label_id for idx, label_id in enumerate(all_label_ids)}

logging.info(f"训练标签到索引映射: {train_label_id_to_index}")
logging.info(f"所有标签到索引映射: {all_label_id_to_index}")

# 数据集类定义
class CodeDefectDataset(Dataset):
    def __init__(self, dataframe, tokenizer_code, tokenizer_text, max_length, selected_cwe, label_descriptions, label_id_to_index):
        self.tokenizer_code = tokenizer_code
        self.tokenizer_text = tokenizer_text
        self.max_length = max_length
        self.selected_cwe = selected_cwe
        self.label_descriptions = label_descriptions
        self.label_id_to_index = label_id_to_index

        dataframe = dataframe.copy()
        if dataframe['CWE_ID'].dtype == object:
            # 假设 CWE_ID 以 'CWE-xxx' 的形式存在
            dataframe['CWE_ID'] = dataframe['CWE_ID'].str.replace('CWE-', '', regex=False)
            dataframe['CWE_ID'] = pd.to_numeric(dataframe['CWE_ID'], errors='coerce')
            dataframe = dataframe[pd.notnull(dataframe['CWE_ID'])].copy()
            dataframe['CWE_ID'] = dataframe['CWE_ID'].astype(int)

        filtered_df = dataframe[dataframe['CWE_ID'].isin(selected_cwe)].reset_index(drop=True)
        self.labels = filtered_df['CWE_ID'].values
        self.codes = filtered_df['Func'].values

        logging.info(f"数据集中选定的 CWE 类别的样本数量: {len(self.labels)}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        code = str(self.codes[idx])
        cwe_id = int(self.labels[idx])

        label_description = self.label_descriptions[cwe_id]
        label_index = self.label_id_to_index[cwe_id]

        code_encoding = self.tokenizer_code(
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

# 对比学习模型定义，使用 UniXcoder 作为文本编码器
class CodeTextCLIPModel(nn.Module):
    def __init__(self, code_model_name, text_model_name, temperature, hidden_size, K):
        super(CodeTextCLIPModel, self).__init__()
        self.code_encoder = AutoModel.from_pretrained(code_model_name)
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        self.temperature = temperature
        self.K = K

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

        # 初始化K个基准向量（待优化）
        self.basis_vectors = nn.Parameter(torch.randn(K, hidden_size))

    def forward(self, code_input_ids, code_attention_mask):
        code_outputs = self.code_encoder(input_ids=code_input_ids, attention_mask=code_attention_mask)
        code_embeddings = code_outputs.last_hidden_state[:, 0, :]  # [CLS] token
        code_embeddings = self.code_projection(code_embeddings)
        code_embeddings = nn.functional.normalize(code_embeddings, dim=1)
        return code_embeddings

    def encode_text(self, text_input_ids, text_attention_mask):
        text_outputs = self.text_encoder(input_ids=text_input_ids, attention_mask=text_attention_mask)
        text_embeddings = text_outputs.last_hidden_state[:, 0, :]  # [CLS] token
        text_embeddings = self.text_projection(text_embeddings)
        text_embeddings = nn.functional.normalize(text_embeddings, dim=1)
        return text_embeddings

    def generate_zero_shot_embeddings(self, coefficients):
        # coefficients: dict {cwe_id: [a1, a2, a3, ..., aK]}
        zero_shot_embeddings = {}
        for cwe_id, coeffs in coefficients.items():
            coeffs_tensor = torch.tensor(coeffs, dtype=torch.float32).to(self.basis_vectors.device)  # [K]
            new_vector = torch.matmul(coeffs_tensor, self.basis_vectors)  # [hidden_size]
            new_vector = nn.functional.normalize(new_vector, dim=0)
            zero_shot_embeddings[cwe_id] = new_vector.cpu()
        return zero_shot_embeddings

# 对比损失函数定义
def clip_contrastive_loss(code_embeddings, text_embeddings, temperature):
    logits = torch.matmul(code_embeddings, text_embeddings.T) / temperature
    labels = torch.arange(code_embeddings.size(0)).to(code_embeddings.device)
    loss_fn = nn.CrossEntropyLoss()
    loss_code_to_text = loss_fn(logits, labels)
    loss_text_to_code = loss_fn(logits.T, labels)
    loss = (loss_code_to_text + loss_text_to_code) / 2
    return loss

# 自定义 BatchSampler
class GroupedBatchSampler(Sampler):
    def __init__(self, dataset, shuffle=True):
        self.dataset = dataset
        self.shuffle = shuffle

        if hasattr(dataset, 'labels'):
            labels = dataset.labels
        elif isinstance(dataset, torch.utils.data.Subset):
            labels = [dataset.dataset.labels[i] for i in dataset.indices]
        else:
            raise ValueError('Dataset must have labels or be a Subset of a dataset with labels.')

        self.label_to_indices = defaultdict(list)
        for idx, label in enumerate(labels):
            self.label_to_indices[label].append(idx)

        self.labels = list(self.label_to_indices.keys())
        self.num_classes = len(self.labels)
        self.batch_size = self.num_classes  # 动态设置 batch_size 等于类别数

        self.class_sample_counts = {label: len(indices) for label, indices in self.label_to_indices.items()}
        self.max_samples = max(self.class_sample_counts.values())

    def __iter__(self):
        if self.shuffle:
            for label in self.labels:
                random.shuffle(self.label_to_indices[label])
            random.shuffle(self.labels)

        class_iters = {}
        for label in self.labels:
            indices = self.label_to_indices[label]
            if self.shuffle:
                random.shuffle(indices)
            class_iters[label] = cycle(indices)

        for _ in range(self.max_samples):
            batch = []
            for label in self.labels:
                batch.append(next(class_iters[label]))
            yield batch

    def __len__(self):
        return self.max_samples

# 创建数据集和数据加载器
tokenizer_code = AutoTokenizer.from_pretrained(CODE_MODEL_NAME)
tokenizer_text = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)
dataframe = pd.read_csv('CWEID_Function_vul10.4_10000.csv')  # 使用扩充后的数据集

# 划分数据集为训练+验证 和 测试 两部分
from sklearn.model_selection import train_test_split

test_split_ratio = 0.1
train_val_df, test_df = train_test_split(
    dataframe,
    test_size=test_split_ratio,
    random_state=42,
    shuffle=True
)

train_val_dataset = CodeDefectDataset(
    train_val_df,
    tokenizer_code,
    tokenizer_text,
    MAX_LENGTH,
    [cwe['id'] for cwe in training_cwe],
    LABEL_DESCRIPTIONS_TRAIN,
    train_label_id_to_index
)

train_size = int(0.9 * len(train_val_dataset))
val_size = len(train_val_dataset) - train_size
train_dataset, val_dataset = random_split(train_val_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

logging.info(f"训练集大小: {len(train_dataset)}")
logging.info(f"验证集大小: {len(val_dataset)}")

# 创建自定义的 BatchSampler
train_batch_sampler = GroupedBatchSampler(
    train_dataset,
    shuffle=True
)

train_dataloader = DataLoader(
    train_dataset,
    batch_sampler=train_batch_sampler
)

val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

test_dataset = CodeDefectDataset(
    test_df,
    tokenizer_code,
    tokenizer_text,
    MAX_LENGTH,
    all_label_ids,
    LABEL_DESCRIPTIONS_ALL,
    all_label_id_to_index
)

logging.info(f"测试集大小: {len(test_dataset)}")

test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CodeTextCLIPModel(CODE_MODEL_NAME, TEXT_MODEL_NAME, TEMPERATURE, HIDDEN_SIZE, K).to(device)
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# 准备所有标签描述的编码（用于训练和测试）
def prepare_label_embeddings(tokenizer_text, model, device, labels, label_descriptions):
    model.eval()
    label_embeddings = {}
    with torch.no_grad():
        for label_id in labels:
            description = label_descriptions[label_id]
            encoding = tokenizer_text(
                description,
                add_special_tokens=True,
                max_length=MAX_LENGTH // 2,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)

            embedding = model.encode_text(input_ids, attention_mask)
            label_embeddings[label_id] = embedding.squeeze(0).cpu().float().numpy()  # 转换为 float32
    model.train()
    return label_embeddings

# 特征向量分解并保存系数
def preprocess_and_save_coefficients(model, tokenizer_text, device, train_label_ids, zero_shot_cwe_ids, label_descriptions_train, label_descriptions_all, K, save_path=COEFFICIENTS_SAVE_PATH):
    # 编码所有标签描述
    all_label_embeddings = prepare_label_embeddings(tokenizer_text, model, device, all_label_ids, label_descriptions_all)
    train_label_embeddings = prepare_label_embeddings(tokenizer_text, model, device, train_label_ids, label_descriptions_train)
    zero_shot_embeddings = {cwe_id: all_label_embeddings[cwe_id] for cwe_id in zero_shot_cwe_ids}

    # 将训练标签的特征向量堆叠成矩阵 X
    X = np.stack([train_label_embeddings[label_id] for label_id in train_label_ids])  # [num_train_labels, embedding_dim]

    # 使用PCA将训练标签的特征向量分解为K个基准向量
    pca = PCA(n_components=K)
    pca.fit(X)
    basis_vectors = pca.components_  # [K, embedding_dim]

    # 保存PCA模型以便后续使用
    with open("pca_model.pkl", 'wb') as f:
        pickle.dump(pca, f)
    logging.info("PCA模型已保存。")

    # 将训练标签的特征向量表示为基准向量的线性组合
    coefficients = {}
    for cwe_id, z_embedding in zero_shot_embeddings.items():
        # 使用PCA的转换结果作为系数
        coeffs = pca.transform([z_embedding])[0]  # [K]
        coefficients[cwe_id] = coeffs  # 保存系数

    # 保存系数到文件
    with open(save_path, 'wb') as f:
        pickle.dump(coefficients, f)
    logging.info(f"Zero-Shot 标签的系数已保存到 {save_path}")

    return coefficients, basis_vectors

# 加载保存的系数
def load_coefficients(save_path=COEFFICIENTS_SAVE_PATH):
    if not os.path.exists(save_path):
        raise FileNotFoundError(f"系数文件 {save_path} 不存在。请先运行预处理阶段。")
    with open(save_path, 'rb') as f:
        coefficients = pickle.load(f)
    logging.info(f"Zero-Shot 标签的系数已从 {save_path} 加载。")
    return coefficients

# 基准向量提取和生成新的特征向量（训练后）
def generate_zero_shot_embeddings(model, tokenizer_text, device, train_label_ids, zero_shot_cwe_ids, coefficients, label_descriptions_train, label_descriptions_all):
    # 提取训练标签的基准向量（经过训练的基准向量）
    train_label_embeddings = prepare_label_embeddings(tokenizer_text, model, device, train_label_ids, label_descriptions_train)
    baseline_vectors = torch.stack([torch.tensor(train_label_embeddings[label_id], dtype=torch.float32) for label_id in train_label_ids]).to(device)  # [num_train_labels, embedding_dim]

    # 加载系数
    coefficients_tensor = {}
    for cwe_id, coeffs in coefficients.items():
        coefficients_tensor[cwe_id] = torch.tensor(coeffs, dtype=torch.float32).to(device)  # [K]

    # 生成新的 Zero-Shot 向量
    new_vectors = {}
    for cwe_id, coeffs in coefficients_tensor.items():
        # Multiply coefficients with basis_vectors to get new embedding
        # Assuming basis_vectors are trained and represent K basis vectors
        # Here, model.basis_vectors is [K, hidden_size]
        new_vector = torch.matmul(coeffs, model.basis_vectors)  # [hidden_size]
        new_vector = nn.functional.normalize(new_vector, dim=0)
        new_vectors[cwe_id] = new_vector.cpu()
    logging.info("新的 Zero-Shot 描述特征向量已生成。")
    return new_vectors

# 更新所有标签嵌入并确保其与标签映射一致
def update_all_label_embeddings(model, tokenizer_text, device, all_label_ids, label_descriptions_all, train_label_ids, label_descriptions_train, coefficients):
    # 提取训练标签的基准向量（经过训练的基准向量）
    train_label_embeddings = prepare_label_embeddings(tokenizer_text, model, device, train_label_ids, label_descriptions_train)
    baseline_vectors = torch.stack([torch.tensor(train_label_embeddings[label_id], dtype=torch.float32) for label_id in train_label_ids]).to(device)  # [num_train_labels, embedding_dim]

    # 加载系数
    coefficients_tensor = {}
    for cwe_id, coeffs in coefficients.items():
        coefficients_tensor[cwe_id] = torch.tensor(coeffs, dtype=torch.float32).to(device)  # [K]

    # 生成新的 Zero-Shot 向量
    zero_shot_embeddings = {}
    for cwe_id, coeffs in coefficients_tensor.items():
        # Multiply coefficients with basis_vectors to get new embedding
        new_vector = torch.matmul(coeffs, model.basis_vectors)  # [hidden_size]
        new_vector = nn.functional.normalize(new_vector, dim=0)
        zero_shot_embeddings[cwe_id] = new_vector.cpu()

    # 编码训练标签（再次确保）
    label_embeddings = {label_id: baseline_vectors[i].cpu() for i, label_id in enumerate(train_label_ids)}

    # 合并 Zero-Shot 嵌入
    label_embeddings.update(zero_shot_embeddings)

    # 生成 all_label_embeddings 和 all_label_ids 保持一致
    all_label_ids_sorted = list(label_embeddings.keys())
    all_label_embeddings = torch.stack([label_embeddings[label_id] for label_id in all_label_ids_sorted])

    # 更新映射
    all_index_to_label_id = {idx: label_id for idx, label_id in enumerate(all_label_ids_sorted)}
    all_label_id_to_index = {label_id: idx for idx, label_id in enumerate(all_label_ids_sorted)}

    return all_label_embeddings.to(device), all_label_ids_sorted, all_label_id_to_index, all_index_to_label_id

# 修改后的评估函数
def evaluate(model, dataloader, device, all_label_embeddings, all_label_ids_sorted, all_label_id_to_index, all_index_to_label_id, mode='test'):
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluating on {mode.capitalize()} Set"):
            code_input_ids = batch['code_input_ids'].to(device)
            code_attention_mask = batch['code_attention_mask'].to(device)
            labels = batch['label_id'].cpu().numpy()

            code_embeddings = model(code_input_ids, code_attention_mask)
            logits = torch.matmul(code_embeddings, all_label_embeddings.T)
            predicted_indices = torch.argmax(logits, dim=1).cpu().numpy()

            # 检查预测的索引是否在有效范围内
            predicted_labels = []
            for idx in predicted_indices:
                if idx in all_index_to_label_id:
                    predicted_labels.append(all_index_to_label_id[idx])
                else:
                    logging.warning(f"预测索引 {idx} 无效，不在映射范围内，将其设为 'unknown'")
                    predicted_labels.append('unknown')

            # 获取真实标签
            true_labels = [all_index_to_label_id[idx] for idx in labels]

            # 过滤未知标签
            filtered_pairs = [
                (true, pred) for true, pred in zip(true_labels, predicted_labels)
                if pred in all_label_id_to_index and true in all_label_id_to_index
            ]

            # 拆分为过滤后的 y_true 和 y_pred
            if filtered_pairs:
                filtered_y_true, filtered_y_pred = zip(*filtered_pairs)
                y_true.extend(filtered_y_true)
                y_pred.extend(filtered_y_pred)

    if len(y_true) == 0:
        logging.error("所有样本在评估时均被过滤，检查数据或模型输出。")
        return 0.0

    accuracy = accuracy_score(y_true, y_pred)
    conf_mat = confusion_matrix(y_true, y_pred, labels=all_label_ids_sorted)
    per_class_accuracy = {}
    for idx, label_id in enumerate(all_label_ids_sorted):
        TP = conf_mat[idx, idx]
        support = conf_mat[idx].sum()
        accuracy_i = TP / support if support > 0 else 0.0
        per_class_accuracy[label_id] = accuracy_i

    if mode == 'test':
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=all_label_ids_sorted, zero_division=0
        )
        metrics_df = pd.DataFrame({
            'CWE ID': all_label_ids_sorted,
            'Description': [LABEL_DESCRIPTIONS_ALL[label_id] for label_id in all_label_ids_sorted],
            'Accuracy': [per_class_accuracy[label_id] for label_id in all_label_ids_sorted],
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1
        })

        print(f'\nTest Accuracy: {accuracy:.4f}\n')
        print("Per-class Evaluation Metrics:")
        print(metrics_df.to_string(index=False))

        print("\nClassification Report:")
        report = classification_report(
            y_true, y_pred, zero_division=0,
            target_names=[LABEL_DESCRIPTIONS_ALL[label_id] for label_id in all_label_ids_sorted]
        )
        print(report)

    elif mode == 'validation':
        return accuracy

# 训练函数
def train(model, train_dataloader, val_dataloader, optimizer, device, num_epochs, coefficients, train_label_ids, label_descriptions_train, tokenizer_text):
    best_accuracy = 0.0
    best_model_state = None

    # 准备训练标签的嵌入字典
    label_embeddings_dict = prepare_label_embeddings(tokenizer_text, model, device, train_label_ids, label_descriptions_train)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}/{num_epochs}"):
            code_input_ids = batch['code_input_ids'].to(device)
            code_attention_mask = batch['code_attention_mask'].to(device)
            labels = batch['label_id'].to(device)  # [batch_size]

            # 获取当前批次中所有类别的标签嵌入
            # labels 是类别索引，需要映射回 label_id
            label_ids = [train_index_to_label_id[label_idx.item()] for label_idx in labels]
            text_embeddings = torch.stack([torch.tensor(label_embeddings_dict[label_id], dtype=torch.float32) for label_id in label_ids]).to(device)  # [batch_size, hidden_size]

            code_embeddings = model(code_input_ids, code_attention_mask)  # [batch_size, hidden_size]

            loss = clip_contrastive_loss(code_embeddings, text_embeddings, model.temperature)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_dataloader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

        # 在验证集上评估
        all_label_embeddings, all_label_ids_sorted, all_label_id_to_index, all_index_to_label_id = update_all_label_embeddings(
            model, tokenizer_text, device,
            all_label_ids,
            LABEL_DESCRIPTIONS_ALL,
            train_label_ids,
            label_descriptions_train,
            coefficients
        )

        accuracy = evaluate(model, val_dataloader, device, all_label_embeddings, all_label_ids_sorted, all_label_id_to_index, all_index_to_label_id, mode='validation')
        print(f'Epoch [{epoch+1}/{num_epochs}], Validation Accuracy: {accuracy:.4f}')

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_state = model.state_dict()
            torch.save(best_model_state, MODEL_SAVE_PATH)
            print(f'最佳模型已保存，准确率: {best_accuracy:.4f}')

    if best_model_state:
        model.load_state_dict(best_model_state)
        print(f'最佳模型已加载，准确率: {best_accuracy:.4f}')

# 集成整个流程
# 特征向量分解并保存系数
coefficients, basis_vectors = preprocess_and_save_coefficients(
    model, tokenizer_text, device,
    train_label_ids,
    [cwe['id'] for cwe in zero_shot_cwe],
    LABEL_DESCRIPTIONS_TRAIN,
    LABEL_DESCRIPTIONS_ALL,
    K,
    save_path=COEFFICIENTS_SAVE_PATH
)

# 初始化基准向量为PCA得到的基准向量
with open("pca_model.pkl", 'rb') as f:
    pca = pickle.load(f)
model.basis_vectors.data = torch.tensor(pca.components_, dtype=torch.float32).to(device)  # [K, hidden_size]

# 训练模型
train(model, train_dataloader, val_dataloader, optimizer, device, NUM_EPOCHS, coefficients, train_label_ids, LABEL_DESCRIPTIONS_TRAIN, tokenizer_text)

# 训练后的生成 Zero-Shot 向量并更新标签嵌入
zero_shot_embeddings = generate_zero_shot_embeddings(
    model, tokenizer_text, device,
    train_label_ids,
    [cwe['id'] for cwe in zero_shot_cwe],
    coefficients,
    LABEL_DESCRIPTIONS_TRAIN,
    LABEL_DESCRIPTIONS_ALL
)

all_label_embeddings, all_label_ids_sorted, all_label_id_to_index, all_index_to_label_id = update_all_label_embeddings(
    model, tokenizer_text, device,
    all_label_ids,
    LABEL_DESCRIPTIONS_ALL,
    train_label_ids,
    LABEL_DESCRIPTIONS_TRAIN,
    coefficients
)

# 最终评估
evaluate(model, test_dataloader, device, all_label_embeddings, all_label_ids_sorted, all_label_id_to_index, all_index_to_label_id, mode='test')
