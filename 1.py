import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import csv
import gzip
import time
import math
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import os
import random

# 配置中文显示
plt.rcParams["font.family"] = ["SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

# 全局参数配置（调整训练轮数为10轮，优化稳定性）
HIDDEN_SIZE = 128
BATCH_SIZE = 128
N_EPOCHS = 10  # 训练轮数改为10轮
USE_GPU = torch.cuda.is_available()
MAX_VOCAB_SIZE = 10000
EMBEDDING_DIM = 100
N_GRAM = 2
SENTIMENT_CLASSES = 5
LEARNING_RATE = 5e-5  # 降低学习率，避免梯度爆炸
GRAD_CLIP = 0.5  # 增强梯度裁剪
MAX_REVIEW_LENGTH = 300  # 进一步限制评论长度
VAL_SPLIT = 0.1
SEED = 42

# 固定随机种子
random.seed(SEED)
torch.manual_seed(SEED)
if USE_GPU:
    torch.cuda.manual_seed(SEED)

# 选择计算设备
device = torch.device("cuda:0" if USE_GPU else "cpu")
print(f"正在使用的计算设备: {device}")


# ===================== 数据集类定义 =====================
class MovieReviewDataset(Dataset):
    def __init__(self, reviews, sentiments, vocab=None, ngram_vocab=None, is_train=True):
        self.reviews = reviews
        self.sentiments = sentiments
        self.is_train = is_train
        self.len = len(self.reviews)

        if is_train:
            self.vocab, self.ngram_vocab = self.build_vocab()
            self.vocab_size = len(self.vocab)
            self.ngram_vocab_size = len(self.ngram_vocab)
            self.total_vocab_size = self.vocab_size + self.ngram_vocab_size
            print(
                f"词汇表构建完成: 单字 {self.vocab_size} 个, {N_GRAM}-gram {self.ngram_vocab_size} 个, 总计 {self.total_vocab_size} 个")
        else:
            if vocab is None or ngram_vocab is None:
                raise ValueError("验证集/测试集必须传入训练集的词汇表")
            self.vocab = vocab
            self.ngram_vocab = ngram_vocab
            self.total_vocab_size = len(vocab) + len(ngram_vocab)

    @staticmethod
    def _truncate_review(review, max_len):
        words = review.split()
        return ' '.join(words[:max_len]) if len(words) > max_len else review

    @staticmethod
    def _is_gzipped(filepath):
        if not os.path.exists(filepath):
            return False
        try:
            with open(filepath, 'rb') as f:
                return f.read(2) == b'\x1f\x8b'
        except Exception:
            return False

    def build_vocab(self):
        word_counter = Counter()
        for review in self.reviews:
            word_counter.update(review.lower().split())
        common_words = word_counter.most_common(MAX_VOCAB_SIZE - 2)
        vocab = {'<PAD>': 0, '<UNK>': 1}
        for idx, (word, _) in enumerate(common_words, 2):
            vocab[word] = idx

        ngram_counter = Counter()
        for review in self.reviews:
            words = review.lower().split()
            ngrams = self.get_ngrams(words)
            ngram_counter.update(ngrams)
        ngram_vocab = {ngram: idx for idx, (ngram, _) in enumerate(ngram_counter.most_common(), len(vocab))}
        return vocab, ngram_vocab

    def get_ngrams(self, words):
        return [' '.join(words[i:i + N_GRAM]) for i in range(len(words) - N_GRAM + 1)]

    def __getitem__(self, idx):
        return self.reviews[idx], self.sentiments[idx]

    def __len__(self):
        return self.len


# ===================== 数据加载与划分 =====================
def load_data(train_path='train.tsv', test_path='test.tsv'):
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"训练集文件不存在: {train_path}")

    if MovieReviewDataset._is_gzipped(train_path):
        train_file = gzip.open(train_path, 'rt', encoding='utf-8')
    else:
        train_file = open(train_path, 'r', encoding='utf-8')

    with train_file:
        reader = csv.reader(train_file, delimiter='\t')
        header = next(reader)
        print(f"训练集表头: {header}")
        rows = [row for row in reader if len(row) == len(header)]

    print(f"训练集有效数据: {len(rows)} 条")
    TEXT_COL = 2
    LABEL_COL = 3

    all_reviews = []
    all_labels = []
    for row in rows:
        review = MovieReviewDataset._truncate_review(row[TEXT_COL], MAX_REVIEW_LENGTH)
        label = int(row[LABEL_COL])
        all_reviews.append(review)
        all_labels.append(label)

    dataset_size = len(all_reviews)
    val_size = int(VAL_SPLIT * dataset_size)
    train_size = dataset_size - val_size
    indices = list(range(dataset_size))
    random.shuffle(indices)

    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_reviews = [all_reviews[i] for i in train_indices]
    train_labels = [all_labels[i] for i in train_indices]
    val_reviews = [all_reviews[i] for i in val_indices]
    val_labels = [all_labels[i] for i in val_indices]

    print(f"训练集: {len(train_reviews)} 条, 验证集: {len(val_reviews)} 条")

    trainset = MovieReviewDataset(train_reviews, train_labels, is_train=True)
    valset = MovieReviewDataset(val_reviews, val_labels,
                                vocab=trainset.vocab,
                                ngram_vocab=trainset.ngram_vocab,
                                is_train=False)

    testset = None
    if os.path.exists(test_path):
        if MovieReviewDataset._is_gzipped(test_path):
            test_file = gzip.open(test_path, 'rt', encoding='utf-8')
        else:
            test_file = open(test_path, 'r', encoding='utf-8')

        with test_file:
            reader = csv.reader(test_file, delimiter='\t')
            test_header = next(reader)
            print(f"测试集表头: {test_header}")
            test_rows = [row for row in reader if len(row) == len(test_header)]

        test_reviews = [MovieReviewDataset._truncate_review(row[TEXT_COL], MAX_REVIEW_LENGTH) for row in test_rows]
        test_labels = [0] * len(test_reviews)
        testset = MovieReviewDataset(test_reviews, test_labels,
                                     vocab=trainset.vocab,
                                     ngram_vocab=trainset.ngram_vocab,
                                     is_train=False)
        print(f"测试集: {len(testset)} 条")
    else:
        print(f"未找到测试集文件: {test_path}")

    return trainset, valset, testset


# ===================== 模型定义 =====================
class FastTextClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.fc1 = nn.Linear(embedding_dim, hidden_size)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x, seq_lengths):
        embed = self.embedding(x)
        pooled = []
        for i in range(embed.size(0)):
            valid_embed = embed[i, :seq_lengths[i]]
            pooled.append(torch.mean(valid_embed, dim=0))
        pooled = torch.stack(pooled)
        hidden = F.relu(self.fc1(pooled))
        hidden = self.dropout(hidden)
        output = self.fc2(hidden)
        return output


# ===================== 工具函数 =====================
def create_tensor(tensor):
    return tensor.to(device) if USE_GPU else tensor


def text_to_ids(review, vocab, ngram_vocab):
    words = review.lower().split()[:MAX_REVIEW_LENGTH]  # 限制单词数
    word_ids = [vocab.get(word, vocab['<UNK>']) for word in words]
    ngrams = [' '.join(words[i:i + N_GRAM]) for i in range(len(words) - N_GRAM + 1)]
    ngram_ids = [ngram_vocab.get(ngram, 0) for ngram in ngrams]
    # 过滤空序列
    return word_ids + ngram_ids if (word_ids + ngram_ids) else [vocab['<UNK>']]


def make_batch(reviews, labels, vocab, ngram_vocab):
    sequences = []
    lengths = []
    valid_labels = []  # 保存有效样本的标签

    for review, label in zip(reviews, labels):
        ids = text_to_ids(review, vocab, ngram_vocab)
        if len(ids) == 0:
            continue  # 跳过空序列
        sequences.append(ids)
        lengths.append(len(ids))
        valid_labels.append(label)

    if not sequences:  # 避免空批次
        return torch.tensor([]).long().to(device), torch.tensor([]).long().to(device), torch.tensor([]).long().to(
            device)

    max_len = max(lengths)
    seq_tensor = torch.zeros(len(sequences), max_len, dtype=torch.long)
    for i, seq in enumerate(sequences):
        seq_tensor[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)

    lengths_tensor = torch.tensor(lengths, dtype=torch.long)
    labels_tensor = torch.tensor(valid_labels, dtype=torch.long)

    lengths_tensor, perm_idx = lengths_tensor.sort(descending=True)
    seq_tensor = seq_tensor[perm_idx]
    labels_tensor = labels_tensor[perm_idx]

    return create_tensor(seq_tensor), create_tensor(lengths_tensor), create_tensor(labels_tensor)


def time_since(since):
    s = time.time() - since
    return f"{int(s // 60)}m {int(s % 60)}s"


# ===================== 训练与评估函数 =====================
def train_epoch(model, train_loader, criterion, optimizer, epoch):
    model.train()
    total_loss = 0.0
    total_samples = 0
    start_time = time.time()

    for batch_idx, (reviews, labels) in enumerate(train_loader, 1):
        inputs, lengths, targets = make_batch(
            reviews, labels,
            train_loader.dataset.vocab,
            train_loader.dataset.ngram_vocab
        )

        # 跳过空批次
        if inputs.size(0) == 0:
            continue

        optimizer.zero_grad()
        outputs = model(inputs, lengths)
        loss = criterion(outputs, targets)

        # 检测并跳过NaN损失
        if torch.isnan(loss):
            print(f"警告：Epoch {epoch}, Batch {batch_idx} 出现NaN损失，已跳过")
            continue

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)  # 梯度裁剪
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        total_samples += inputs.size(0)

        if batch_idx % 20 == 0:
            avg_loss = total_loss / total_samples
            print(f"[{time_since(start_time)}] Epoch {epoch} "
                  f"[{total_samples}/{len(train_loader.dataset)}] "
                  f"损失: {avg_loss:.4f}")

    return total_loss / total_samples if total_samples > 0 else 0


def evaluate(model, data_loader, is_test=False):
    model.eval()
    correct = 0
    total = 0
    set_name = "测试集" if is_test else "验证集"

    with torch.no_grad():
        for reviews, labels in data_loader:
            inputs, lengths, targets = make_batch(
                reviews, labels,
                data_loader.dataset.vocab,
                data_loader.dataset.ngram_vocab
            )

            if inputs.size(0) == 0:
                continue

            outputs = model(inputs, lengths)
            _, preds = torch.max(outputs, 1)

            total += targets.size(0)
            if not is_test:
                correct += (preds == targets).sum().item()

    if not is_test and total > 0:
        acc = 100 * correct / total
        print(f"{set_name} 准确率: {correct}/{total} ({acc:.2f}%)")
        return acc
    else:
        print(f"{set_name} 预测完成，共 {total} 条数据")
        return 0


# ===================== 主函数 =====================
def main():
    # 加载数据
    try:
        print("开始加载数据...")
        trainset, valset, testset = load_data()
        train_loader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(valset, batch_size=BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False) if testset else None
        print("数据加载完成！")
    except Exception as e:
        print(f"数据加载失败: {e}")
        return

    # 初始化模型、损失函数、优化器
    model = FastTextClassifier(
        vocab_size=trainset.total_vocab_size,
        embedding_dim=EMBEDDING_DIM,
        hidden_size=HIDDEN_SIZE,
        num_classes=SENTIMENT_CLASSES
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    print(f"模型初始化完成: {model}")

    # 训练过程（10轮）
    print(f"\n开始训练（共 {N_EPOCHS} 轮）...")
    start = time.time()
    train_losses = []
    val_accs = []

    for epoch in range(1, N_EPOCHS + 1):
        print(f"\n----- Epoch {epoch}/{N_EPOCHS} -----")
        # 训练
        train_loss = train_epoch(model, train_loader, criterion, optimizer, epoch)
        train_losses.append(train_loss)
        # 验证
        val_acc = evaluate(model, val_loader, is_test=False)
        val_accs.append(val_acc)

    # 测试集预测（如果有）
    if test_loader:
        evaluate(model, test_loader, is_test=True)

    # 绘制训练曲线
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='训练损失')
    plt.xlabel('Epoch')
    plt.ylabel('损失')
    plt.title('训练损失变化（10轮）')
    plt.legend()
    plt.grid(linestyle='--', alpha=0.6)

    plt.subplot(1, 2, 2)
    plt.plot(val_accs, label='验证集准确率')
    plt.xlabel('Epoch')
    plt.ylabel('准确率 (%)')
    plt.title('验证集准确率变化（10轮）')
    plt.legend()
    plt.grid(linestyle='--', alpha=0.6)
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.show()

    print(f"\n训练完成！总耗时: {time_since(start)}")
    print(f"最终训练损失: {train_losses[-1]:.4f}")
    print(f"最终验证集准确率: {val_accs[-1]:.2f}%")


if __name__ == '__main__':
    main()