
import merlin.io
from merlin.dataloader.torch import Loader

# Просто создаем Dataset из Parquet файлов
dataset = merlin.io.Dataset('/path/to/your/files.parquet', engine="parquet")

# И используем его в даталоадере PyTorch
loader = Loader(dataset, batch_size=1024)
for batch in loader:
    # batch уже на GPU (если доступен)
    pass



import polars as pl
from torch.utils.data import IterableDataset, DataLoader

class ParquetStreamingDataset(IterableDataset):
    def __init__(self, parquet_path, batch_size=64):
        super().__init__()
        self.path = parquet_path
        self.batch_size = batch_size
        # Ленивое сканирование: файл не читается, только смотрятся метаданные
        self.scan = pl.scan_parquet(self.path)

    def __iter__(self):
        # Здесь вы определяете логику того, как читать данные.
        # Например, можно читать по группам строк (row groups)
        # или по заранее вычисленным индексам батчей.
        for batch_df in self._stream_batches():
            # Конвертация DataFrame из Polars в тензоры PyTorch
            images = torch.tensor(batch_df['image'].to_numpy())
            labels = torch.tensor(batch_df['label'].to_numpy())
            yield images, labels

    def _stream_batches(self):
        # Собираем батч из нужного количества строк
        return self.scan.collect().iter_slices(n_rows=self.batch_size)

# Использование со стандартным DataLoader
dataset = ParquetStreamingDataset('/path/to/data.parquet', batch_size=64)
dataloader = DataLoader(dataset, batch_size=None, num_workers=4)



import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, IterableDataset
from petastorm import make_reader
from petastorm.pytorch import DataLoader as PetastormDataLoader  # для сравнения, но мы будем делать свой Dataset
import pyarrow.parquet as pq



# ============================
# 1. Подготовка данных: создаём синтетический датасет в формате Parquet
# ============================
def create_sample_parquet(num_samples=10000, output_path='./sample_data'):
    os.makedirs(output_path, exist_ok=True)
    parquet_path = os.path.join(output_path, 'part.parquet')

    # Генерируем случайные данные: изображения 3x32x32 (плоский вектор) и метки классов
    data = []
    for i in range(num_samples):
        image = np.random.randn(3072).astype(np.float32)  # 3*32*32
        label = np.random.randint(0, 10, dtype=np.int64)
        data.append({'image': image, 'label': label})

    df = pd.DataFrame(data)
    df.to_parquet(parquet_path, engine='pyarrow', compression='snappy')
    print(f"Датасет сохранён в {parquet_path}, {num_samples} образцов")
    return parquet_path


parquet_path = create_sample_parquet(10000)


# ============================
# 2. Определяем собственный IterableDataset, оборачивающий petastorm.Reader
# ============================
class PetastormIterableDataset(IterableDataset):
    """
    Обёртка над petastorm.Reader для использования в torch.utils.data.DataLoader.
    Позволяет читать данные из Parquet потоково, без загрузки в память целиком.
    """

    def __init__(self, parquet_path, shuffle_rows=True, shuffle_row_groups=True):
        super().__init__()
        self.parquet_path = parquet_path
        self.shuffle_rows = shuffle_rows
        self.shuffle_row_groups = shuffle_row_groups

    def __iter__(self):
        # Создаём reader внутри __iter__, чтобы каждый воркер имел своё соединение
        reader = make_reader(
            self.parquet_path,
            shuffle_rows=self.shuffle_rows,
            shuffle_row_groups=self.shuffle_row_groups,
            num_epochs=1  # одна эпоха за проход
        )
        for row in reader:
            # Преобразуем строку в тензоры (можно делать прямо здесь)
            image = torch.tensor(row.image).reshape(3, 32, 32)  # (C, H, W)
            label = torch.tensor(row.label, dtype=torch.long)
            yield image, label
        reader.stop()  # важно закрыть reader после использования

    def __len__(self):
        # Petastorm умеет быстро получать общее количество строк через метаданные Parquet
        # Можно использовать pyarrow.parquet для получения числа строк
        metadata = pq.read_metadata(self.parquet_path)
        return metadata.num_rows


# ============================
# 3. Простая свёрточная сеть
# ============================
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# ============================
# 4. Цикл обучения с использованием нашего Dataset и стандартного DataLoader
# ============================
def train_with_custom_dataset(parquet_path, num_epochs=3, batch_size=64, learning_rate=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleCNN(num_classes=10).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Создаём Dataset
    train_dataset = PetastormIterableDataset(
        parquet_path,
        shuffle_rows=True,
        shuffle_row_groups=True
    )

    # Оборачиваем в стандартный DataLoader PyTorch
    # Важно: num_workers > 0 требует осторожности, но Petastorm поддерживает многопроцессность
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=2,  # можно использовать несколько воркеров
        pin_memory=True if device.type == 'cuda' else False
    )

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        # Итерация по батчам
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            if batch_idx % 10 == 0:
                print(f"Epoch {epoch + 1}/{num_epochs}, Batch {batch_idx}, "
                      f"Loss: {loss.item():.4f}, Acc: {100. * correct / total:.2f}%")

        epoch_loss = total_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        print(f"Epoch {epoch + 1} finished. Avg Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.2f}%")

    print("Обучение завершено")
    return model


# ============================
# 5. Запуск
# ============================
if __name__ == '__main__':
    trained_model = train_with_custom_dataset(parquet_path, num_epochs=3)
    torch.save(trained_model.state_dict(), 'model_petastorm.pth')