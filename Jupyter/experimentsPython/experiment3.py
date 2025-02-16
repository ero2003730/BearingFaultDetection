import os
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import Callback
from tensorflow.keras import backend as K
import gc

# --------------------------------------------------
# 1. Semente para reprodutibilidade
# --------------------------------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

# --------------------------------------------------
# 2. Funções auxiliares de leitura e pré-processamento
# --------------------------------------------------

def read_and_rename_file(file_path):
    """
    Lê um arquivo XLS (no formato TSV) pulando 22 linhas de cabeçalho
    e renomeia as colunas.
    """
    df = pd.read_csv(file_path, skiprows=22, header=None, delimiter='\t')
    df.columns = ['indices', 'velocidade', 'AceleracaoX', 'AceleracaoY', 'AceleracaoZ']
    return df

def determine_damage(filename):
    """
    Determina se o rolamento é 'Saudável', 'Moderado' ou 'Severo',
    baseado no nome do arquivo.
    """
    if filename.startswith("H"):
        return "Saudável"
    elif "0.5X" in filename:
        return "Moderado"
    else:
        return "Severo"

def is_variable_speed(filename):
    """Retorna 1 se 'VS' no nome, senão 0."""
    return 1 if "VS" in filename else 0

def calculate_variable_speed(index):
    """
    Exemplo de função para calcular velocidade de forma variável.
    """
    sampling_frequency = 25600
    time = index
    time_in_cycle = time % 2

    if time_in_cycle <= 1.0:
        speed = time_in_cycle * 40
    else:
        speed = 40 - (time_in_cycle - 1) * 40

    return round(speed * 2) / 2

def extract_condition(df, filename):
    """
    Cria a coluna 'Condição':
    - Se 'VS' no nome do arquivo, aplica 'calculate_variable_speed'.
    - Caso contrário, extrai a frequência do nome do arquivo.
    """
    if "VS" in filename:
        df['Condição'] = df['indices'].apply(calculate_variable_speed)
    else:
        condition = filename.split('_')[-1].replace('.xls', '').replace('Hz', '')
        df['Condição'] = int(condition) if condition.isdigit() else None
    return df

def split_train_test(df, train_ratio=0.9):
    """
    Separa um DataFrame em treino (90%) e teste (10%), de forma simples.
    """
    train_size = int(len(df) * train_ratio)
    df_train = df.iloc[:train_size]
    df_test = df.iloc[train_size:]
    return df_train, df_test

# --------------------------------------------------
# 3. Criação de janelas deslizantes (Multiclasses)
# --------------------------------------------------

def create_sliding_windows_multiclass(df, window_size, step_size, label):
    """
    Cria janelas de tamanho `window_size` com passo `step_size`,
    adaptando para problemas multiclasse (rótulos são arrays).
    """
    X, y = [], []
    for i in range(0, len(df) - window_size + 1, step_size):
        window = df.iloc[i:i + window_size]
        X.append(window[['AceleracaoX', 'AceleracaoY', 'AceleracaoZ', 'Condição']].values)
        y.append(label)  # label já é um array [0,0,0,1], p.ex.
    return np.array(X), np.array(y)

def process_files_multiclass(df, window_size, step_size):
    """
    Percorre todos os arquivos dentro de `df`, detecta prefixo (I_, B_, C_, O_),
    e cria janelas deslizantes com rótulos one-hot correspondentes.
    """
    X_list, y_list = [], []
    for arquivo in df['Arquivo'].unique():
        arquivo_df = df[df['Arquivo'] == arquivo]

        # Definir label com base no prefixo
        if arquivo.startswith("I_"):
            label = [0, 0, 0, 1]  # 0001
        elif arquivo.startswith("B_"):
            label = [0, 0, 1, 0]  # 0010
        elif arquivo.startswith("C_"):
            label = [0, 1, 0, 0]  # 0100
        elif arquivo.startswith("O_"):
            label = [1, 0, 0, 0]  # 1000
        else:
            continue

        X, y = create_sliding_windows_multiclass(arquivo_df, window_size, step_size, label)
        X_list.append(X)
        y_list.append(y)

    return np.vstack(X_list), np.vstack(y_list)

# --------------------------------------------------
# 4. Callback personalizado para progresso do treinamento
# --------------------------------------------------

class TrainingProgressCallback(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        print(f"\nIniciando Época {epoch + 1}/{self.params['epochs']}")

    def on_epoch_end(self, epoch, logs=None):
        print(f"Época {epoch + 1} finalizada. Loss: {logs['loss']:.4f}, "
              f"Val Loss: {logs['val_loss']:.4f}, Val Accuracy: {logs['val_accuracy']:.4f}")

    def on_batch_end(self, batch, logs=None):
        if batch % 50 == 0:
            print(f"  Batch {batch} concluído. Loss: {logs['loss']:.4f}, Accuracy: {logs['accuracy']:.4f}")

# --------------------------------------------------
# 5. Função Principal
# --------------------------------------------------

def main():
    # -------------------------------------
    # (A) Configurar semente para reprodutibilidade
    # -------------------------------------
    set_seed(seed=42)

    # -------------------------------------
    # (B) Configurar GPU (opcional)
    # -------------------------------------
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("Configuração de memória dinâmica para GPU ativada.")
        except RuntimeError as e:
            print(e)

    # -------------------------------------
    # (C) Leitura e pré-processamento dos arquivos
    # -------------------------------------
    data_dir = "/Users/enzooliveira/Pessoal/VS CODE/IC/raw data"
    all_dataframes = []

    for filename in os.listdir(data_dir):
        if filename.endswith(".xls"):
            file_path = os.path.join(data_dir, filename)
            df_temp = read_and_rename_file(file_path)

            df_temp['VelocidadeConstante'] = is_variable_speed(filename)
            df_temp['Dano'] = determine_damage(filename)
            df_temp = extract_condition(df_temp, filename)
            df_temp['Arquivo'] = filename

            all_dataframes.append(df_temp)

    # Concatenar todos os dataframes em um único DataFrame
    df = pd.concat(all_dataframes, ignore_index=True)

    # Remover as colunas 'indices' e 'velocidade'
    df = df.drop(columns=['indices', 'velocidade'])

    # (Opcional) Remover a coluna 'Dano'
    if 'Dano' in df.columns:
        df = df.drop(columns=['Dano'])

    # Substituir valores NaN por 25.0
    df = df.fillna(25.0)

    # Remover VelocidadeConstante (se ainda existir)
    if 'VelocidadeConstante' in df.columns:
        df = df.drop(columns=['VelocidadeConstante'])

    # -------------------------------------
    # (D) Dividir DataFrame por prefixo (I_, B_, C_, O_)
    # -------------------------------------
    df_I = df[df['Arquivo'].str.startswith('I_')].copy()
    df_B = df[df['Arquivo'].str.startswith('B_')].copy()
    df_C = df[df['Arquivo'].str.startswith('C_')].copy()
    df_O = df[df['Arquivo'].str.startswith('O_')].copy()

    # Dividir cada um em treino e teste
    df_I_train, df_I_test = split_train_test(df_I)
    df_B_train, df_B_test = split_train_test(df_B)
    df_C_train, df_C_test = split_train_test(df_C)
    df_O_train, df_O_test = split_train_test(df_O)

    # -------------------------------------
    # (E) Gerar dados de treinamento e teste via janelas
    # -------------------------------------
    window_size = 200
    step_size = 50

    df_train_all = pd.concat([df_I_train, df_B_train, df_C_train, df_O_train])
    df_test_all = pd.concat([df_I_test, df_B_test, df_C_test, df_O_test])

    X_train, y_train = process_files_multiclass(df_train_all, window_size, step_size)
    X_test, y_test = process_files_multiclass(df_test_all, window_size, step_size)

    # Embaralhar os dados de treinamento
    indices_train = np.arange(X_train.shape[0])
    np.random.shuffle(indices_train)
    X_train = X_train[indices_train]
    y_train = y_train[indices_train]

    # Normalizar dados
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

    # -------------------------------------
    # (F) Definir Modelo (LSTM, multiclass)
    # -------------------------------------
    model = Sequential([
        LSTM(128, input_shape=(window_size, 4), activation='tanh', return_sequences=False),
        Dense(4, activation='softmax')  # 4 classes: I_, B_, C_, O_
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # -------------------------------------
    # (G) Treinar Modelo
    # -------------------------------------
    print("\nIniciando treinamento...")
    try:
        with tf.device('/GPU:0'):
            history = model.fit(
                X_train, y_train,
                epochs=10,
                batch_size=32,
                validation_split=0.1,
                callbacks=[TrainingProgressCallback()]
            )
    except RuntimeError as e:
        print(f"Erro ao treinar no GPU: {e}")
        print("Tentando no CPU...")
        with tf.device('/CPU:0'):
            history = model.fit(
                X_train, y_train,
                epochs=10,
                batch_size=32,
                validation_split=0.1,
                callbacks=[TrainingProgressCallback()]
            )

    # -------------------------------------
    # (H) Avaliação do Modelo
    # -------------------------------------
    y_pred_probs = model.predict(X_test)  # Probabilidades previstas
    y_pred = np.argmax(y_pred_probs, axis=1)  # Classes previstas (0..3)
    y_test_labels = np.argmax(y_test, axis=1)  # Classes verdadeiras (0..3)

    # Cálculo de métricas
    accuracy = accuracy_score(y_test_labels, y_pred)
    balanced_acc = balanced_accuracy_score(y_test_labels, y_pred)
    f1_macro = f1_score(y_test_labels, y_pred, average='macro')
    f1_weighted = f1_score(y_test_labels, y_pred, average='weighted')
    precision_macro = precision_score(y_test_labels, y_pred, average='macro')
    precision_weighted = precision_score(y_test_labels, y_pred, average='weighted')
    recall_macro = recall_score(y_test_labels, y_pred, average='macro')
    recall_weighted = recall_score(y_test_labels, y_pred, average='weighted')
    mse = mean_squared_error(y_test, y_pred_probs)
    mae = mean_absolute_error(y_test, y_pred_probs)
    conf_matrix = confusion_matrix(y_test_labels, y_pred)

    # Montar DataFrame de métricas
    metrics_data = {
        "Accuracy": [accuracy],
        "Balanced Accuracy": [balanced_acc],
        "F1 Score (Macro)": [f1_macro],
        "F1 Score (Weighted)": [f1_weighted],
        "Precision (Macro)": [precision_macro],
        "Precision (Weighted)": [precision_weighted],
        "Recall (Macro)": [recall_macro],
        "Recall (Weighted)": [recall_weighted],
        "MSE": [mse],
        "MAE": [mae]
    }
    metrics_df = pd.DataFrame(metrics_data)
    print("\nMétricas de Avaliação:")
    print(metrics_df)

    # **Salvar métricas em CSV** (para o script .sh)
    metrics_df.to_csv("metrics_experiment3.csv", index=False)

    # -------------------------------------
    # (I) Plotar Matriz de Confusão (salvar em PNG)
    # -------------------------------------
    plt.figure(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=["O_", "C_", "B_", "I_"])
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Confusion Matrix")
    plt.grid(False)
    # Salvar a figura
    plt.savefig("confusion_experiment3.png", dpi=100)
    plt.close()

    # -------------------------------------
    # (J) Plotar Curva de Perda (Loss) e salvar
    # -------------------------------------
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    # Salvar a figura
    plt.savefig("loss_experiment3.png", dpi=100)
    plt.close()

    # Liberar memória se desejar
    K.clear_session()
    gc.collect()

# --------------------------------------------------
# 6. Execução do Script
# --------------------------------------------------
if __name__ == "__main__":
    main()