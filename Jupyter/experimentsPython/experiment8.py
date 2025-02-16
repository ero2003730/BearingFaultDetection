import os
import random
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, 
    precision_score, recall_score, mean_squared_error, mean_absolute_error,
    ConfusionMatrixDisplay
)
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, Callback
from tensorflow.keras import backend as K

# Reduzir logs do TensorFlow para evitar poluição
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# -----------------------------------------------------------------------------
# 1) Função de semente para reprodutibilidade
# -----------------------------------------------------------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

# -----------------------------------------------------------------------------
# 2) Configuração de GPU (TensorFlow-Metal no macOS, etc.)
# -----------------------------------------------------------------------------
def configure_gpu():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("✅ TensorFlow-Metal configurado para GPU")
        except RuntimeError as e:
            print(f"Erro ao configurar GPU: {e}")
    else:
        print("⚠️ Nenhuma GPU disponível! Rodando na CPU.")

    device = "/GPU:0" if gpus else "/CPU:0"
    print(f"Rodando no dispositivo: {device}")
    return device

# -----------------------------------------------------------------------------
# 3) Callback para mostrar progresso do treinamento
# -----------------------------------------------------------------------------
class TrainingProgressCallback(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        print(f"\nIniciando Época {epoch + 1}/{self.params['epochs']}")

    def on_epoch_end(self, epoch, logs=None):
        print(f"Época {epoch + 1} finalizada. Loss: {logs['loss']:.4f}, "
              f"Val Loss: {logs['val_loss']:.4f}, Val Accuracy: {logs['val_accuracy']:.4f}")

# -----------------------------------------------------------------------------
# 4) Leitura e pré-processamento (funções auxiliares)
# -----------------------------------------------------------------------------
def read_and_rename_file(file_path):
    """
    Lê um arquivo XLS em formato TSV (pulando 22 linhas).
    """
    df = pd.read_csv(file_path, skiprows=22, header=None, delimiter='\t')
    df.columns = ['indices', 'velocidade', 'AceleracaoX', 'AceleracaoY', 'AceleracaoZ']
    return df

def determine_damage(filename):
    """Exemplo para identificar dano."""
    if filename.startswith("H"):
        return "Saudável"
    elif "0.5X" in filename:
        return "Moderado"
    else:
        return "Severo"

def is_variable_speed(filename):
    return 1 if "VS" in filename else 0

def calculate_variable_speed(index):
    """
    Exemplo de função para gerar velocidade variável.
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
    """Cria a coluna 'Condição'."""
    if "VS" in filename:
        df['Condição'] = df['indices'].apply(calculate_variable_speed)
    else:
        condition = filename.split('_')[-1].replace('.xls', '').replace('Hz', '')
        df['Condição'] = int(condition) if condition.isdigit() else None
    return df

def split_train_test(df, train_ratio=0.9):
    """
    Separa 90% do DF para treino e 10% para teste
    """
    train_size = int(len(df) * train_ratio)
    return df.iloc[:train_size], df.iloc[train_size:]

# -----------------------------------------------------------------------------
# 5) Criação de janelas deslizantes (Multiclasse)
# -----------------------------------------------------------------------------
def create_sliding_windows_multiclass(df, window_size, step_size, label):
    """
    Cria janelas de tamanho window_size com passo step_size.
    """
    X, y = [], []
    for i in range(0, len(df) - window_size + 1, step_size):
        window = df.iloc[i:i + window_size]
        X.append(window[['AceleracaoX', 'AceleracaoY', 'AceleracaoZ', 'Condição']].values)
        y.append(label)
    return np.array(X), np.array(y)

def process_files_multiclass(df, window_size, step_size):
    """
    Percorre o DF, identificando prefixos I_, B_, C_, O_ e
    cria janelas com labels one-hot.
    """
    X_list, y_list = [], []
    for arquivo in df['Arquivo'].unique():
        arquivo_df = df[df['Arquivo'] == arquivo]
        if arquivo.startswith("I_"):
            label = [0, 0, 0, 1]
        elif arquivo.startswith("B_"):
            label = [0, 0, 1, 0]
        elif arquivo.startswith("C_"):
            label = [0, 1, 0, 0]
        elif arquivo.startswith("O_"):
            label = [1, 0, 0, 0]
        else:
            continue

        X, y = create_sliding_windows_multiclass(arquivo_df, window_size, step_size, label)
        X_list.append(X)
        y_list.append(y)
    return np.vstack(X_list), np.vstack(y_list)

# -----------------------------------------------------------------------------
# 6) Construir modelo LSTM
# -----------------------------------------------------------------------------
def build_lstm_model(window_size, dropout_rate=0.5, learning_rate=0.001):
    """
    Constroi uma LSTM de 2 camadas com 128 neurônios cada,
    e saída de 4 classes (softmax).
    """
    K.clear_session()
    gc.collect()

    opt = Adam(learning_rate=learning_rate)
    model = Sequential([
        LSTM(128, input_shape=(window_size, 4), activation='tanh', return_sequences=True),
        LSTM(128, activation='tanh'),
        Dense(4, activation='softmax')
    ])
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# -----------------------------------------------------------------------------
# 7) Plotar gráfico de perda e matriz de confusão
# -----------------------------------------------------------------------------
def plot_loss(history_dict, window_size, step_size):
    """
    Plota o gráfico de perda (train vs val).
    """
    plt.figure(figsize=(8,5))
    plt.plot(history_dict['loss'], label='Train Loss')
    plt.plot(history_dict['val_loss'], label='Val Loss')
    plt.title(f"Loss (Window={window_size}, Step={step_size})")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

def plot_confusion_mtx(conf_matrix, labels, window_size, step_size):
    """
    Plota a matriz de confusão.
    """
    plt.figure(figsize=(6,5))
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=labels)
    disp.plot(cmap="Blues", values_format='d')
    plt.title(f"Confusion Matrix (Window={window_size}, Step={step_size})")
    plt.grid(False)
    plt.show()

# -----------------------------------------------------------------------------
# 8) Função Principal
# -----------------------------------------------------------------------------
def main():
    # (A) Seed
    set_seed(42)

    # (B) Configurar GPU
    device = configure_gpu()

    # (C) Ler e pré-processar
    data_dir = "/Users/enzooliveira/Pessoal/VS CODE/IC/raw data"  # Ajuste se necessário
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

    df = pd.concat(all_dataframes, ignore_index=True)

    # Remove colunas desnecessárias
    for col in ['indices', 'velocidade', 'Dano', 'VelocidadeConstante']:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Substituir NaNs
    df = df.fillna(25.0)

    # Exemplo: separar df em I_, B_, C_, O_
    df_I = df[df['Arquivo'].str.startswith('I_')].copy()
    df_B = df[df['Arquivo'].str.startswith('B_')].copy()
    df_C = df[df['Arquivo'].str.startswith('C_')].copy()
    df_O = df[df['Arquivo'].str.startswith('O_')].copy()

    # Dividir train/test
    df_I_train, df_I_test = split_train_test(df_I)
    df_B_train, df_B_test = split_train_test(df_B)
    df_C_train, df_C_test = split_train_test(df_C)
    df_O_train, df_O_test = split_train_test(df_O)

    # (D) Definir listas de window_sizes e step_sizes
    window_sizes = [50, 100, 150, 200, 250]
    step_sizes = [50, 100, 150]

    results = []
    class_labels = ["O_", "C_", "B_", "I_"]  # Ajuste se necessário

    # (E) Loop sobre combinações de window_size e step_size
    for w_size in window_sizes:
        for s_size in step_sizes:
            print(f"\n🚀 Treinando com Window Size={w_size}, Step Size={s_size}")

            # Criar X_train, y_train e X_test, y_test
            X_train, y_train = process_files_multiclass(
                pd.concat([df_I_train, df_B_train, df_C_train, df_O_train]),
                w_size, s_size
            )
            X_test, y_test = process_files_multiclass(
                pd.concat([df_I_test, df_B_test, df_C_test, df_O_test]),
                w_size, s_size
            )

            # Embaralhar treino
            indices_train = np.arange(X_train.shape[0])
            np.random.shuffle(indices_train)
            X_train = X_train[indices_train]
            y_train = y_train[indices_train]

            # Normalizar
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
            X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

            # Construir modelo
            model = build_lstm_model(w_size)

            # EarlyStopping
            early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

            # Treinar
            with tf.device(device):
                history = model.fit(
                    X_train, y_train,
                    epochs=10,
                    batch_size=32,
                    validation_split=0.1,
                    callbacks=[early_stopping, TrainingProgressCallback()],
                    verbose=1
                )

            # Avaliar
            y_pred_probs = model.predict(X_test)
            y_pred = np.argmax(y_pred_probs, axis=1)
            y_test_labels = np.argmax(y_test, axis=1)

            # Métricas
            accuracy = accuracy_score(y_test_labels, y_pred)
            balanced_acc = balanced_accuracy_score(y_test_labels, y_pred)
            f1_macro = f1_score(y_test_labels, y_pred, average='macro')
            f1_weighted = f1_score(y_test_labels, y_pred, average='weighted')
            precision_macro = precision_score(y_test_labels, y_pred, average='macro')
            recall_macro = recall_score(y_test_labels, y_pred, average='macro')
            mse = mean_squared_error(y_test, y_pred_probs)
            mae = mean_absolute_error(y_test, y_pred_probs)
            conf_matrix = confusion_matrix(y_test_labels, y_pred)

            # Salvar resultados
            results.append({
                "Window Size": w_size,
                "Step Size": s_size,
                "Accuracy": accuracy,
                "Balanced Accuracy": balanced_acc,
                "F1 Macro": f1_macro,
                "F1 Weighted": f1_weighted,
                "Precision Macro": precision_macro,
                "Recall Macro": recall_macro,
                "MSE": mse,
                "MAE": mae,
                "Confusion Matrix": conf_matrix,
                "History": history.history
            })

            # Plotar Loss
            plot_loss(history.history, w_size, s_size)

            # Plotar Matriz de Confusão
            plot_confusion_mtx(conf_matrix, class_labels, w_size, s_size)

            # Remover modelo
            del model
            K.clear_session()
            gc.collect()

    # Limpeza final
    K.clear_session()
    gc.collect()

    # (F) Exibir os resultados no console
    print("\nRESULTADOS FINAIS:")
    for i, res in enumerate(results):
        print(f"Combinação {i+1}: Window={res['Window Size']} Step={res['Step Size']}")
        print(f"  -> Accuracy:           {res['Accuracy']:.4f}")
        print(f"  -> Balanced Accuracy:  {res['Balanced Accuracy']:.4f}")
        print(f"  -> F1 Macro:           {res['F1 Macro']:.4f}")
        print(f"  -> F1 Weighted:        {res['F1 Weighted']:.4f}")
        print(f"  -> Precision Macro:    {res['Precision Macro']:.4f}")
        print(f"  -> Recall Macro:       {res['Recall Macro']:.4f}")
        print(f"  -> MSE:                {res['MSE']:.6f}")
        print(f"  -> MAE:                {res['MAE']:.6f}")
        print(f"  -> Confusion Matrix:\n{res['Confusion Matrix']}")
        print("-------------------------------------------------")

# -----------------------------------------------------------------------------
# 9) Executar Script
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    main()