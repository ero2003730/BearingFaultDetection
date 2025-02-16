import os
import random
import gc
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
    precision_score, recall_score, mean_squared_error, mean_absolute_error,
    ConfusionMatrixDisplay
)
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras import backend as K

# Reduzir logs do TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# -----------------------------------------------------------------------------
# 1) Definir semente para reprodutibilidade
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
            print("✅ TensorFlow-Metal configurado para uso da GPU")
        except RuntimeError as e:
            print(f"Erro ao configurar GPU: {e}")
    else:
        print("⚠️ Nenhuma GPU disponível! O código será executado na CPU.")

    device = "/GPU:0" if gpus else "/CPU:0"
    print(f"Rodando no dispositivo: {device}")
    return device

# -----------------------------------------------------------------------------
# 3) Funções de leitura / pré-processamento
# -----------------------------------------------------------------------------

def read_and_rename_file(file_path):
    """Lê um arquivo XLS (TSV) e renomeia as colunas."""
    df = pd.read_csv(file_path, skiprows=22, header=None, delimiter='\t')
    df.columns = ['indices', 'velocidade', 'AceleracaoX', 'AceleracaoY', 'AceleracaoZ']
    return df

def is_variable_speed(filename):
    return 1 if "VS" in filename else 0

def determine_damage(filename):
    """Exemplo de função para classificar dano (não obrigatória aqui)."""
    if filename.startswith("H"):
        return "Saudável"
    elif "0.5X" in filename:
        return "Moderado"
    else:
        return "Severo"

def calculate_variable_speed(index):
    """
    Função de exemplo para calcular velocidade variável,
    caso seja 'VS' no arquivo.
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
    """Cria a coluna 'Condição' para cada arquivo."""
    if "VS" in filename:
        df['Condição'] = df['indices'].apply(calculate_variable_speed)
    else:
        condition = filename.split('_')[-1].replace('.xls', '').replace('Hz', '')
        df['Condição'] = int(condition) if condition.isdigit() else None
    return df

def create_sliding_windows_multiclass(df, window_size, step_size, label):
    """
    Cria janelas de tamanho `window_size` com passo `step_size`,
    associando cada janela ao label one-hot (ex: [0,0,0,1]).
    """
    X, y = [], []
    for i in range(0, len(df) - window_size + 1, step_size):
        window = df.iloc[i:i + window_size]
        X.append(window[['AceleracaoX', 'AceleracaoY', 'AceleracaoZ', 'Condição']].values)
        y.append(label)
    return np.array(X), np.array(y)

def process_files_multiclass(df, window_size, step_size):
    """
    Para cada arquivo no DF, identifica prefixo (I_, B_, C_, O_)
    e gera janelas deslizantes com labels one-hot.
    """
    X_list, y_list = [], []
    for arquivo in df['Arquivo'].unique():
        arquivo_df = df[df['Arquivo'] == arquivo]
        # Define label com base no prefixo
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
# 4) Classe Callback para exibir progresso no treinamento
# -----------------------------------------------------------------------------
class TrainingProgressCallback(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        print(f"\nIniciando Época {epoch + 1}/{self.params['epochs']}")

    def on_epoch_end(self, epoch, logs=None):
        print(f"Época {epoch + 1} finalizada. Loss: {logs['loss']:.4f}, "
              f"Val Loss: {logs['val_loss']:.4f}, Val Accuracy: {logs['val_accuracy']:.4f}")

# -----------------------------------------------------------------------------
# 5) Função Principal (main)
# -----------------------------------------------------------------------------
def main():
    # --------------------------------------------
    # (A) Semente para reprodutibilidade
    # --------------------------------------------
    set_seed(42)

    # --------------------------------------------
    # (B) Configurar GPU (TensorFlow-Metal, etc.)
    # --------------------------------------------
    device = configure_gpu()

    # --------------------------------------------
    # (C) Leitura / montagem do DataFrame
    # --------------------------------------------
    data_dir = "/Users/enzooliveira/Pessoal/VS CODE/IC/raw data"  # Ajuste se necessário
    all_dataframes = []

    for filename in os.listdir(data_dir):
        if filename.endswith(".xls"):
            file_path = os.path.join(data_dir, filename)
            df_temp = read_and_rename_file(file_path)
            # Exemplo de colunas extras
            df_temp['VelocidadeConstante'] = is_variable_speed(filename)
            df_temp['Dano'] = determine_damage(filename)
            df_temp = extract_condition(df_temp, filename)
            df_temp['Arquivo'] = filename

            all_dataframes.append(df_temp)

    df = pd.concat(all_dataframes, ignore_index=True)

    # Remover colunas desnecessárias
    for col in ['indices', 'velocidade', 'Dano', 'VelocidadeConstante']:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Substituir NaNs
    df = df.fillna(25.0)

    # --------------------------------------------
    # (D) Separar DF em 4 classes
    #     (Exemplo: df_I_train, df_B_train, etc.)
    #     Ajuste conforme sua divisão de dados
    # --------------------------------------------
    # Vamos supor que já existam df_I_train, df_B_train, etc. 
    # ou que você crie aqui a divisão manualmente.
    #
    # Exemplo fictício de split (apenas se já tiver as DF separadas):
    # df_I_train, df_B_train, df_C_train, df_O_train
    # df_I_test, df_B_test, df_C_test, df_O_test
    #
    # Se não tiver, faça algo como:
    df_I = df[df['Arquivo'].str.startswith('I_')].copy()
    df_B = df[df['Arquivo'].str.startswith('B_')].copy()
    df_C = df[df['Arquivo'].str.startswith('C_')].copy()
    df_O = df[df['Arquivo'].str.startswith('O_')].copy()

    # Dividir manualmente (exemplo: 90%/10%)
    def split_train_test(df_, train_ratio=0.9):
        split_idx = int(len(df_) * train_ratio)
        return df_.iloc[:split_idx], df_.iloc[split_idx:]

    df_I_train, df_I_test = split_train_test(df_I)
    df_B_train, df_B_test = split_train_test(df_B)
    df_C_train, df_C_test = split_train_test(df_C)
    df_O_train, df_O_test = split_train_test(df_O)

    # --------------------------------------------
    # (E) Parametros de janela
    # --------------------------------------------
    window_size = 200
    step_size = 50

    # --------------------------------------------
    # (F) Criação X_train, y_train, X_test, y_test
    # --------------------------------------------
    X_train, y_train = process_files_multiclass(
        pd.concat([df_I_train, df_B_train, df_C_train, df_O_train]),
        window_size, step_size
    )
    X_test, y_test = process_files_multiclass(
        pd.concat([df_I_test, df_B_test, df_C_test, df_O_test]),
        window_size, step_size
    )

    # Embaralhar dados de treino
    indices_train = np.arange(X_train.shape[0])
    np.random.shuffle(indices_train)
    X_train = X_train[indices_train]
    y_train = y_train[indices_train]

    # Normalizar
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

    # --------------------------------------------
    # (G) Construção e Treinamento do modelo
    # --------------------------------------------
    with tf.device(device):
        model = Sequential()
        model.add(LSTM(128, input_shape=(window_size, 4), activation='tanh', return_sequences=True))
        model.add(LSTM(128, activation='tanh'))
        model.add(Dense(4, activation='softmax'))  # 4 classes

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        print("\nTreinando o modelo com 35 épocas...")
        history = model.fit(
            X_train, y_train,
            epochs=35,
            batch_size=32,
            validation_split=0.1,
            callbacks=[TrainingProgressCallback()],
            verbose=1
        )

        # Liberar memória após treino
        gc.collect()

        # Avaliar
        y_pred_probs = model.predict(X_test)
        y_pred = np.argmax(y_pred_probs, axis=1)
        y_test_labels = np.argmax(y_test, axis=1)

        accuracy = accuracy_score(y_test_labels, y_pred)
        balanced_accuracy = balanced_accuracy_score(y_test_labels, y_pred)
        f1_macro = f1_score(y_test_labels, y_pred, average='macro')
        f1_weighted = f1_score(y_test_labels, y_pred, average='weighted')
        precision_macro = precision_score(y_test_labels, y_pred, average='macro')
        precision_weighted = precision_score(y_test_labels, y_pred, average='weighted')
        recall_macro = recall_score(y_test_labels, y_pred, average='macro')
        recall_weighted = recall_score(y_test_labels, y_pred, average='weighted')
        mse = mean_squared_error(y_test, y_pred_probs)
        mae = mean_absolute_error(y_test, y_pred_probs)
        conf_matrix = confusion_matrix(y_test_labels, y_pred)

    # --------------------------------------------
    # (H) Exibir métricas
    # --------------------------------------------
    metrics_data = {
        "Accuracy": [accuracy],
        "Balanced Accuracy": [balanced_accuracy],
        "F1 Macro": [f1_macro],
        "F1 Weighted": [f1_weighted],
        "Precision Macro": [precision_macro],
        "Precision Weighted": [precision_weighted],
        "Recall Macro": [recall_macro],
        "Recall Weighted": [recall_weighted],
        "MSE": [mse],
        "MAE": [mae]
    }
    metrics_df = pd.DataFrame(metrics_data)
    print("\nResultados das Métricas:")
    print(metrics_df)

    # --------------------------------------------
    # (I) Matriz de Confusão
    # --------------------------------------------
    # Ajuste a ordem das labels se necessário,
    # dependendo da forma como sua one-hot encoding está.
    plt.figure(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=conf_matrix,
        display_labels=['I_', 'B_', 'C_', 'O_']  # Ajuste a ordem das classes
    )
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Confusion Matrix")
    plt.grid(False)
    plt.show()

    # --------------------------------------------
    # (J) Plotar Curva de Perda (Loss)
    # --------------------------------------------
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Liberar memória final
    K.clear_session()
    gc.collect()

# -----------------------------------------------------------------------------
# 6) Execução do Script
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    main()