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
    precision_score, recall_score, mean_squared_error, mean_absolute_error
)
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, Callback
from tensorflow.keras import backend as K
from sklearn.metrics import ConfusionMatrixDisplay

# Reduz logs do TensorFlow para evitar poluição
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# -----------------------------------------------------------------------------
# 1) Semente (seed) para reprodutibilidade
# -----------------------------------------------------------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

# -----------------------------------------------------------------------------
# 2) Configuração da GPU (TensorFlow-Metal se macOS)
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
# 3) Exemplo de Callback customizado (Early Stopping manual)
# -----------------------------------------------------------------------------
class CustomEarlyStopping(Callback):
    def __init__(self, patience=5, min_delta=0.001):
        super(CustomEarlyStopping, self).__init__()
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = np.inf
        self.wait = 0

    def on_epoch_end(self, epoch, logs=None):
        current_loss = logs.get("val_loss")
        if current_loss is None:
            return

        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.wait = 0
        else:
            self.wait += 1

        if self.wait >= self.patience:
            print(f"\nParando antecipadamente na época {epoch + 1}. "
                  f"A loss não melhorou significativamente nas últimas {self.patience} épocas.")
            self.model.stop_training = True

# -----------------------------------------------------------------------------
# 4) Leitura / pré-processamento (funções auxiliares)
# -----------------------------------------------------------------------------
def read_and_rename_file(file_path):
    """ Lê um arquivo XLS em formato TSV, pulando 22 linhas. """
    df = pd.read_csv(file_path, skiprows=22, header=None, delimiter='\t')
    df.columns = ['indices', 'velocidade', 'AceleracaoX', 'AceleracaoY', 'AceleracaoZ']
    return df

def determine_damage(filename):
    """ Determina se é Saudável, Moderado ou Severo, caso queira usar. """
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
    Exemplo de função para gerar velocidade variável
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
    """ Cria coluna 'Condição'. """
    if "VS" in filename:
        df['Condição'] = df['indices'].apply(calculate_variable_speed)
    else:
        condition = filename.split('_')[-1].replace('.xls', '').replace('Hz', '')
        df['Condição'] = int(condition) if condition.isdigit() else None
    return df

def split_train_test(df, train_ratio=0.9):
    """ 90% para treino, 10% para teste. """
    train_size = int(len(df) * train_ratio)
    return df.iloc[:train_size], df.iloc[train_size:]

# -----------------------------------------------------------------------------
# 5) Criação de janelas deslizantes (multiclasse)
# -----------------------------------------------------------------------------
def create_sliding_windows_multiclass(df, window_size, step_size, label):
    X, y = [], []
    for i in range(0, len(df) - window_size + 1, step_size):
        window = df.iloc[i:i + window_size]
        X.append(window[['AceleracaoX', 'AceleracaoY', 'AceleracaoZ', 'Condição']].values)
        y.append(label)
    return np.array(X), np.array(y)

def process_files_multiclass(df, window_size, step_size):
    """
    Identifica prefixo (I_, B_, C_, O_)
    e gera janelas com rótulos one-hot (4 classes).
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
# 6) Construção do Modelo LSTM (como no snippet original)
# -----------------------------------------------------------------------------
def build_lstm_model(dropout_rate, learning_rate, window_size=200):
    # Limpar sessão antes de criar
    K.clear_session()
    gc.collect()

    opt = Adam(learning_rate=learning_rate)
    model = Sequential([
        LSTM(128, input_shape=(window_size, 4), activation='tanh', return_sequences=True),
        LSTM(128, activation='tanh'),
        # Se quiser usar dropout, podemos inserir algo como:
        # Dropout(dropout_rate),
        Dense(4, activation='softmax')
    ])
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# -----------------------------------------------------------------------------
# 7) Função para Avaliar o Modelo e Calcular Métricas
# -----------------------------------------------------------------------------
def evaluate_model(model, X_test, y_test):
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_test_labels = np.argmax(y_test, axis=1)
    
    metrics = {
        "Accuracy": accuracy_score(y_test_labels, y_pred),
        "Balanced Accuracy": balanced_accuracy_score(y_test_labels, y_pred),
        "F1 Macro": f1_score(y_test_labels, y_pred, average='macro'),
        "F1 Weighted": f1_score(y_test_labels, y_pred, average='weighted'),
        "Precision Macro": precision_score(y_test_labels, y_pred, average='macro'),
        "Recall Macro": recall_score(y_test_labels, y_pred, average='macro'),
        "MSE": mean_squared_error(y_test, y_pred_probs),
        "MAE": mean_absolute_error(y_test, y_pred_probs),
        "Confusion Matrix": confusion_matrix(y_test_labels, y_pred)
    }
    return metrics, y_pred, y_test_labels

# -----------------------------------------------------------------------------
# 8) Funções de Plot (Loss e Matriz de Confusão)
# -----------------------------------------------------------------------------
def plot_loss(history_dict, title_info=""):
    plt.figure(figsize=(8, 5))
    plt.plot(history_dict['loss'], label='Training Loss')
    plt.plot(history_dict['val_loss'], label='Validation Loss')
    plt.title(f"Training and Validation Loss {title_info}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

def plot_confusion_mtx(conf_matrix, labels, title_info=""):
    plt.figure(figsize=(6,5))
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=labels)
    disp.plot(cmap="Blues", values_format='d')
    plt.title(f"Confusion Matrix {title_info}")
    plt.grid(False)
    plt.show()

# -----------------------------------------------------------------------------
# 9) Função Principal
# -----------------------------------------------------------------------------
def main():
    # (A) Seed
    set_seed(42)

    # (B) Configuração GPU
    device = configure_gpu()

    # (C) Exemplo de leitura do DF
    data_dir = "/Users/enzooliveira/Pessoal/VS CODE/IC/raw data"  # Ajuste
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

    # Remover colunas desnecessárias
    for col in ['indices', 'velocidade', 'Dano', 'VelocidadeConstante']:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Substituir NaN
    df = df.fillna(25.0)

    # Exemplo: separar em I_, B_, C_, O_ + split train/test
    df_I = df[df['Arquivo'].str.startswith('I_')].copy()
    df_B = df[df['Arquivo'].str.startswith('B_')].copy()
    df_C = df[df['Arquivo'].str.startswith('C_')].copy()
    df_O = df[df['Arquivo'].str.startswith('O_')].copy()

    df_I_train, df_I_test = split_train_test(df_I)
    df_B_train, df_B_test = split_train_test(df_B)
    df_C_train, df_C_test = split_train_test(df_C)
    df_O_train, df_O_test = split_train_test(df_O)

    # (D) Configurações de janela
    window_size = 200
    step_size = 50

    # Montar X_train, y_train, X_test, y_test
    X_train, y_train = process_files_multiclass(
        pd.concat([df_I_train, df_B_train, df_C_train, df_O_train]),
        window_size, step_size
    )
    X_test, y_test = process_files_multiclass(
        pd.concat([df_I_test, df_B_test, df_C_test, df_O_test]),
        window_size, step_size
    )

    # Embaralhar
    indices_train = np.arange(X_train.shape[0])
    np.random.shuffle(indices_train)
    X_train = X_train[indices_train]
    y_train = y_train[indices_train]

    # Normalizar
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

    # (E) Definição de hiperparâmetros (só 1 combinação no snippet)
    hyperparameter_combinations = [
        (0.5, 32, 0.001, 'adam')  # (dropout_rate, batch_size, learning_rate, opt_name)
    ]
    patience = 5
    min_delta = 0.001

    # (F) Loop de hiperparâmetros
    results = []
    class_labels = ["O_", "C_", "B_", "I_"]  # Ajuste a ordem caso necessário

    for (dropout_rate, batch_size, learning_rate, opt_name) in hyperparameter_combinations:
        print(f"\nTreinando com Dropout={dropout_rate}, Batch Size={batch_size}, "
              f"Learning Rate={learning_rate}, Optimizer={opt_name}")

        # Construir modelo
        model = build_lstm_model(dropout_rate, learning_rate, window_size=window_size)

        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            min_delta=min_delta
        )
        custom_stopping = CustomEarlyStopping(patience=patience, min_delta=min_delta)

        with tf.device(device):
            history = model.fit(
                X_train,
                y_train,
                epochs=50,
                batch_size=batch_size,
                validation_split=0.1,
                callbacks=[early_stopping, custom_stopping],
                verbose=1
            )

        # Liberar memória
        gc.collect()

        # Avaliar
        metrics, y_pred, y_test_labels = evaluate_model(model, X_test, y_test)

        # Salvar no results
        metrics["Dropout"] = dropout_rate
        metrics["Batch Size"] = batch_size
        metrics["Learning Rate"] = learning_rate
        metrics["Optimizer"] = opt_name
        metrics["History"] = history.history
        results.append(metrics)

        # Plotar Loss
        plot_loss(history.history, f"(Dropout={dropout_rate}, LR={learning_rate})")

        # Plotar Matriz de Confusão
        plot_confusion_mtx(metrics["Confusion Matrix"], class_labels, 
                           f"(Dropout={dropout_rate}, LR={learning_rate})")

        # Remover modelo
        del model
        K.clear_session()
        gc.collect()

    # Imprimir resultados no console
    print("\nRESULTADOS FINAIS:\n")
    for i, res in enumerate(results):
        print(f"Combinação {i+1}: Dropout={res['Dropout']}, Batch Size={res['Batch Size']}, "
              f"LR={res['Learning Rate']}, Opt={res['Optimizer']}")
        print(f"  -> Accuracy:           {res['Accuracy']:.4f}")
        print(f"  -> Balanced Accuracy:  {res['Balanced Accuracy']:.4f}")
        print(f"  -> F1 Macro:           {res['F1 Macro']:.4f}")
        print(f"  -> F1 Weighted:        {res['F1 Weighted']:.4f}")
        print(f"  -> Precision Macro:    {res['Precision Macro']:.4f}")
        print(f"  -> Recall Macro:       {res['Recall Macro']:.4f}")
        print(f"  -> MSE:                {res['MSE']:.6f}")
        print(f"  -> MAE:                {res['MAE']:.6f}")
        print("  -> Confusion Matrix:\n", res["Confusion Matrix"])
        print("-------------------------------------------------")

    # Limpeza final
    K.clear_session()
    gc.collect()


# -----------------------------------------------------------------------------
# 10) Executar Script
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    main()