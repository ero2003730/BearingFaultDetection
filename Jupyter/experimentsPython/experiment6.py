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
from tensorflow.keras.optimizers import Adam, Nadam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import backend as K

# Reduzir logs do TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# -----------------------------------------------------------------------------
# 1) Função para seed e reprodutibilidade
# -----------------------------------------------------------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

# -----------------------------------------------------------------------------
# 2) Configuração da GPU (TensorFlow-Metal)
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
# 3) Leitura e Pré-processamento
# -----------------------------------------------------------------------------
def read_and_rename_file(file_path):
    """
    Lê um arquivo XLS no formato TSV,
    pulando 22 linhas iniciais.
    """
    df = pd.read_csv(file_path, skiprows=22, header=None, delimiter='\t')
    df.columns = ['indices', 'velocidade', 'AceleracaoX', 'AceleracaoY', 'AceleracaoZ']
    return df

def is_variable_speed(filename):
    return 1 if "VS" in filename else 0

def determine_damage(filename):
    if filename.startswith("H"):
        return "Saudável"
    elif "0.5X" in filename:
        return "Moderado"
    else:
        return "Severo"

def calculate_variable_speed(index):
    """
    Exemplo de cálculo de velocidade variável
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
    Cria a coluna 'Condição'
    """
    if "VS" in filename:
        df['Condição'] = df['indices'].apply(calculate_variable_speed)
    else:
        condition = filename.split('_')[-1].replace('.xls', '').replace('Hz', '')
        df['Condição'] = int(condition) if condition.isdigit() else None
    return df

def split_train_test(df, train_ratio=0.9):
    """
    Separa 90% para treino e 10% para teste
    """
    train_size = int(len(df) * train_ratio)
    return df.iloc[:train_size], df.iloc[train_size:]

# -----------------------------------------------------------------------------
# 4) Criação de janelas deslizantes (multiclasse)
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
# 5) Funções de Plot (Confusion Matrix e Loss)
# -----------------------------------------------------------------------------
def plot_loss(history, lr_value):
    """
    Plota o gráfico de perda (Train vs Validation)
    """
    plt.figure(figsize=(8,5))
    plt.plot(history['loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title(f"Training and Validation Loss (LR={lr_value})")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

def plot_confusion_mtx(conf_matrix, labels, lr_value):
    """
    Plota a matriz de confusão
    """
    plt.figure(figsize=(6,5))
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=labels)
    disp.plot(cmap="Blues", values_format='d')
    plt.title(f"Confusion Matrix (LR={lr_value})")
    plt.grid(False)
    plt.show()

# -----------------------------------------------------------------------------
# 6) Função Principal
# -----------------------------------------------------------------------------
def main():
    # (A) Seed para reprodutibilidade
    set_seed(42)

    # (B) Configuração GPU (opcional)
    device = configure_gpu()

    # (C) Leitura do DataFrame
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

    # Remover colunas desnecessárias
    for col in ['indices', 'velocidade', 'Dano', 'VelocidadeConstante']:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Substituir NaN
    df = df.fillna(25.0)

    # Separar DF em 4 classes
    df_I = df[df['Arquivo'].str.startswith('I_')].copy()
    df_B = df[df['Arquivo'].str.startswith('B_')].copy()
    df_C = df[df['Arquivo'].str.startswith('C_')].copy()
    df_O = df[df['Arquivo'].str.startswith('O_')].copy()

    # Dividir train e test (90% / 10%)
    df_I_train, df_I_test = split_train_test(df_I)
    df_B_train, df_B_test = split_train_test(df_B)
    df_C_train, df_C_test = split_train_test(df_C)
    df_O_train, df_O_test = split_train_test(df_O)

    # (D) Parâmetros de janela
    window_size = 200
    step_size = 50

    # (E) Criar X_train, y_train, X_test, y_test
    X_train, y_train = process_files_multiclass(
        pd.concat([df_I_train, df_B_train, df_C_train, df_O_train]),
        window_size, step_size
    )
    X_test, y_test = process_files_multiclass(
        pd.concat([df_I_test, df_B_test, df_C_test, df_O_test]),
        window_size, step_size
    )

    # Embaralhar treinamento
    indices_train = np.arange(X_train.shape[0])
    np.random.shuffle(indices_train)
    X_train = X_train[indices_train]
    y_train = y_train[indices_train]

    # Normalizar
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

    # (F) Definir 5 taxas de aprendizado
    learning_rates = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]  

    results = []
    class_labels = ["O_", "C_", "B_", "I_"]  # Ajuste conforme sua one-hot

    # (G) Treinar e avaliar para cada LR
    for lr_value in learning_rates:
        print(f"\nTreinando modelo com Learning Rate={lr_value}...")

        # Limpeza de sessão a cada loop
        K.clear_session()
        gc.collect()

        # Construir modelo
        model = Sequential([
            LSTM(128, input_shape=(window_size, 4), activation='tanh', return_sequences=True),
            LSTM(128, activation='tanh'),
            Dense(4, activation='softmax')
        ])

        # Compilar com Adam e LR variável
        optimizer = Adam(learning_rate=lr_value)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

        # EarlyStopping
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True
        )

        with tf.device(device):
            history = model.fit(
                X_train, y_train,
                epochs=10,
                batch_size=32,
                validation_split=0.1,
                callbacks=[early_stopping],
                verbose=1
            )

        # Prever
        y_pred_probs = model.predict(X_test)
        y_pred = np.argmax(y_pred_probs, axis=1)
        y_test_labels = np.argmax(y_test, axis=1)

        # Métricas
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

        # Armazenar
        results.append({
            "Learning Rate": lr_value,
            "Accuracy": accuracy,
            "Balanced Accuracy": balanced_acc,
            "F1 Macro": f1_macro,
            "F1 Weighted": f1_weighted,
            "Precision Macro": precision_macro,
            "Precision Weighted": precision_weighted,
            "Recall Macro": recall_macro,
            "Recall Weighted": recall_weighted,
            "MSE": mse,
            "MAE": mae,
            "Confusion Matrix": conf_matrix,
            "History": history.history
        })

        # Plot de Loss
        plot_loss(history.history, lr_value)

        # Plot de Matriz de Confusão
        plot_confusion_mtx(conf_matrix, class_labels, lr_value)

        # Limpeza
        del model
        K.clear_session()
        gc.collect()

    # (H) Exibir resultados finais no console
    print("\nRESULTADOS FINAIS - Diferentes Learning Rates:\n")
    for i, res in enumerate(results):
        lr = res["Learning Rate"]
        print(f"LR = {lr}")
        print(f"  -> Accuracy:           {res['Accuracy']:.4f}")
        print(f"  -> Balanced Accuracy:  {res['Balanced Accuracy']:.4f}")
        print(f"  -> F1 Macro:           {res['F1 Macro']:.4f}")
        print(f"  -> F1 Weighted:        {res['F1 Weighted']:.4f}")
        print(f"  -> Precision Macro:    {res['Precision Macro']:.4f}")
        print(f"  -> Precision Weighted: {res['Precision Weighted']:.4f}")
        print(f"  -> Recall Macro:       {res['Recall Macro']:.4f}")
        print(f"  -> Recall Weighted:    {res['Recall Weighted']:.4f}")
        print(f"  -> MSE (prob):         {res['MSE']:.6f}")
        print(f"  -> MAE (prob):         {res['MAE']:.6f}")
        print("  -> Confusion Matrix:\n", res['Confusion Matrix'])
        print("-------------------------------------------------")

    # Limpeza final
    K.clear_session()
    gc.collect()

# -----------------------------------------------------------------------------
# 7) Executa Script
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    main()