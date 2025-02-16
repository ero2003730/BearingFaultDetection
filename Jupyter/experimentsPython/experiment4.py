import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, Callback
from tensorflow.keras import backend as K

from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, 
    precision_score, recall_score, mean_squared_error, mean_absolute_error
)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import ConfusionMatrixDisplay

import gc

# -----------------------------------------------------------------------------
# 1) Definir semente para reprodutibilidade
# -----------------------------------------------------------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

# -----------------------------------------------------------------------------
# 2) Leitura e Pré-processamento
# -----------------------------------------------------------------------------
def read_and_rename_file(file_path):
    """
    Lê o arquivo XLS no formato TSV, pulando 22 linhas iniciais.
    """
    df = pd.read_csv(file_path, skiprows=22, header=None, delimiter='\t')
    df.columns = ['indices', 'velocidade', 'AceleracaoX', 'AceleracaoY', 'AceleracaoZ']
    return df

def determine_damage(filename):
    """Exemplo para classificar dano (não necessariamente usado aqui)."""
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
    time_in_cycle = index % 2
    if time_in_cycle <= 1.0:
        speed = time_in_cycle * 40
    else:
        speed = 40 - (time_in_cycle - 1) * 40
    # Apenas arredondando em 0.5
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
    """Divide o DataFrame em 90% para treino e 10% para teste."""
    train_size = int(len(df) * train_ratio)
    df_train = df.iloc[:train_size]
    df_test = df.iloc[train_size:]
    return df_train, df_test

# -----------------------------------------------------------------------------
# 3) Criação de janelas deslizantes para multiclasse
# -----------------------------------------------------------------------------
def create_sliding_windows_multiclass(df, window_size, step_size, label):
    """
    Cria janelas de tamanho `window_size` com passo `step_size`.
    'label' é um array one-hot com 4 posições (por exemplo [0,0,1,0]).
    """
    X, y = [], []
    for i in range(0, len(df) - window_size + 1, step_size):
        window = df.iloc[i:i + window_size]
        X.append(window[['AceleracaoX', 'AceleracaoY', 'AceleracaoZ', 'Condição']].values)
        y.append(label)
    return np.array(X), np.array(y)

def process_files_multiclass(df, window_size, step_size):
    """
    Percorre o DF, identificando prefixo do arquivo (I_, B_, C_, O_).
    Gera janelas deslizantes e rótulos one-hot.
    """
    X_list, y_list = [], []
    for arquivo in df['Arquivo'].unique():
        arquivo_df = df[df['Arquivo'] == arquivo]

        # Ajuste de one-hot (ordem: O_, C_, B_, I_) => 0,1,2,3
        # Mas o script original associava:
        #   I_ => [0,0,0,1]
        #   B_ => [0,0,1,0]
        #   C_ => [0,1,0,0]
        #   O_ => [1,0,0,0]
        # Isso é coerente com as labels: 0=O_, 1=C_, 2=B_, 3=I_
        if arquivo.startswith("I_"):
            label = [0, 0, 0, 1]
        elif arquivo.startswith("B_"):
            label = [0, 0, 1, 0]
        elif arquivo.startswith("C_"):
            label = [0, 1, 0, 0]
        elif arquivo.startswith("O_"):
            label = [1, 0, 0, 0]
        else:
            # Se não encaixar em nenhum prefixo esperado, ignora
            continue

        X, y = create_sliding_windows_multiclass(arquivo_df, window_size, step_size, label)
        X_list.append(X)
        y_list.append(y)

    return np.vstack(X_list), np.vstack(y_list)

# -----------------------------------------------------------------------------
# 4) Callback para progresso do treinamento (opcional)
# -----------------------------------------------------------------------------
class TrainingProgressCallback(Callback):
    def on_epoch_begin(self, epoch, logs=None):
        print(f"\nIniciando Época {epoch + 1}/{self.params['epochs']}")

    def on_epoch_end(self, epoch, logs=None):
        print(f"Época {epoch + 1} finalizada. Loss: {logs['loss']:.4f}, "
              f"Val Loss: {logs['val_loss']:.4f}, Val Accuracy: {logs['val_accuracy']:.4f}")

# -----------------------------------------------------------------------------
# 5) Funções de Plot (Loss e Matriz de Confusão) - adaptadas para salvar .png
# -----------------------------------------------------------------------------
def plot_loss(history_dict, model_name=""):
    """
    Plota e salva o gráfico de perda (training vs validation).
    """
    plt.figure(figsize=(8,5))
    plt.plot(history_dict['loss'], label='Train Loss')
    plt.plot(history_dict['val_loss'], label='Val Loss')
    plt.title(f"Training and Validation Loss - {model_name}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    # Salvar a figura
    safe_name = model_name.replace(" ", "_").replace("=", "").replace(",", "")
    plt.savefig(f"loss_{safe_name}.png", dpi=100)
    plt.close()

def plot_confusion_matrix(conf_matrix, labels, model_name=""):
    """
    Plota e salva a matriz de confusão usando ConfusionMatrixDisplay.
    """
    plt.figure(figsize=(6,5))
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=labels)
    disp.plot(cmap="Blues", values_format='d')
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.grid(False)
    safe_name = model_name.replace(" ", "_").replace("=", "").replace(",", "")
    plt.savefig(f"confusion_{safe_name}.png", dpi=100)
    plt.close()

# -----------------------------------------------------------------------------
# 6) Criação do modelo LSTM de forma dinâmica
# -----------------------------------------------------------------------------
def create_lstm_model(config, input_shape):
    model = Sequential()
    
    for i, units in enumerate(config["layers"]):
        return_sequences = (i < len(config["layers"]) - 1)

        # Verifica se é bidirecional
        if config.get("bidirectional", False):
            model.add(
                Bidirectional(
                    LSTM(units, return_sequences=return_sequences, activation='tanh'),
                    input_shape=input_shape
                )
            )
        else:
            if i == 0:
                model.add(LSTM(units, return_sequences=return_sequences, activation='tanh', 
                               input_shape=input_shape))
            else:
                model.add(LSTM(units, return_sequences=return_sequences, activation='tanh'))

        # Se quiser batch normalization
        if config.get("batch_norm", False):
            model.add(BatchNormalization())

        # Dropout após cada camada LSTM
        if config["dropout"]:
            model.add(Dropout(0.3))

    # Camada densa final (4 classes)
    model.add(Dense(4, activation='softmax'))
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# -----------------------------------------------------------------------------
# 7) Função Principal
# -----------------------------------------------------------------------------
def main():
    # Seed para reprodutibilidade
    set_seed(42)

    # Configuração de GPU (opcional)
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("Configuração de memória dinâmica para GPU ativada.")
        except RuntimeError as e:
            print(e)

    if tf.config.list_physical_devices('GPU'):
        device = "/GPU:0"
        print("Treinando com GPU disponível.")
    else:
        device = "/CPU:0"
        print("Treinando apenas com CPU.")

    # -------------------------------------------------------------------------
    # (A) Leitura dos arquivos e pré-processamento
    # -------------------------------------------------------------------------
    data_dir = "/Users/enzooliveira/Pessoal/VS CODE/IC/raw data"  # Ajuste se necessário
    all_dataframes = []

    for filename in os.listdir(data_dir):
        if filename.endswith(".xls"):
            file_path = os.path.join(data_dir, filename)
            df_temp = read_and_rename_file(file_path)

            # Adicionamos colunas auxiliares
            df_temp['VelocidadeConstante'] = is_variable_speed(filename)
            df_temp['Dano'] = determine_damage(filename)
            df_temp = extract_condition(df_temp, filename)
            df_temp['Arquivo'] = filename

            all_dataframes.append(df_temp)

    # Concatenar todos os dataframes
    df = pd.concat(all_dataframes, ignore_index=True)

    # Remover colunas desnecessárias
    df.drop(columns=['indices', 'velocidade'], inplace=True, errors='ignore')
    # (Opcional) remover a coluna 'Dano'
    if 'Dano' in df.columns:
        df.drop(columns=['Dano'], inplace=True)
    # Remover a coluna 'VelocidadeConstante' (opcional, se não for usar)
    if 'VelocidadeConstante' in df.columns:
        df.drop(columns=['VelocidadeConstante'], inplace=True)

    # Substituir valores NaN por 25.0 (como no primeiro script)
    df = df.fillna(25.0)

    # -------------------------------------------------------------------------
    # (B) Dividir DataFrame por prefixos e criar splits de treino e teste
    # -------------------------------------------------------------------------
    df_I = df[df['Arquivo'].str.startswith('I_')].copy()
    df_B = df[df['Arquivo'].str.startswith('B_')].copy()
    df_C = df[df['Arquivo'].str.startswith('C_')].copy()
    df_O = df[df['Arquivo'].str.startswith('O_')].copy()

    df_I_train, df_I_test = split_train_test(df_I)
    df_B_train, df_B_test = split_train_test(df_B)
    df_C_train, df_C_test = split_train_test(df_C)
    df_O_train, df_O_test = split_train_test(df_O)

    # Concatenar para ter um único DF de treino e um de teste
    df_train_all = pd.concat([df_I_train, df_B_train, df_C_train, df_O_train], ignore_index=True)
    df_test_all = pd.concat([df_I_test, df_B_test, df_C_test, df_O_test], ignore_index=True)

    # -------------------------------------------------------------------------
    # (C) Criar janelas deslizantes
    # -------------------------------------------------------------------------
    window_size = 200
    step_size = 50
    X_train, y_train = process_files_multiclass(df_train_all, window_size, step_size)
    X_test, y_test = process_files_multiclass(df_test_all, window_size, step_size)

    # Embaralhar os dados de treino
    indices_train = np.arange(X_train.shape[0])
    np.random.shuffle(indices_train)
    X_train = X_train[indices_train]
    y_train = y_train[indices_train]

    # -------------------------------------------------------------------------
    # (D) Normalizar (StandardScaler)
    # -------------------------------------------------------------------------
    scaler = StandardScaler()
    X_train_2d = X_train.reshape(-1, X_train.shape[-1])
    X_test_2d = X_test.reshape(-1, X_test.shape[-1])

    X_train_2d = scaler.fit_transform(X_train_2d)
    X_test_2d = scaler.transform(X_test_2d)

    X_train = X_train_2d.reshape(X_train.shape)
    X_test = X_test_2d.reshape(X_test.shape)

    # -------------------------------------------------------------------------
    # (E) Definir múltiplas arquiteturas LSTM para teste
    # -------------------------------------------------------------------------
    lstm_configs = [
        {"architecture": "basic",        "layers": [128],       "dropout": False},
        {"architecture": "basic_dropout","layers": [128],       "dropout": True},
        {"architecture": "stacked",      "layers": [128, 64],   "dropout": False},
        {"architecture": "small_dropout","layers": [64],        "dropout": True},
        {"architecture": "deep_stacked", "layers": [128, 128],  "dropout": False},
        {"architecture": "deep_dropout", "layers": [128, 128],  "dropout": True},
    ]

    class_labels = ["O_", "C_", "B_", "I_"]  # Ordem coerente: 0=O_, 1=C_, 2=B_, 3=I_
    results = []

    # -------------------------------------------------------------------------
    # (F) Treinar e avaliar cada arquitetura
    # -------------------------------------------------------------------------
    for i, config in enumerate(lstm_configs):
        model_name = f"{config['architecture']} - Dropout={config['dropout']}"
        print(f"\nTreinando o modelo {i + 1}/{len(lstm_configs)}: {model_name}")

        # Colocamos todo o treinamento e avaliação em um try/except
        try:
            model = create_lstm_model(config, input_shape=(window_size, 4))

            early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

            with tf.device(device):
                history = model.fit(
                    X_train,
                    y_train,
                    epochs=10,
                    batch_size=32,
                    validation_split=0.1,
                    callbacks=[early_stopping],
                    verbose=1
                )

            # Previsões no conjunto de teste
            y_pred_probs = model.predict(X_test)
            y_pred = np.argmax(y_pred_probs, axis=1)
            y_test_labels = np.argmax(y_test, axis=1)

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

            # Salvamos os resultados
            results.append({
                "Model Config": model_name,
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

            # Plotar e salvar gráficos para cada modelo
            plot_loss(history.history, model_name=model_name)
            plot_confusion_matrix(conf_matrix, labels=class_labels, model_name=model_name)

        except Exception as e:
            # Se ocorrer qualquer erro, imprime e pula para o próximo modelo
            print(f"\n[ERRO] O modelo '{model_name}' falhou com a mensagem:")
            print(e)
            print("Ignorando este modelo e seguindo para o próximo.\n")
            
        finally:
            # Limpeza de sessão/tensorflow para evitar acumular GPU/CPU
            K.clear_session()
            gc.collect()

    # -------------------------------------------------------------------------
    # (G) Exibir resultados finais e salvar CSV
    # -------------------------------------------------------------------------
    print("\nResultados Finais de Todas as Arquiteturas:\n")
    for i, res in enumerate(results):
        print(f"Modelo {i+1}: {res['Model Config']}")
        print(f"  -> Accuracy:           {res['Accuracy']:.4f}")
        print(f"  -> Balanced Accuracy:  {res['Balanced Accuracy']:.4f}")
        print(f"  -> F1 Macro:           {res['F1 Macro']:.4f}")
        print(f"  -> F1 Weighted:        {res['F1 Weighted']:.4f}")
        print(f"  -> Precision Macro:    {res['Precision Macro']:.4f}")
        print(f"  -> Precision Weighted: {res['Precision Weighted']:.4f}")
        print(f"  -> Recall Macro:       {res['Recall Macro']:.4f}")
        print(f"  -> Recall Weighted:    {res['Recall Weighted']:.4f}")
        print(f"  -> MSE:                {res['MSE']:.6f}")
        print(f"  -> MAE:                {res['MAE']:.6f}")
        print("-------------------------------------------------")

    # Salvar métricas em CSV
    results_df = pd.DataFrame(results, columns=[
        "Model Config", "Accuracy", "Balanced Accuracy", "F1 Macro", "F1 Weighted",
        "Precision Macro", "Precision Weighted", "Recall Macro", "Recall Weighted",
        "MSE", "MAE"
    ])
    results_df.to_csv("metrics_experiment4.csv", index=False)

if __name__ == "__main__":
    main()