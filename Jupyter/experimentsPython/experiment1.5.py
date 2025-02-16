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
    confusion_matrix
)
from sklearn.preprocessing import StandardScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, Bidirectional,
    BatchNormalization, Conv1D, MaxPooling1D, Input
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback, EarlyStopping  # <--- IMPORT AQUI
from tensorflow.keras.layers import Attention

# --------------------------------------------------
# Funções Auxiliares
# --------------------------------------------------

def set_seed(seed=42):
    """ Semente para reprodutibilidade. """
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def read_and_rename_file(file_path):
    df = pd.read_csv(file_path, skiprows=22, header=None, delimiter='\t')
    df.columns = ['indices', 'velocidade', 'AceleracaoX', 'AceleracaoY', 'AceleracaoZ']
    return df

def determine_damage(filename):
    if filename.startswith("H"):
        return "Saudável"
    elif "0.5X" in filename:
        return "Moderado"
    else:
        return "Severo"

def is_variable_speed(filename):
    return 1 if "VS" in filename else 0

def calculate_variable_speed(index):
    sampling_frequency = 25600
    time = index
    time_in_cycle = time % 2

    if time_in_cycle <= 1.0:
        speed = time_in_cycle * 40
    else:
        speed = 40 - (time_in_cycle - 1) * 40

    return round(speed * 2) / 2

def extract_condition(df, filename):
    if "VS" in filename:
        df['Condição'] = df['indices'].apply(calculate_variable_speed)
    else:
        condition = filename.split('_')[-1].replace('.xls', '').replace('Hz', '')
        df['Condição'] = int(condition) if condition.isdigit() else None
    return df

def create_sliding_windows(df, window_size, step_size, label):
    """
    Cria janelas deslizantes de tamanho `window_size` com passo `step_size`.
    """
    X, y = [], []
    for i in range(0, len(df) - window_size + 1, step_size):
        window = df.iloc[i:i + window_size]
        X.append(window[['AceleracaoX', 'AceleracaoY', 'AceleracaoZ', 'Condição']].values)
        y.append(label)
    return np.array(X), np.array(y)

def process_files(df, window_size, step_size):
    """
    Percorre todos os arquivos dentro de um DataFrame único e gera as janelas.
    """
    X_list, y_list = [], []
    for arquivo in df['Arquivo'].unique():
        arquivo_df = df[df['Arquivo'] == arquivo]
        label = arquivo_df['Dano_Binário'].iloc[0]  # Obter o label do arquivo
        X, y = create_sliding_windows(arquivo_df, window_size, step_size, label)
        X_list.append(X)
        y_list.append(y)
    return np.vstack(X_list), np.hstack(y_list)

def create_lstm_model(architecture, input_shape):
    """
    Cria variações de modelo baseado em LSTM, dependendo do parâmetro `architecture`.
    Nesta experiência, usaremos 'stacked' e 'bidirectional'.
    """
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import (
        LSTM, Dense, Dropout, Bidirectional,
        BatchNormalization, Conv1D, MaxPooling1D, Flatten, Attention, Input
    )

    model = Sequential()
    
    if architecture == "stacked":
        # Empilhado
        model.add(LSTM(128, return_sequences=True, input_shape=input_shape))
        model.add(LSTM(64, return_sequences=False))
    
    elif architecture == "bidirectional":
        # Bidirecional
        model.add(Bidirectional(LSTM(128, return_sequences=True), input_shape=input_shape))
        model.add(Bidirectional(LSTM(64, return_sequences=False)))
    
    else:
        raise ValueError("Arquitetura inválida para este experimento: escolha 'stacked' ou 'bidirectional'.")

    # Camada final
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def plot_loss(history, architecture):
    """
    Plota e salva o gráfico de perda (loss) de treinamento e validação
    para uma dada `architecture`.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['loss'], label='Treinamento')
    plt.plot(history.history['val_loss'], label='Validação')
    plt.title(f'Loss - {architecture}')
    plt.xlabel('Época')
    plt.ylabel('Loss')
    plt.legend()
    # Salvar a figura
    plt.savefig(f"loss_{architecture}.png", dpi=100)
    plt.close()

def plot_confusion_mtx(y_true, y_pred, architecture):
    """
    Plota e salva a matriz de confusão para a `architecture`.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Matriz de Confusão - {architecture}')
    plt.xlabel('Previsto')
    plt.ylabel('Verdadeiro')
    # Salvar a figura
    plt.savefig(f"confusion_{architecture}.png", dpi=100)
    plt.close()

# --------------------------------------------------
# Função Principal
# --------------------------------------------------

def main():
    # Define a semente
    set_seed(42)

    # Evitar estouro de memória na GPU (opcional)
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("Configuração de memória dinâmica para GPU ativada.")
        except RuntimeError as e:
            print(e)

    # Caminho dos dados
    data_dir = "/Users/enzooliveira/Pessoal/VS CODE/IC/raw data"

    # Ler todos os arquivos .xls
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

    # Remover 'indices' e 'velocidade'
    df = df.drop(columns=['indices', 'velocidade'])

    # Criar nova coluna binária para Dano (Moderado/Severo = 1, Saudável = 0)
    df['Dano_Binário'] = df['Dano'].apply(lambda x: 1 if x in ['Moderado', 'Severo'] else 0)
    df = df.drop(columns=['Dano'])

    # Substituir valores NaN por 25.0
    df = df.fillna(25.0)

    # Remover a coluna VelocidadeConstante
    if 'VelocidadeConstante' in df.columns:
        df = df.drop(columns=['VelocidadeConstante'])

    # Separar dados saudáveis e danificados
    df_healthy_train = df[df['Dano_Binário'] == 0]
    damaged_files = [
        'I_20Hz.xls','B_25Hz.xls','C_30Hz.xls','O_35Hz.xls','I_40Hz.xls',
        'B_60Hz.xls','C_65Hz.xls','O_70Hz.xls','I_75Hz.xls','B_80Hz.xls','I_VS_0_40_0Hz.xls'
    ]
    df_damaged_train = df[df['Arquivo'].isin(damaged_files)]

    assert df_healthy_train.shape[0] == df_damaged_train.shape[0], \
        "Os DataFrames saudáveis e danificados para treino têm tamanhos diferentes!"

    print(f"Tamanho de df_healthy_train: {df_healthy_train.shape[0]} linhas")
    print(f"Tamanho de df_damaged_train: {df_damaged_train.shape[0]} linhas")

    # Dados de teste danificados (excluindo os que foram usados no treino)
    df_damaged_test = df[(df['Dano_Binário'] == 1) & (~df['Arquivo'].isin(damaged_files))]

    # Criar 7 cópias de df_healthy_train para compor o teste
    df_healthy_test = pd.concat([df_healthy_train.copy() for _ in range(7)], ignore_index=True)

    # Definir window_size e step_size
    window_size = 200
    step_size = 50

    # Criar X_train, y_train, X_test, y_test
    X_train, y_train = process_files(
        pd.concat([df_healthy_train, df_damaged_train]),
        window_size,
        step_size
    )
    X_test, y_test = process_files(
        pd.concat([df_healthy_test, df_damaged_test]),
        window_size,
        step_size
    )

    # Embaralhar os dados de treinamento
    indices_train = np.arange(X_train.shape[0])
    np.random.shuffle(indices_train)
    X_train = X_train[indices_train]
    y_train = y_train[indices_train]

    # Normalizar
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

    # Vamos rodar apenas 2 arquiteturas: "bidirectional" e "stacked"
    architectures = ["bidirectional", "stacked"]

    input_shape = (X_train.shape[1], X_train.shape[2])  # (200, 4)

    # Criando callback de Early Stopping
    # Monitorar 'val_loss' e parar caso não melhore em 5 épocas consecutivas
    early_stop = EarlyStopping(monitor='val_loss', 
                               patience=5, 
                               restore_best_weights=True,
                               verbose=1)

    results = []
    for arch in architectures:
        print(f"\nTreinando arquitetura: {arch} (50 épocas)")

        model = create_lstm_model(arch, input_shape)

        # Treinar por 50 épocas (com Early Stopping)
        try:
            with tf.device('/GPU:0'):
                history = model.fit(
                    X_train, y_train,
                    epochs=50,     
                    batch_size=32,
                    validation_split=0.1,
                    callbacks=[early_stop],  # <--- INCLUIR A CALLBACK AQUI
                    verbose=1
                )
        except:
            with tf.device('/CPU:0'):
                history = model.fit(
                    X_train, y_train,
                    epochs=50,
                    batch_size=32,
                    validation_split=0.1,
                    callbacks=[early_stop],  # <--- INCLUIR A CALLBACK AQUI
                    verbose=1
                )
        
        # Avaliar
        y_pred = (model.predict(X_test) > 0.5).astype("int32")
        
        # Métricas
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        balanced_acc = balanced_accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)

        # Guardar resultados
        results.append([arch, mse, mae, accuracy, balanced_acc, f1, precision])

        # Plotar e salvar Loss
        plot_loss(history, arch)

        # Plotar e salvar Matriz de Confusão
        plot_confusion_mtx(y_test, y_pred, arch)

    # Mostrar e Salvar tabela final de resultados
    results_df = pd.DataFrame(results, columns=[
        "Architecture", "MSE", "MAE", "Accuracy", "Balanced Accuracy", "F1 Score", "Precision"
    ])
    print("\nResultados Finais (50 épocas, bidirectional vs stacked):")
    print(results_df)

    # Salvar em CSV para posterior coleta via script
    results_df.to_csv("metrics_experiment_bidirect_stacked.csv", index=False)


# --------------------------------------------------
# Inicialização do Script
# --------------------------------------------------
if __name__ == "__main__":
    main()