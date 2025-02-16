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

# Camadas e otimização para CNN
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization,
    GlobalAveragePooling1D, Input
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Activation
from tensorflow.keras import Model

# Import da callback EarlyStopping
from tensorflow.keras.callbacks import EarlyStopping

# --------------------------------------------------
# Funções Auxiliares de Leitura e Pré-processamento
# --------------------------------------------------

def read_and_rename_file(file_path):
    """
    Lê um arquivo .xls com formato específico (salvo como TSV),
    pulando 22 linhas iniciais, sem cabeçalho, e renomeia as colunas.
    """
    df = pd.read_csv(file_path, skiprows=22, header=None, delimiter='\t')
    df.columns = ['indices', 'velocidade', 'AceleracaoX', 'AceleracaoY', 'AceleracaoZ']
    return df

def determine_damage(filename):
    """
    Determina se o arquivo é saudável (H..), moderado (0.5X)
    ou severo (qualquer outro caso).
    """
    if filename.startswith("H"):
        return "Saudável"
    elif "0.5X" in filename:
        return "Moderado"
    else:
        return "Severo"

def is_variable_speed(filename):
    """
    Retorna 1 se o arquivo tiver "VS" no nome (velocidade variável),
    caso contrário, 0.
    """
    return 1 if "VS" in filename else 0

def calculate_variable_speed(index):
    """
    Cálculo de velocidade variável para cada índice, caso seja 'VS'.
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
    Adiciona uma coluna 'Condição' ao df:
    - Se 'VS' no nome do arquivo, aplica velocidade variável.
    - Caso contrário, extrai o valor (Hz) do nome do arquivo.
    """
    if "VS" in filename:
        df['Condição'] = df['indices'].apply(calculate_variable_speed)
    else:
        condition = filename.split('_')[-1].replace('.xls', '').replace('Hz', '')
        df['Condição'] = int(condition) if condition.isdigit() else None
    return df

def create_sliding_windows(df, window_size, step_size, label):
    """
    Cria janelas deslizantes de tamanho `window_size` com passo `step_size`.
    Cada janela obtém as colunas [AceleracaoX, AceleracaoY, AceleracaoZ, Condição].
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
    Cada arquivo tem um rótulo (Dano_Binário) que se aplica a todas as janelas.
    """
    X_list, y_list = [], []
    for arquivo in df['Arquivo'].unique():
        arquivo_df = df[df['Arquivo'] == arquivo]
        label = arquivo_df['Dano_Binário'].iloc[0]  # Obter o label do arquivo
        X, y = create_sliding_windows(arquivo_df, window_size, step_size, label)
        X_list.append(X)
        y_list.append(y)
    return np.vstack(X_list), np.hstack(y_list)

# --------------------------------------------------
# Definir arquiteturas de CNN
# --------------------------------------------------

def create_cnn_model(architecture, input_shape):
    """
    Cria diferentes variações de arquitetura CNN 1D, dependendo do valor de `architecture`.
    """
    model = Sequential()
    
    # Ajuste de sintaxe: trocado o primeiro "elif" por "if"
    if architecture == "deep":
        model.add(Conv1D(64, 3, activation='relu', input_shape=input_shape))
        model.add(Conv1D(128, 3, activation='relu'))
        model.add(Flatten())
    
    elif architecture == "maxpool":
        model.add(Conv1D(64, 3, activation='relu', input_shape=input_shape))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Conv1D(128, 3, activation='relu'))
        model.add(Flatten())

    else:
        raise ValueError("Invalid architecture")
    
    # Camada densa final para saída binária
    model.add(Dense(1, activation='sigmoid'))

    # Compilação
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# --------------------------------------------------
# Funções de Plot (adaptadas para salvar .png)
# --------------------------------------------------

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
    # -----------------------------
    # Semente (seed) para reprodutibilidade
    # -----------------------------
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    # -----------------------------
    # Evitar estouro de memória na GPU (opcional)
    # -----------------------------
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("Configuração de memória dinâmica para GPU ativada.")
        except RuntimeError as e:
            print(e)

    # -----------------------------
    # Caminho dos dados
    # -----------------------------
    data_dir = "/Users/enzooliveira/Pessoal/VS CODE/IC/raw data"

    # -----------------------------
    # Leitura e pré-processamento
    # -----------------------------
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

    # Criar nova coluna binária para Dano: Moderado/Severo = 1; Saudável = 0
    df['Dano_Binário'] = df['Dano'].apply(lambda x: 1 if x in ['Moderado', 'Severo'] else 0)
    df = df.drop(columns=['Dano'])

    # Substituir valores NaN por 25.0
    df = df.fillna(25.0)

    # Remover a coluna VelocidadeConstante (opcional)
    if 'VelocidadeConstante' in df.columns:
        df = df.drop(columns=['VelocidadeConstante'])

    # -----------------------------
    # Separação em conjuntos
    # -----------------------------
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

    # -----------------------------
    # Configurações de janelas deslizantes
    # -----------------------------
    window_size = 200
    step_size = 50

    # Gerar dados de treinamento (saudáveis + danificados)
    X_train, y_train = process_files(
        pd.concat([df_healthy_train, df_damaged_train]),
        window_size,
        step_size
    )
    # Gerar dados de teste (saudáveis + danificados)
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

    # Normalizar os dados
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

    # -----------------------------
    # Arquiteturas de CNN
    # -----------------------------
    architectures = [
        "deep",
        "maxpool", 
    ]

    # Input shape para a CNN: (200, 4)
    input_shape = (X_train.shape[1], X_train.shape[2])

    # Cria callback de Early Stopping
    early_stop = EarlyStopping(
        monitor='val_loss',   # Pode trocar para 'val_accuracy' se preferir
        patience=5,           # Tenta melhorar por 5 épocas consecutivas
        restore_best_weights=True,
        verbose=1
    )

    results = []

    for arch in architectures:
        print(f"\nTreinando arquitetura: {arch}")
        model = create_cnn_model(arch, input_shape)

        # Tentar usar GPU, se não, CPU
        try:
            with tf.device('/GPU:0'):
                history = model.fit(
                    X_train, y_train,
                    epochs=10,
                    batch_size=32,
                    validation_split=0.1,
                    callbacks=[early_stop],   # <--- AQUI incluímos o Early Stopping
                    verbose=1
                )
        except:
            with tf.device('/CPU:0'):
                history = model.fit(
                    X_train, y_train,
                    epochs=10,
                    batch_size=32,
                    validation_split=0.1,
                    callbacks=[early_stop],   # <--- AQUI incluímos o Early Stopping
                    verbose=1
                )
        
        # Avaliar no conjunto de teste
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

        # Plotar e salvar gráfico de perda
        plot_loss(history, arch)

        # Plotar e salvar matriz de confusão
        plot_confusion_mtx(y_test, y_pred, arch)

    # -----------------------------
    # Exibir e salvar resultados
    # -----------------------------
    results_df = pd.DataFrame(
        results,
        columns=["Architecture", "MSE", "MAE", "Accuracy", "Balanced Accuracy", "F1 Score", "Precision"]
    )
    print("\nResultados Finais:")
    print(results_df)

    # Salvar em CSV para posterior coleta
    results_df.to_csv("metrics_experiment2.csv", index=False)


# --------------------------------------------------
# Inicialização do Script
# --------------------------------------------------
if __name__ == "__main__":
    main()