import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam

# Configuração do MLflow local
mlflow.set_experiment("Detecção_Parkinson_Deep_MLP")

# =====================================================================
# PASSO 1 e 2: Carregamento, Limpeza Básica e Escalonamento dos Dados
# =====================================================================
def carregar_e_preparar_dados(caminho_csv):
    # Carrega o dataset
    df = pd.read_csv(caminho_csv)
    
    # Remove a coluna 'name' (identificador ASCII)
    if 'name' in df.columns:
        df = df.drop(columns=['name'])
    
    # Separa features (X) e alvo (y)
    X = df.drop(columns=['status'])
    y = df['status']
    
    # Divisão em treino e teste (80% treino, 20% teste)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Aplicar Padronização (StandardScaler)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, X.shape[1]

# =====================================================================
# PASSO 4: Função para criar dinamicamente a arquitetura MLP Profunda
# =====================================================================
def criar_modelo_mlp(input_dim, hidden_layers, learning_rate):
    """
    hidden_layers: lista contendo a quantidade de neurônios por camada. 
                   Ex: [64, 32] criará 2 camadas ocultas.
    """
    modelo = Sequential()
    modelo.add(Input(shape=(input_dim,)))
    
    # Adiciona as camadas ocultas dinamicamente
    for neurons in hidden_layers:
        modelo.add(Dense(neurons, activation='relu'))
        
    # Camada de saída para classificação binária (probabilidade entre 0 e 1)
    modelo.add(Dense(1, activation='sigmoid'))
    
    # Otimizador com a taxa de aprendizado especificada
    otimizador = Adam(learning_rate=learning_rate)
    
    # Compilação utilizando Binary Crossentropy (BCE)
    modelo.compile(
        optimizer=otimizador,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return modelo

# =====================================================================
# PASSOS 5 a 9: Função de Execução de Experimento com MLflow
# =====================================================================
def rodar_experimento(X_train, X_test, y_train, y_test, input_dim, nome_run, hidden_layers, lr, epochs=100, batch_size=16):
    
    # PASSO 5: Configurar o mlflow.start_run()
    with mlflow.start_run(run_name=nome_run):
        
        # PASSO 6: Registrar hiperparâmetros no MLflow
        mlflow.log_param("num_hidden_layers", len(hidden_layers))
        mlflow.log_param("architecture", str(hidden_layers))
        mlflow.log_param("learning_rate", lr)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("optimizer", "Adam")
        mlflow.log_param("loss_function", "binary_crossentropy")
        
        # Inicializa o modelo estruturado
        model = criar_modelo_mlp(input_dim, hidden_layers, lr)
        
        # PASSO 7: Treinar o modelo
        print(f"\nIniciando: {nome_run} {hidden_layers}...")
        model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            verbose=0 # Oculta logs do Keras para clareza no terminal
        )
        
        # PASSO 8: Avaliar o modelo no conjunto de teste
        y_pred_prob = model.predict(X_test)
        y_pred = (y_pred_prob >= 0.5).astype(int).flatten()
        
        # Métricas de avaliação
        loss, _ = model.evaluate(X_test, y_test, verbose=0)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        # PASSO 9: Registrar as métricas finais no MLflow
        mlflow.log_metric("loss", loss)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)
        
        # Logar também o artefato do modelo treinado do Keras
        mlflow.keras.log_model(model, "model")
        
        print(f"Resultado {nome_run} -> Acurácia: {acc:.4f} | F1-Score: {f1:.4f}")

# =====================================================================
# PASSO 10: Execução dos Experimentos de Profundidade Variada
# =====================================================================
if __name__ == "__main__":
    # Caminho do arquivo enviado (ajuste se necessário para o seu diretório de execução)
    caminho_dataset = "parkisons.csv" 
    
    if not os.path.exists(caminho_dataset):
        raise FileNotFoundError(f"Por favor, certifique-se de que o arquivo '{caminho_dataset}' está no mesmo diretório.")
        
    # Carrega e processa os dados
    X_train, X_test, y_train, y_test, input_dim = carregar_e_preparar_dados(caminho_dataset)
    
    # Experimento 1: Rede Rasa / Padrão (2 Camadas Ocultas)
    rodar_experimento(
        X_train, X_test, y_train, y_test, input_dim,
        nome_run="MLP_Rede_Rasa_2_Camadas",
        hidden_layers=[64, 32],
        lr=0.001,
        epochs=120,
        batch_size=16
    )
    
    # Experimento 2: Rede Mais Profunda (3 Camadas Ocultas)
    rodar_experimento(
        X_train, X_test, y_train, y_test, input_dim,
        nome_run="MLP_Deep_3_Camadas",
        hidden_layers=[64, 32, 16],
        lr=0.001,
        epochs=120,
        batch_size=16
    )

    # Experimento 3: Rede Muito Profunda com ajuste fino (4 Camadas Ocultas)
    rodar_experimento(
        X_train, X_test, y_train, y_test, input_dim,
        nome_run="MLP_Deep_4_Camadas",
        hidden_layers=[128, 64, 32, 16],
        lr=0.0005,
        epochs=150,
        batch_size=16
    )
    
    print("\nTodos os experimentos foram concluídos e salvos com sucesso!")
    print("Para visualizar os resultados comparativos, execute o comando abaixo no terminal:")
    print("mlflow ui")