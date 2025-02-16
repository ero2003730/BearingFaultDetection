#!/bin/bash

# =========================================
# SCRIPT PARA RODAR 4 EXPERIMENTOS EM SEQUÊNCIA
# =========================================

# 1) Criar uma pasta geral para agrupar os resultados
mkdir -p results

# 2) Lista de experimentos (podem ser ajustados para 5,6,7,8...)
EXPS=( "experiment1.5" "experiment2.5" "experiment4" "experiment4.5")

# 3) Loop sobre cada experimento
for EXP in "${EXPS[@]}"; do

  echo "=========================================="
  echo " Rodando $EXP.py "
  echo "=========================================="

  # 3.1) Criar subpasta de resultados para este experimento
  mkdir -p "results/$EXP"

  # 3.2) Executar o script Python do experimento
  #  -u => modo unbuffered, 2>&1 => redirecionar erros para stdout,
  #        | tee => mostrar no terminal E salvar em log
  python -u "$EXP.py" 2>&1 | tee "results/$EXP/log.txt"

  # 3.3) Verificar se deu erro (exit code != 0)
  if [ "${PIPESTATUS[0]}" -ne 0 ]; then
    echo "O experimento $EXP falhou! Veja results/$EXP/log.txt para detalhes."
    exit 1
  fi

  # 3.4) Mover/copiar os arquivos de resultado
  # Se cada experimento gerar um metrics.csv (ou metrics_experimentX.csv),
  # você ajusta conforme seu script. Exemplo:
  if [ -f "metrics.csv" ]; then
    mv "metrics.csv" "results/$EXP/metrics_$EXP.csv"
  fi

  # Mover todos os arquivos .png
  mv *.png "results/$EXP/" 2>/dev/null

  echo "Arquivos do $EXP salvos em results/$EXP"
done

echo "Todos os experimentos foram executados e organizados em ./results"