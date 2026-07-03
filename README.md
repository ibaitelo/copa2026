# Copa do Mundo 2026 — Previsão de Placares

## Objetivo

Prever os placares dos jogos da Copa do Mundo FIFA 2026 utilizando dados históricos das fases de qualificatórias e competições anteriores.

A abordagem combina estatísticas de desempenho das seleções (gols, posse de bola, chutes, defesas, etc.) com modelos de aprendizado de máquina para estimar o resultado mais provável de cada partida.

## Estrutura do Projeto

```
copa2026/
├── data/
│   ├── raw/          # Dados brutos extraídos das fontes (não modificar)
│   ├── processed/    # Dados limpos e transformados prontos para modelagem
│   └── external/     # Dados de fontes externas (rankings FIFA, clima, etc.)
├── notebooks/        # Jupyter Notebooks de exploração e análise
├── src/              # Código-fonte modular (coleta, processamento, modelos)
├── outputs/          # Resultados: gráficos, tabelas, previsões exportadas
├── README.md
└── requirements.txt
```

## Fontes de Dados

- **soccerdata** — estatísticas de ligas e seleções nacionais
- **statsbombpy** — dados detalhados de eventos de partidas (StatsBomb Open Data)
- Rankings FIFA, resultados históricos de Copas e qualificatórias

## Tecnologias

| Categoria       | Bibliotecas                          |
|-----------------|--------------------------------------|
| Dados           | pandas, numpy, soccerdata, statsbombpy |
| Modelagem       | scikit-learn, scipy                  |
| Visualização    | matplotlib, seaborn                  |
| Ambiente        | jupyter                              |

## Como Começar

```bash
# 1. Criar e ativar ambiente virtual
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2. Instalar dependências
pip install -r requirements.txt

# 3. Iniciar o Jupyter
jupyter notebook
```

## Fluxo de Trabalho

1. **Coleta** (`src/`) — scripts para baixar e salvar dados em `data/raw/`
2. **Processamento** (`src/`) — limpeza e feature engineering em `data/processed/`
3. **Análise** (`notebooks/`) — EDA e experimentos com modelos
4. **Resultados** (`outputs/`) — previsões e visualizações finais
