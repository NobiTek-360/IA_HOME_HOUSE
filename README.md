# IA_HOME_HOUSE
IA pequena para automação residencial ou classificação de mapeamento de contexto falado por voz ou por escrito.

# 🏠 NLU Contextual para Controle de Dispositivos Domésticos

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Modelo de entendimento de linguagem natural (NLU) especializado para assistentes domésticos**, capaz de interpretar comandos em português com alta precisão, extrair entidades e lidar com contexto conversacional.

---

## 📌 Visão Geral

Este projeto implementa um **sistema NLU com arquitetura baseada em Transformer**, otimizado para dispositivos embarcados e baixa latência. Ele suporta:

- **13 intenções** (abrir, fechar, ligar, desligar, definir brilho/cor/velocidade/temperatura/volume/tensão, iniciar/parar, consultar status).
- **Reconhecimento de entidades** (dispositivos, locais, cores, valores numéricos).
- **Resolução de contexto** (pronomes como "ela", "isso", "ali").
- **Detecção de intenções desconhecidas** (UNKNOWN) com base em confiança e margem.
- **Pipeline de treinamento flexível** com suporte a *fine-tuning*, *resume* e expansão de vocabulário.

O modelo foi treinado com um dataset próprio (`dataset_compact.json`) e aprimorado com um **dataset de contexto** (`context_dataset.json`) que fornece variações linguísticas ricas. So irá incluir na próxima versão do projeto 🤝.

---

## ✨ Diferenciais

- **Encoder contextual híbrido**: BPE + CharCNN + Transformer com atenção posicional relativa.
- **Cabeça especializada para intenção**: pooling aprendido que foca em verbos, objetos e modificadores.
- **MLM de palavra inteira**: melhora a compreensão semântica das frases.
- **Perda contrastiva adaptativa**: agrupa embeddings da mesma intenção e separa diferentes.
- **Hard‑example mining**: dá mais peso a exemplos onde o modelo ainda erra.
- **Suporte a CRF** para tagging de entidades.

---
 - Inclui também o gerador de banco de dados "Gerador_nlu_profissional.py" artificial use ele para gerar o banco de dado principal para treinar o modelo.
 - Também inclui o Infer_NLU_cont_especializado.py, ele será usado para fazer a interferência do arquivo binário depois de treinar, ou também durante o treinamento, já que o sistema vai salvando a cada melhora um arquivo binário será salvo pronto para uso.


## 🚀 Como usar

### 1. Pré‑requisitos

- Python 3.8 ou superior
- PyTorch 2.0+ (ou versão compatível com sua GPU)
- Bibliotecas listadas em `requirements.txt`

### 2. Instalação

```bash
git clone https://github.com/seu-usuario/seu-repositorio.git
cd seu-repositorio
pip install -r requirements.txt
