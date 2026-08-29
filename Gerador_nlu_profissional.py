#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gerador NLU V63.21 – Cobertura determinística por família/construção, balanceamento circular,
paráfrases semânticas, linguagem coloquial controlada, negativas de entidade e fronteiras contrastivas.
Schedule como slot temporal opcional; 13 intenções.
"""
import argparse
import json
import logging
import random
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# Tentar importar tqdm para barra de progresso
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

logging.basicConfig(level=logging.INFO,
                     format="%(asctime)s [%(levelname)s] %(message)s",
                     datefmt="%H:%M:%S")
logger = logging.getLogger("NLU_V63")

# ======================== MAPAS PRINCIPAIS ========================
INTENT_MAP = {
    "CLOSE": 0, "GET_STATUS": 1, "OPEN": 2,
    "SET_BRIGHTNESS": 3, "SET_COLOR": 4, "SET_SPEED": 5,
    "SET_TEMPERATURE": 6, "SET_VOLTAGE": 7, "SET_VOLUME": 8,
    "START": 9, "STOP": 10, "TURN_OFF": 11, "TURN_ON": 12
}
ENTITY_TYPE_MAP = {"COLOR": 0, "DATE": 1, "DEVICE": 2, "LOCATION": 3,
                    "RECURRENCE": 4, "TIME": 5, "RELATIVE_TIME": 6, "MEASURE": 7}
OPERATION_MAP = {"NONE": 0, "INCREASE": 1, "DECREASE": 2, "SET": 3}

# ======================== NOVOS MAPAS SEMÂNTICOS ========================
ACTION_MODE_MAP = {"IMMEDIATE": 0, "SCHEDULED": 1, "RECURRING": 2}
TARGET_SCOPE_MAP = {"SINGLE": 0, "GROUP": 1}
VALUE_TYPE_MAP = {
    "NUMBER": 0, "PERCENTAGE": 1, "TEMPERATURE": 2,
    "VOLTAGE": 3, "COLOR": 4, "UNKNOWN": 5
}

# ======================== MATRIZ DE COMPATIBILIDADE ========================
COMPATIBLE_OPERATIONS = {
    "TURN_ON": {"NONE"},
    "TURN_OFF": {"NONE"},
    "OPEN": {"NONE"},
    "CLOSE": {"NONE"},
    "START": {"NONE"},
    "STOP": {"NONE"},
    "GET_STATUS": {"NONE"},
    "SET_BRIGHTNESS": {"NONE", "SET", "INCREASE", "DECREASE"},
    "SET_COLOR": {"SET"},
    "SET_SPEED": {"NONE", "SET", "INCREASE", "DECREASE"},
    "SET_TEMPERATURE": {"NONE", "SET", "INCREASE", "DECREASE"},
    "SET_VOLTAGE": {"NONE", "SET", "INCREASE", "DECREASE"},
    "SET_VOLUME": {"NONE", "SET", "INCREASE", "DECREASE"},
}

# ======================== MAPEAMENTO VERBO → OPERAÇÃO ========================
VERB_TO_OPERATION = {
    "aumente": "INCREASE", "aumentar": "INCREASE", "suba": "INCREASE", "subir": "INCREASE",
    "eleve": "INCREASE", "elevar": "INCREASE", "acelere": "INCREASE", "acelerar": "INCREASE", "clareie": "INCREASE",
    "esquente": "INCREASE", "esquentar": "INCREASE", "aumenta": "INCREASE",
    "diminua": "DECREASE", "diminuir": "DECREASE", "baixe": "DECREASE", "baixar": "DECREASE",
    "reduza": "DECREASE", "reduzir": "DECREASE", "abaixe": "DECREASE", "abaixar": "DECREASE", "esfrie": "DECREASE", "esfriar": "DECREASE",
    "desacelere": "DECREASE", "desacelerar": "DECREASE", "escureça": "DECREASE", "escurecer": "DECREASE", "diminui": "DECREASE",
    "reduz": "DECREASE", "abaixa": "DECREASE", "escurece": "DECREASE", "escureça": "DECREASE", "clareia": "INCREASE", "clareie": "INCREASE",
    "defina": "SET", "configure": "SET", "ajuste": "SET", "regule": "SET",
    "coloque": "SET", "deixe": "SET", "ponha": "SET", "mude": "SET",
    "altera": "SET", "define": "SET", "configura": "SET", "ajusta": "SET",
    "regula": "SET", "coloca": "SET", "deixa": "SET", "põe": "SET",
    "muda": "SET", "altera": "SET"
}

def get_valid_operations(intent: str) -> List[str]:
    return list(COMPATIBLE_OPERATIONS.get(intent, {"NONE"}))

def get_operation_from_verb(text: str) -> str:
    low = text.casefold()
    for verb, op in VERB_TO_OPERATION.items():
        if re.search(rf'\b{re.escape(verb)}\b', low):
            return op
    return "NONE"

def infer_operation_from_text(text: str, intent: str) -> str:
    if intent not in NUMERIC_INTENTS:
        return "NONE"
    low = text.casefold()

    # 1) Formas relativas têm prioridade sobre verbos genéricos como
    # "deixar", que em outras construções significam SET.
    short_attrs = {
        "SET_BRIGHTNESS": ("brilho", "brilhos", "luz", "claridade", "luminosidade", "intensidade", "iluminação", "iluminacao"),
        "SET_SPEED": ("velocidade", "velocidades", "rotação", "rotacao", "ritmo"),
        "SET_TEMPERATURE": ("temperatura", "temperaturas", "graus", "calor"),
        "SET_VOLTAGE": ("voltagem", "tensão", "tensao"),
        "SET_VOLUME": ("volume", "som", "áudio", "audio"),
    }
    attrs = short_attrs.get(intent, ())
    if attrs:
        alt = "|".join(re.escape(a) for a in attrs)
        if re.search(rf"\bmais\s+(?:{alt})\b", low):
            return "INCREASE"
        if re.search(rf"\bmenos\s+(?:{alt})\b", low):
            return "DECREASE"
        if intent == "SET_BRIGHTNESS":
            if re.search(r"\bmais\s+(?:forte|fortes|claro|clara|claros|claras|escuro|escura|escuros|escuras|iluminado|iluminada|iluminados|iluminadas)\b", low):
                return "INCREASE"
            if re.search(r"\bmenos\s+(?:forte|fortes|claro|clara|claros|claras|escuro|escura|escuros|escuras|iluminado|iluminada|iluminados|iluminadas)\b", low):
                return "DECREASE"
            if re.search(r"\bmais\s+(?:fraco|fraca|fracos|fracas)\b", low):
                return "DECREASE"
            if re.search(r"\bmenos\s+(?:fraco|fraca|fracos|fracas)\b", low):
                return "INCREASE"
        if intent == "SET_SPEED":
            if re.search(r"\bmais\s+(?:rápido|rapido|depressa)\b", low):
                return "INCREASE"
            if re.search(r"\b(?:menos|mais)\s+devagar\b", low):
                return "DECREASE"

    # 2) Verbos de ajuste explícito.
    op = get_operation_from_verb(low)
    if op in {"INCREASE", "DECREASE"}:
        return op
    set_words = (
        "defina", "define", "definir", "configure", "configura", "configurar",
        "ajuste", "ajusta", "ajustar", "regule", "regula", "regular",
        "coloque", "coloca", "colocar", "deixe", "deixa", "deixar",
        "ponha", "põe", "por", "mude", "muda", "mudar"
    )
    if any(_contains_lexeme(low, w) for w in set_words):
        return "SET"

    # 3) Medida explícita sem verbo também é SET.
    if re.search(r"\b\d+(?:[.,]\d+)?\s*(?:°C|graus|V|volts|%|por cento|porcento)(?!\w)", low, re.I):
        return "SET"
    return "NONE"

# ======================== CONSTANTES EXISTENTES ========================
NUMERIC_INTENTS = {"SET_BRIGHTNESS", "SET_SPEED", "SET_TEMPERATURE",
                    "SET_VOLTAGE", "SET_VOLUME"}
ACTION_INTENTS = ("TURN_ON", "TURN_OFF", "OPEN", "CLOSE", "START", "STOP")

TEMPORAL_ENTITY_TYPES = {"DATE", "TIME", "RELATIVE_TIME", "RECURRENCE"}

# Expressões deliberadamente NÃO temporais. Elas são importantes como
# hard-negatives porque aparecem com frequência em comandos de ajuste.
# O gerador nunca deve usá-las como RELATIVE_TIME.
TEMPORAL_HARD_NEGATIVES = (
    "mais", "menos", "mais forte", "mais fraca", "mais claro", "mais clara",
    "menos forte", "menos fraca", "menos claro", "menos clara",
    "um pouco", "pouquinho", "ligeiramente", "bastante",
)
SCHEDULE_MARKERS = (
    "amanhã", "depois de amanhã", "na segunda-feira", "na terça-feira",
    "na quarta-feira", "na quinta-feira", "na sexta-feira", "no sábado",
    "no domingo", "na próxima semana", "às ", "daqui a ", "em um minuto",
    "em instantes", "mais tarde", "logo mais", "em breve", "todos os dias",
    "toda semana", "todos os dias úteis", "toda segunda-feira", "todo sábado",
    "todo domingo", "de segunda a sexta", "diariamente", "semanalmente",
    "toda noite", "toda manhã", "todos os finais de semana", "no dia "
)

NUMERIC_ATTR = {
    "SET_BRIGHTNESS": "brilho", "SET_SPEED": "velocidade",
    "SET_TEMPERATURE": "temperatura", "SET_VOLTAGE": "voltagem",
    "SET_VOLUME": "volume"
}

# ======================== ONTOLOGIA ========================
OBJECTS = {
    'luz': {'genero': 'f', 'atributo': 'brilho', 'acoes': ['ligar', 'desligar', 'status', 'valor', 'cor']},
    'lâmpada': {'genero': 'f', 'atributo': 'brilho', 'acoes': ['ligar', 'desligar', 'status', 'valor', 'cor']},
    'luminária': {'genero': 'f', 'atributo': 'brilho', 'acoes': ['ligar', 'desligar', 'status', 'valor', 'cor']},
    'iluminação': {'genero': 'f', 'atributo': 'brilho', 'acoes': ['ligar', 'desligar', 'status', 'valor', 'cor']},
    'abajur': {'genero': 'm', 'atributo': 'brilho', 'acoes': ['ligar', 'desligar', 'status', 'valor', 'cor']},
    'led': {'genero': 'm', 'atributo': 'brilho', 'acoes': ['ligar', 'desligar', 'status', 'valor', 'cor']},
    'televisão': {'genero': 'f', 'atributo': 'volume', 'acoes': ['ligar', 'desligar', 'status', 'valor']},
    'tv': {'genero': 'f', 'atributo': 'volume', 'acoes': ['ligar', 'desligar', 'status', 'valor']},
    'ar condicionado': {'genero': 'm', 'atributo': 'temperatura', 'acoes': ['ligar', 'desligar', 'status', 'valor']},
    'ventilador': {'genero': 'm', 'atributo': 'velocidade', 'acoes': ['ligar', 'desligar', 'status', 'valor']},
    'exaustor': {'genero': 'm', 'atributo': 'velocidade', 'acoes': ['ligar', 'desligar', 'status', 'valor']},
    'termostato': {'genero': 'm', 'atributo': 'temperatura', 'acoes': ['ligar', 'desligar', 'status', 'valor']},
    'umidificador': {'genero': 'm', 'atributo': 'velocidade', 'acoes': ['ligar', 'desligar', 'status', 'valor']},
    'purificador de ar': {'genero': 'm', 'atributo': 'velocidade', 'acoes': ['ligar', 'desligar', 'status', 'valor']},
    'projetor': {'genero': 'm', 'atributo': 'volume', 'acoes': ['ligar', 'desligar', 'status', 'valor']},
    'porta': {'genero': 'f', 'atributo': 'abertura', 'acoes': ['abrir', 'fechar', 'status']},
    'janela': {'genero': 'f', 'atributo': 'abertura', 'acoes': ['abrir', 'fechar', 'status']},
    'cortina': {'genero': 'f', 'atributo': 'abertura', 'acoes': ['abrir', 'fechar', 'status']},
    'persiana': {'genero': 'f', 'atributo': 'abertura', 'acoes': ['abrir', 'fechar', 'status']},
    'fechadura inteligente': {'genero': 'f', 'atributo': 'estado', 'acoes': ['abrir', 'fechar', 'status']},
    'alarme': {'genero': 'm', 'atributo': 'estado', 'acoes': ['ligar', 'desligar', 'status']},
    'câmera': {'genero': 'f', 'atributo': 'estado', 'acoes': ['ligar', 'desligar', 'iniciar', 'parar', 'status']},
    'cafeteira': {'genero': 'f', 'atributo': 'estado', 'acoes': ['ligar', 'desligar', 'status']},
    'aspirador': {'genero': 'm', 'atributo': 'estado', 'acoes': ['ligar', 'desligar', 'iniciar', 'parar', 'status']},
    'robô aspirador': {'genero': 'm', 'atributo': 'estado', 'acoes': ['ligar', 'desligar', 'iniciar', 'parar', 'status']},
    'lavadora': {'genero': 'f', 'atributo': 'estado', 'acoes': ['ligar', 'desligar', 'iniciar', 'parar', 'status']},
    'máquina': {'genero': 'f', 'atributo': 'estado', 'acoes': ['ligar', 'desligar', 'iniciar', 'parar', 'status']},
    'tomada inteligente': {'genero': 'f', 'atributo': 'estado', 'acoes': ['ligar', 'desligar', 'status']},
    'ferro de solda': {'genero': 'm', 'atributo': 'temperatura', 'acoes': ['ligar', 'desligar', 'status', 'valor']},
    'fonte de bancada': {'genero': 'f', 'atributo': 'voltagem', 'acoes': ['ligar', 'desligar', 'status', 'valor']},
    'nobreak': {'genero': 'm', 'atributo': 'voltagem', 'acoes': ['ligar', 'desligar', 'status', 'valor']},
    'bateria': {'genero': 'f', 'atributo': 'voltagem', 'acoes': ['ligar', 'desligar', 'status', 'valor']},
    'banco de baterias': {'genero': 'm', 'atributo': 'voltagem', 'acoes': ['ligar', 'desligar', 'status', 'valor']},
    'inversor': {'genero': 'm', 'atributo': 'voltagem', 'acoes': ['ligar', 'desligar', 'iniciar', 'parar', 'status', 'valor']},
    'estabilizador': {'genero': 'm', 'atributo': 'voltagem', 'acoes': ['ligar', 'desligar', 'status', 'valor']},
    'gerador elétrico': {'genero': 'm', 'atributo': 'voltagem', 'acoes': ['ligar', 'desligar', 'iniciar', 'parar', 'status', 'valor']},
    'painel solar': {'genero': 'm', 'atributo': 'voltagem', 'acoes': ['ligar', 'desligar', 'status', 'valor']},
    'carregador': {'genero': 'm', 'atributo': 'voltagem', 'acoes': ['ligar', 'desligar', 'status', 'valor']},
    'furadeira': {'genero': 'f', 'atributo': 'estado', 'acoes': ['ligar', 'desligar', 'iniciar', 'parar', 'status']},
    'Makita': {'genero': 'f', 'atributo': 'estado', 'acoes': ['ligar', 'desligar', 'iniciar', 'parar', 'status']},
    'Skil': {'genero': 'f', 'atributo': 'estado', 'acoes': ['ligar', 'desligar', 'iniciar', 'parar', 'status']},
    'parafusadeira': {'genero': 'f', 'atributo': 'estado', 'acoes': ['ligar', 'desligar', 'iniciar', 'parar', 'status']},
    'soprador térmico': {'genero': 'm', 'atributo': 'temperatura', 'acoes': ['ligar', 'desligar', 'iniciar', 'parar', 'status', 'valor']},
    'serra circular': {'genero': 'f', 'atributo': 'estado', 'acoes': ['ligar', 'desligar', 'iniciar', 'parar', 'status']},
    'esmerilhadeira': {'genero': 'f', 'atributo': 'estado', 'acoes': ['ligar', 'desligar', 'iniciar', 'parar', 'status']},
    'lixadeira': {'genero': 'f', 'atributo': 'estado', 'acoes': ['ligar', 'desligar', 'iniciar', 'parar', 'status']},
}

LOCATIONS = {
    'sala': 'f', 'sala de estar': 'f', 'quarto': 'm', 'cozinha': 'f',
    'banheiro': 'm', 'escritório': 'm', 'garagem': 'f', 'jardim': 'm',
    'corredor': 'm', 'lavanderia': 'f', 'hall': 'm', 'sala de jantar': 'f',
    'quarto de hóspedes': 'm', 'varanda': 'f', 'quintal': 'm', 'terraço': 'm',
    'área de serviço': 'f', 'escada': 'f', 'pátio': 'm', 
    'porão': 'm', 'sótão': 'm', 'telhado': 'm', 'área externa': 'f',
    'bancada': 'f', 'oficina': 'f', 'área de trabalho': 'f'
}

ALLOWED_LOCATIONS = {
    'cafeteira': ['cozinha', 'escritório'],
    'aspirador': ['sala', 'sala de estar', 'quarto', 'cozinha', 'corredor', 'garagem'],
    'robô aspirador': ['sala', 'sala de estar', 'quarto', 'cozinha', 'corredor', 'garagem'],
    'lavadora': ['garagem', 'banheiro', 'cozinha', 'lavanderia', 'área de serviço'],
    'máquina': ['garagem', 'banheiro', 'cozinha', 'lavanderia', 'área de serviço'],
    'ferro de solda': ['garagem', 'escritório', 'porão'],
    'fonte de bancada': ['garagem', 'escritório', 'porão'],
    'ar condicionado': ['sala', 'sala de estar', 'quarto', 'escritório'],
    'exaustor': ['cozinha', 'banheiro', 'lavanderia', 'garagem'],
    'televisão': ['sala', 'sala de estar', 'quarto', 'escritório'],
    'tv': ['sala', 'sala de estar', 'quarto', 'escritório'],
    'cortina': ['sala', 'sala de estar', 'quarto', 'escritório'],
    'persiana': ['sala', 'sala de estar', 'quarto', 'escritório'],
    'fechadura inteligente': ['garagem', 'escritório', 'hall', 'porta'],
    'termostato': ['sala', 'sala de estar', 'quarto', 'escritório', 'corredor'],
    'umidificador': ['quarto', 'escritório', 'sala'],
    'purificador de ar': ['sala', 'sala de estar', 'quarto', 'escritório'],
    'projetor': ['sala de estar', 'sala de jantar', 'escritório'],
    'tomada inteligente': ['sala', 'sala de estar', 'quarto', 'cozinha', 'escritório', 'garagem'],
    'nobreak': ['escritório', 'sala', 'quarto', 'garagem', 'porão'],
    'bateria': ['garagem', 'porão', 'área de serviço', 'quintal'],
    'banco de baterias': ['garagem', 'porão', 'área de serviço'],
    'inversor': ['garagem', 'área de serviço', 'porão', 'área externa'],
    'estabilizador': ['escritório', 'sala', 'quarto', 'garagem'],
    'gerador elétrico': ['garagem', 'quintal', 'área externa', 'porão'],
    'painel solar': ['telhado', 'quintal', 'área externa'],
    'carregador': ['quarto', 'sala', 'escritório', 'garagem', 'cozinha'],
    'furadeira': ['bancada', 'garagem', 'oficina', 'área de trabalho'],
    'Makita': ['bancada', 'garagem', 'oficina', 'área de trabalho'],
    'Skil': ['bancada', 'garagem', 'oficina', 'área de trabalho'],
    'parafusadeira': ['bancada', 'garagem', 'oficina', 'área de trabalho'],
    'soprador térmico': ['bancada', 'garagem', 'oficina', 'área de trabalho'],
    'serra circular': ['bancada', 'garagem', 'oficina', 'área de trabalho'],
    'esmerilhadeira': ['bancada', 'garagem', 'oficina', 'área de trabalho'],
    'lixadeira': ['bancada', 'garagem', 'oficina', 'área de trabalho']
}

COLORS_SINGLE = ["azul", "vermelho", "verde", "amarelo", "roxo", "laranja", "rosa", "branco", "preto"]
COLORS_COMPOUND = [
    "azul escuro", "azul claro", "verde escuro", "verde claro",
    "vermelho escuro", "vermelho claro", "amarelo claro", "cinza escuro"
]
COLORS = COLORS_SINGLE + COLORS_COMPOUND
COLOR_FEM = {"vermelho": "vermelha", "amarelo": "amarela", "roxo": "roxa",
             "branco": "branca", "preto": "preta",
             "azul": "azul", "verde": "verde", "laranja": "laranja", "rosa": "rosa"}

DEVICE_COMPOUND_FORMS = {o for o in OBJECTS if " " in o}
LOCATION_COMPOUND_FORMS = {l for l in LOCATIONS if " " in l}
COLOR_COMPOUND_FORMS = set(COLORS_COMPOUND)
DEVICE_COMMAND_GROUPS = [("Makita", "Skil")]

LINGUISTIC_FAMILIES = (
    "short", "imperative", "question", "declarative",
    "polite", "colloquial", "negative", "conditional"
)

# ======================== FUNÇÕES DE INFERÊNCIA DOS NOVOS ATRIBUTOS (CORRIGIDAS) ========================
def has_explicit_temporal_entity(entities: List[Dict]) -> bool:
    return any(e.get("type") in TEMPORAL_ENTITY_TYPES for e in entities)

def temporal_entities_are_unambiguous(text: str, entities: List[Dict]) -> bool:
    """Protege a fronteira temporal: palavras de intensidade não contam como tempo."""
    low = text.casefold()
    for e in entities:
        if e.get("type") == "RELATIVE_TIME":
            value = str(e.get("value", "")).casefold().strip()
            if value in {"mais", "menos", "mais forte", "mais fraca", "mais claro", "mais clara",
                         "menos forte", "menos fraca", "menos claro", "menos clara",
                         "um pouco", "pouquinho", "ligeiramente", "bastante"}:
                return False
    return True

def infer_action_mode(text: str, entities: List[Dict]) -> str:
    low = text.casefold()
    if any(e.get("type") == "RECURRENCE" for e in entities):
        return "RECURRING"
    if any(e.get("type") in ("DATE", "TIME", "RELATIVE_TIME") for e in entities):
        return "SCHEDULED"
    return "IMMEDIATE"

def infer_target_scope(entities: List[Dict], text: str) -> str:
    devices = [e for e in entities if e.get("type") == "DEVICE"]
    if len(devices) > 1:
        return "GROUP"
    low = text.casefold()
    if re.search(r'\b(todas|todos|vários|várias|ambas|ambos|os\s+dispositivos|as\s+luzes)\b', low):
        return "GROUP"
    return "SINGLE"

def infer_value_type(entities: List[Dict], intent: str) -> str:
    if intent == "SET_COLOR":
        return "COLOR"
    for e in entities:
        typ = e.get("type")
        if typ == "COLOR":
            return "COLOR"
        if typ != "MEASURE":
            continue
        val = str(e.get("value", "")).casefold().strip()
        if "%" in val or "por cento" in val or "porcento" in val:
            return "PERCENTAGE"
        if "°c" in val or "grau" in val:
            return "TEMPERATURE"
        if re.search(r"\d+(?:[.,]\d+)?\s*(?:v|volts?)\b", val, re.I):
            return "VOLTAGE"
        if re.fullmatch(r"\d+(?:[.,]\d+)?", val):
            return "NUMBER"
        return "NUMBER"
    return "UNKNOWN"

def enrich_sample(sample: Dict) -> Dict:
    if sample:
        sample["action_mode"] = infer_action_mode(sample["text"], sample.get("entities", []))
        sample["target_scope"] = infer_target_scope(sample.get("entities", []), sample["text"])
        sample["value_type"] = infer_value_type(sample.get("entities", []), sample["intent"])
    return sample

# ======================== FUNÇÕES AUXILIARES EXISTENTES ========================
def entity_form(value: str) -> str:
    if value in DEVICE_COMPOUND_FORMS or value in LOCATION_COMPOUND_FORMS or value in COLOR_COMPOUND_FORMS:
        return "compound"
    return "single"

def entity_signature(sample: Dict) -> str:
    devices = [e["value"] for e in sample.get("entities", []) if e["type"] == ENTITY_TYPE_MAP["DEVICE"] or e["type"] == "DEVICE"]
    locations = [e["value"] for e in sample.get("entities", []) if e["type"] == ENTITY_TYPE_MAP["LOCATION"] or e["type"] == "LOCATION"]
    colors = [e["value"] for e in sample.get("entities", []) if e["type"] == ENTITY_TYPE_MAP["COLOR"] or e["type"] == "COLOR"]
    def form(v: str) -> str:
        base = re.sub(r"^(minha|meu|minhas|meus)\s+", "", v.casefold())
        base = re.sub(r"\s+3?$", "", base)
        base = re.sub(r"\s+aqui$", "", base)
        return "compound" if " " in base else "single"
    d = "multi" if len(devices) > 1 else (form(devices[0]) if devices else "none")
    l = form(locations[0]) if locations else "none"
    c = "compound" if any(" " in c for c in colors) else ("single" if colors else "none")
    return f"D:{d}|L:{l}|C:{c}"

def syntax_family(text: str) -> str:
    s = text.casefold().strip()
    if s.endswith("?"):
        return "question"
    if re.match(r"^(por favor|por gentileza),", s):
        return "polite"
    if re.match(r"^(gostaria|eu gostaria|você poderia|eu queria que|se puder)\b", s):
        return "polite"
    if re.match(r"^(não|nao)\b", s):
        return "negative"
    if re.match(r"^(seria bom|queria que)\b", s):
        return "conditional"
    if re.match(r"^(quero|eu quero|vou|preciso|eu queria)\b", s):
        return "declarative"
    if re.match(r"^(bota|deixa|liga|desliga|abre|fecha|ativa|desativa|inicia|inicie|pare|para|ligue|desligue|abra|feche|ative|desative|comece|programe|programe-se)\b", s):
        if re.search(r"\b(aí|pra mim|pra funcionar|direitinho)\b", s):
            return "colloquial"
        return "imperative"
    return "short"

ATTR_SYNONYMS = {
    "brilho": [("brilho", "m"), ("intensidade", "f"), ("luminosidade", "f"), ("claridade", "f")],
    "velocidade": [("velocidade", "f"), ("rotação", "f"), ("ritmo", "m")],
    "temperatura": [("temperatura", "f"), ("grau de temperatura", "m")],
    "voltagem": [("voltagem", "f"), ("tensão", "f")],
    "volume": [("volume", "m"), ("som", "m"), ("áudio", "m")],
    "abertura": [("abertura", "f"), ("posição", "f")],
    "estado": [("estado", "m"), ("situação", "f"), ("status", "m"), ("modo", "m")],
}

VERB_FORMS = {
    "TURN_ON": {
        # V63.16: núcleo exclusivo de ENERGIA/ESTADO.
        # Removidos verbos genéricos de "funcionar" para não colidir com START.
        "imp": ["ligue", "acenda", "ative", "acione", "energize", "conecte",
                "liga", "pode ligar", "deixa ligado", "deixe aceso", "deixa acesa",
                "energiza"],
        "fin": ["liga", "acende", "aciona", "ativa", "conecta", "deixa ligado",
                "deixa aceso", "deixa acesa"],
        "inf": "ligar",
        "subj": ["ligasse", "ativasse"]
    },
    "TURN_OFF": {
        "imp": ["desligue", "apague", "desative", "desconecte", "corte a alimentação",
                "tire da tomada", "pode desligar", "desliga", "deixa desligado",
                "não deixe ligado", "não mantenha aceso", "tira", "apaga", "desliga aí",
                "corta a energia", "desativa", "desconecta", "desligue a luz"],
        "fin": ["desliga", "apaga", "desativa", "desconecta", "corta a alimentação",
                "tira da tomada", "deixa desligado", "não deixa ligado", "não mantenha aceso"],
        "inf": "desligar",
        "subj": ["desligasse", "desativasse"]
    },
    "OPEN": {
        "imp": ["abra", "destranque", "libere", "escancare", "destrave", "deixe aberto",
                "pode abrir", "abre", "destrava", "libera", "desbloqueie"],
        "fin": ["abre", "destranca", "libera", "destrava", "deixa aberto"],
        "inf": "abrir",
        "subj": ["abrisse", "destravasse"]
    },
    "CLOSE": {
        "imp": ["feche", "tranque", "trave", "bloqueie", "deixe fechado",
                "pode fechar", "fecha", "trava", "bloqueia", "não deixe aberto"],
        "fin": ["fecha", "tranca", "trava", "bloqueia", "deixa fechado", "não deixa aberto"],
        "inf": "fechar",
        "subj": ["fechasse", "travasse"]
    },
    "START": {
        # V63.16: início de PROCESSO/CICLO. Nunca usa "ativar" nem
        # "colocar para funcionar", que podem ser interpretados como energia.
        "imp": ["inicie", "comece", "rode", "dê o play", "ponha para rodar",
                "pode iniciar", "começa", "inicia", "mande iniciar",
                "dê partida", "comece o ciclo", "inicie o ciclo"],
        "fin": ["inicia", "começa", "roda", "põe para rodar", "dá partida",
                 "começa o ciclo", "inicia o ciclo", "executa o ciclo"],
        "inf": "iniciar",
        "subj": ["iniciasse", "começasse"]
    },
    "STOP": {
        # V63.16: interrupção de PROCESSO/CICLO. "desative" pertence a
        # TURN_OFF e não deve aparecer em STOP.
        "imp": ["pare", "interrompa", "pause", "aborte", "cesse",
                "pare de rodar", "pode parar", "para", "mande parar",
                "interrompa o funcionamento de", "não deixe rodando"],
        "fin": ["para", "interrompe", "pausa", "cessa", "deixa de rodar", "não deixa rodando"],
        "inf": "parar",
        "subj": ["parasse", "interrompesse"]
    },
}

PARTICIPLE = {
    "TURN_ON": ("ligado", "ligada", "aceso", "acesa"),
    "TURN_OFF": ("desligado", "desligada", "apagado", "apagada"),
    "OPEN": ("aberto", "aberta", "destravado", "destravada"),
    "CLOSE": ("fechado", "fechada", "travado", "travada"),
    "START": ("iniciado", "iniciada", "rodando", "rodando"),
    "STOP": ("parado", "parada", "interrompido", "interrompida"),
}
OPPOSITE_PARTICIPLE = {
    "TURN_OFF": PARTICIPLE["TURN_ON"],
    "TURN_ON": PARTICIPLE["TURN_OFF"],
    "CLOSE": PARTICIPLE["OPEN"],
    "OPEN": PARTICIPLE["CLOSE"],
    "START": PARTICIPLE["STOP"],
    "STOP": PARTICIPLE["START"],
}

# ======================== COMPOSIÇÃO CONTEXTUAL DE ENTIDADES ========================
COMPOUND_ENTITIES = {
    "DEVICE": {
        "soprador térmico", "serra circular", "furadeira de impacto",
        "furadeira elétrica", "furadeira sem fio", "lixadeira orbital",
        "esmerilhadeira angular", "cortador de grama", "robô aspirador",
        "ar condicionado", "purificador de ar", "ferro de solda",
        "fonte de bancada", "banco de baterias", "gerador elétrico",
        "painel solar", "tomada inteligente", "fechadura inteligente",
        "Makita Skil"
    },
    "LOCATION": {
        "sala de estar", "sala de jantar", "quarto de hóspedes",
        "área de serviço", "área de trabalho", "área externa",
        "oficina mecânica", "bancada de trabalho"
    },
    "COLOR": {
        "azul escuro", "azul claro", "azul marinho",
        "verde escuro", "verde claro", "vermelho escuro",
        "vermelho claro", "amarelo claro", "cinza escuro",
        "cinza claro", "branco gelo", "branco fosco"
    }
}

COORDINATION_WORDS = {
    "e", "ou", "nem", "mas", "bem como"
}

def split_coordinated_entities(text: str, candidates: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    ordered = sorted(candidates, key=lambda x: len(x[0]), reverse=True)
    result = []
    occupied = []
    for value, typ in ordered:
        for m in re.finditer(r"(?<![A-Za-zÀ-ÿ0-9])" + re.escape(value) +
                             r"(?![A-Za-zÀ-ÿ0-9])", text, re.I):
            s, e = m.span()
            if any(s < oe and os < e for os, oe in occupied):
                continue
            result.append((value, typ))
            occupied.append((s, e))
            break
    return result

def is_compound_entity(value: str, entity_type: str) -> bool:
    return value.casefold() in {x.casefold() for x in COMPOUND_ENTITIES.get(entity_type, set())}

# ======================== INTENT LEXICAL BOUNDARIES ========================
INTENT_LEXICAL_FAMILIES = {
    "TURN_ON": (
        "liga", "ligar", "ligue", "acende", "acender", "acenda",
        "deixa ligado", "deixa ligada", "deixar ligado", "deixar ligada", "ativa", "ativar", "ative",
        "energiza", "energizar",
        "aceso", "acesa", "deixa aceso", "deixar aceso",
    ),
    "TURN_OFF": (
        "desliga", "desligar", "desligue", "apaga", "apagar", "apague",
        "deixa desligado", "deixar desligado", "desativa", "desativar",
        "corta a energia", "tirar da tomada", "apagado", "apagada",
        "desligado", "desligada", "não ligado", "não aceso",
    ),
    "START": (
        "inicia", "iniciar", "inicie", "começa", "começar", "comece",
        "dá partida", "dar partida", "coloca para rodar", "colocar para rodar",
        "manda rodar", "mandar rodar", "iniciado", "rodando",
    ),
    "STOP": (
        "para", "parar", "pare", "interrompe", "interromper", "interrompa",
        "pausa", "pausar", "pause", "cessa", "cessar",
        "deixa de rodar", "deixar de rodar", "parado", "parada",
    ),
    "OPEN": (
        "abre", "abrir", "abra", "destrava", "destravar", "destrave",
        "libera", "liberar", "libere", "aberto", "aberta", "destravado",
    ),
    "CLOSE": (
        "fecha", "fechar", "feche", "trava", "travar", "trave",
        "bloqueia", "bloquear", "bloqueie", "fechado", "fechada",
        "travado", "travada", "não aberto",
    ),
    "SET_TEMPERATURE": (
        "temperatura", "esfria", "esfriar", "esquenta", "esquentar",
        "aquece", "aquecer", "resfria", "resfriar", "graus",
    ),
    "SET_SPEED": (
        "velocidade", "acelera", "acelerar", "desacelera", "desacelerar",
        "mais rápido", "mais devagar", "mais rapido", "mais devagar",
        "rápido", "devagar",
    ),
    "SET_BRIGHTNESS": (
        "brilho", "luminosidade", "intensidade da luz", "intensidade", "claridade",
        "iluminação", "iluminacao", "mais luz", "menos luz", "luz mais forte", "luz mais fraca",
        "mais claro", "menos claro", "mais escuro", "mais escura", "menos escuro", "menos escura",
        "mais iluminado", "mais iluminada", "menos iluminado", "menos iluminada",
        "clareia", "clarear", "clareie", "escurece", "escurecer",
        "mais forte", "menos forte", "mais fraca", "mais fraco",
        "por cento", "porcento", "%",
    ),
    "SET_VOLUME": (
        "volume", "mais alto", "mais baixo", "aumenta o som", "baixa o som",
        "aumentar o som", "diminuir o som", "alto", "baixo",
    ),
    "SET_VOLTAGE": (
        "voltagem", "tensão", "tensao", "voltagem em", "tensão em", "tensao em",
        "volts", "V",
    ),
    "SET_COLOR": (
        "cor", "em vermelho", "em azul", "em verde", "em amarelo",
        "azul escuro", "azul claro", "verde escuro", "verde claro",
        "vermelho escuro", "vermelho claro", "cor",
    ),
}

HARD_NEGATIVE_CASES = [
    ("liga a luz da sala", "TURN_ON"),
    ("desliga a luz da sala", "TURN_OFF"),
    ("inicia o ventilador da sala", "START"),
    ("para o ventilador da sala", "STOP"),
    ("abre a porta da sala", "OPEN"),
    ("fecha a porta da sala", "CLOSE"),
    ("esfria o ar condicionado da sala", "SET_TEMPERATURE"),
    ("aumenta a velocidade do ventilador da sala", "SET_SPEED"),
    ("aumenta o brilho da luz da sala", "SET_BRIGHTNESS"),
    ("abaixa o volume da tv da sala", "SET_VOLUME"),
    ("sobe a voltagem da fonte de bancada", "SET_VOLTAGE"),
    ("coloca a luz da sala em vermelho", "SET_COLOR"),
    ("qual a temperatura do ar condicionado", "GET_STATUS"),
    ("qual a velocidade do ventilador", "GET_STATUS"),
    ("qual o brilho da luz", "GET_STATUS"),
    ("qual o volume da tv", "GET_STATUS"),
    ("qual a voltagem da fonte", "GET_STATUS"),
    ("qual a cor da luz", "GET_STATUS"),
    ("ativa a câmera da sala", "TURN_ON"),
    ("inicia a câmera da sala", "START"),
    ("desativa a câmera da sala", "TURN_OFF"),
    ("interrompe a câmera da sala", "STOP"),
    ("deixa a luz da sala mais forte", "SET_BRIGHTNESS"),
    ("cor da luz: azul", "GET_STATUS"),
]

# ======================== FRONTEIRAS SEMÂNTICAS ========================
INTENT_SIGNATURES = {
    "TURN_ON": {
        "positive": ("ligar", "liga", "ligue", "acender", "acende", "acenda",
                     "ativar", "ativa", "ative", "energizar", "energiza",
                     "deixar ligado", "deixar ligada", "deixa ligado", "deixa ligada",
                     "aceso", "acesa", "deixa aceso", "deixa acesa"),
        "neighbors": ("iniciar", "inicia", "inicie", "começar", "começa", "comece",
                      "dar partida", "parar", "pare", "interromper",
                      "interrompe", "desligar", "desliga", "desligue", "apagar", "apaga"),
    },
    "TURN_OFF": {
        "positive": ("desligar", "desliga", "desligue", "apagar", "apaga", "apague",
                     "desativar", "desativa", "desative", "deixar desligado",
                     "deixa desligado", "tirar da tomada", "cortar a energia",
                     "apagado", "apagada", "desligado", "desligada", "não ligado"),
        "neighbors": ("parar", "pare", "interromper", "interrompe",
                      "iniciar", "inicia", "ligar", "liga", "ligue", "acender", "acende"),
    },
    "START": {
        "positive": ("iniciar", "inicia", "inicie", "começar", "começa", "comece",
                     "dar partida", "dá partida", "dê partida", "colocar para rodar",
                     "coloca para rodar", "mandar rodar", "manda rodar", "rodando"),
        "neighbors": ("desligar", "desliga", "desligue", "apagar", "apaga",
                      "parar", "pare", "interromper", "interrompe"),
    },
    "STOP": {
        "positive": ("parar", "para", "pare", "interromper", "interrompe",
                     "interrompa", "pausar", "pausa", "pause", "cessar", "cessa",
                     "deixar de rodar", "deixa de rodar", "parado", "parada"),
        "neighbors": ("iniciar", "inicia", "inicie", "começar", "começa",
                      "ligar", "liga", "ligue", "desligar", "desliga", "desligue"),
    },
    "OPEN": {
        "positive": ("abrir", "abre", "abra", "destravar", "destrava", "destrave",
                     "liberar", "libera", "libere", "aberto", "aberta", "destravado"),
        "neighbors": ("fechar", "fecha", "feche", "travar", "trava", "trave",
                      "bloquear", "bloqueia", "fechado", "fechada"),
    },
    "CLOSE": {
        "positive": ("fechar", "fecha", "feche", "travar", "trava", "trave",
                     "bloquear", "bloqueia", "bloqueie", "fechado", "fechada",
                     "travado", "travada", "não aberto"),
        "neighbors": ("abrir", "abre", "abra", "destravar", "destrava", "destrave",
                      "liberar", "libera", "aberto", "aberta"),
    },
    "SET_TEMPERATURE": {
        "positive": ("temperatura", "esfriar", "esfria", "esfrie", "resfriar",
                     "resfria", "esquentar", "esquenta", "esquente", "aquecer",
                     "aquece", "aqueça", "baixe", "reduza", "aumente", "baixar", "reduzir", "aumentar",
                     "mais frio", "mais quente", "graus", "°C"),
        "neighbors": (),
    },
    "SET_SPEED": {
        "positive": ("velocidade", "rotação", "ritmo", "acelerar", "acelera",
                     "acelere", "desacelerar", "desacelera", "desacelere",
                     "aumente", "reduza", "diminua", "baixe",
                     "reduzir", "diminuir", "abaixar", "aumentar", "acelerar", "desacelerar",
                     "mais rápido", "mais devagar", "mais rapido", "rápido", "devagar"),
        "neighbors": (),
    },
    "SET_BRIGHTNESS": {
        "positive": ("brilho", "luminosidade", "intensidade da luz", "intensidade", "claridade",
                      "iluminação", "iluminacao", "luz", "mais claro", "menos claro",
                      "mais iluminado", "mais iluminada", "menos iluminado", "menos iluminada",
                      "clareia", "clarear", "clareie", "escurece", "escurecer",
                      "mais forte", "menos forte", "mais fraca", "mais fraco",
                      "por cento", "porcento", "%"),
        "neighbors": (),
    },
    "SET_VOLUME": {
        "positive": ("volume", "som", "mais alto", "mais baixo", "aumentar",
                     "aumenta", "aumente", "diminuir", "diminua", "reduza",
                     "abaixar", "abaixa", "abaixe", "aumentar o som", "aumenta o som",
                     "baixar o som", "baixa o som", "alto", "baixo"),
        "neighbors": (),
    },
    "SET_VOLTAGE": {
        "positive": ("voltagem", "tensão", "tensao", "volts", "volt", "V",
                     "aumente", "reduza", "diminua", "baixe", "aumentar", "reduzir", "diminuir", "baixar"),
        "neighbors": (),
    },
    "SET_COLOR": {
        "positive": ("cor", "vermelho", "vermelha", "azul", "verde", "amarelo", "amarela",
                     "roxo", "roxa", "laranja", "branco", "branca", "preto", "preta",
                     "rosa", "ciano", "magenta", "azul escuro", "azul claro",
                     "verde escuro", "verde claro", "vermelho escuro", "vermelho claro",
                     "vermelha escura", "vermelha clara"),
        "neighbors": (),
    },
}

def has_schedule_marker(text: str) -> bool:
    low = text.casefold()
    return any(_contains_lexeme(low, marker.strip()) if marker.strip() else False
               for marker in SCHEDULE_MARKERS)

def _contains_lexeme(text: str, phrase: str) -> bool:
    p = re.escape(phrase.casefold().strip()).replace(r"\ ", r"\s+")
    return bool(re.search(r"(?<![A-Za-zÀ-ÿ0-9])" + p +
                          r"(?![A-Za-zÀ-ÿ0-9])", text.casefold()))

def intent_boundary_ok(text: str, intent: str) -> bool:
    low = text.casefold()
    # Negação controlada: a intenção pode ser expressa pelo estado oposto
    # explicitamente negado ("não deixe a porta aberta" = CLOSE).
    negative_state = {
        "TURN_ON": ("não deixe", "não mantenha", "desligado", "desligada", "apagado", "apagada"),
        "TURN_OFF": ("não deixe", "não mantenha", "ligado", "ligada", "aceso", "acesa"),
        "OPEN": ("não deixe", "não mantenha", "fechado", "fechada", "travado", "travada"),
        "CLOSE": ("não deixe", "não mantenha", "aberto", "aberta", "destravado", "destravada"),
        "START": ("não deixe", "não mantenha", "parado", "parada"),
        "STOP": ("não deixe", "não mantenha", "rodando"),
    }
    if intent in negative_state and _contains_lexeme(low, "não") and any(_contains_lexeme(low, x) for x in negative_state[intent][1:]):
        return True
    sig = INTENT_SIGNATURES.get(intent)
    if not sig:
        return True

    # Estados com objeto no meio: "deixa a luz ligada", "deixe a porta aberta".
    state_patterns = {
        "TURN_ON": (r"\bdeixa(?:r)?\s+.+\s+(?:ligado|ligada|aceso|acesa)\b",),
        "TURN_OFF": (r"\bdeixa(?:r)?\s+.+\s+(?:desligado|desligada|apagado|apagada)\b",),
        "OPEN": (r"\bdeixa(?:r)?\s+.+\s+(?:aberto|aberta|destravado|destravada)\b",),
        "CLOSE": (r"\bdeixa(?:r)?\s+.+\s+(?:fechado|fechada|travado|travada)\b",),
    }
    positive = any(_contains_lexeme(low, x) for x in sig["positive"])
    if not positive and any(re.search(p, low) for p in state_patterns.get(intent, ())):
        positive = True
    if intent in {"SET_TEMPERATURE", "SET_SPEED", "SET_BRIGHTNESS",
                  "SET_VOLUME", "SET_VOLTAGE", "SET_COLOR"}:
        return positive
    if not positive:
        return False
    if any(_contains_lexeme(low, x) for x in sig["neighbors"]):
        return False
    return True

_BAD_PATTERNS = [
    re.compile(p, re.I) for p in [
        r"\bde o\b", r"\bde a\b", r"\bem o\b", r"\bem a\b",
        r"\bdo o\b", r"\bda a\b", r"\bno o\b", r"\bna a\b",
        r"\b(o|a)\s+(o|a)\b", r"\b(da|do|de|em|no|na|para|com)\s+\1\b",
        r"\s+[,.!?;:]", r"[,.!?;:]{2,}",
        r"\b(\w+)\s+\1\b",
        r"\b(\w+)\s+(da|do|de|na|no|em)\s+\1\b",
        r"\b(\w+)\s+(da|do|de)\s+(\w+)\s+(da|do|de)\s+\3\b"
    ]
]

# ======================== FUNÇÕES AUXILIARES ========================

def participle_for(obj_key: str, intent: str) -> str:
    participles = PARTICIPLE[intent]
    if len(participles) >= 2:
        return participles[1] if OBJECTS[obj_key]["genero"] == "f" else participles[0]
    else:
        return participles[0]

def opposite_participle_for(obj_key: str, intent: str) -> str:
    if intent not in OPPOSITE_PARTICIPLE:
        return "oposto"
    opp = OPPOSITE_PARTICIPLE[intent]
    if len(opp) >= 2:
        return opp[1] if OBJECTS[obj_key]["genero"] == "f" else opp[0]
    else:
        return opp[0]

def validar_frase(texto: str, obj_key: Optional[str] = None, loc_key: Optional[str] = None) -> List[str]:
    erros = []
    texto_min = texto.lower()
    if loc_key:
        regras_local_errado = {
            "banheiro": ["geladeira", "forno", "fogão", "micro-ondas"],
            "área de serviço": ["cama", "televisão", "forno"],
            "quarto": ["chuveiro", "torneira do quintal", "fogão"]
        }
        for local, dispositivos in regras_local_errado.items():
            if local in loc_key and any(disp in texto_min for disp in dispositivos):
                erros.append(f"Local incoerente: '{obj_key}' em '{loc_key}'")
    acoes_incompativeis = {
        "abrir": ["luz", "ar condicionado", "televisão", "bateria"],
        "fechar": ["luz", "ar condicionado", "televisão", "bateria"],
        "ligar": ["porta", "janela", "cortina", "nível da bateria"],
        "desligar": ["porta", "janela", "cortina", "nível da bateria"]
    }
    for acao, alvos in acoes_incompativeis.items():
        if re.search(rf'\b{acao}\b', texto_min):
            for alvo in alvos:
                if alvo in texto_min:
                    erros.append(f"Ação incoerente: '{acao}' aplicado a '{alvo}'")
    if obj_key and "bateria" in obj_key.lower():
        padrao_volts = r'(\d+)\s*[vV]\b'
        match_volts = re.search(padrao_volts, texto)
        if match_volts:
            tensao = int(match_volts.group(1))
            if tensao > 24:
                erros.append(f"Valor irreal: bateria com {tensao}V")
    return erros

# ======================== GERADOR SINTÁTICO GENERATIVO V63.1 ========================

# V63.11: assinaturas de estrutura/diversidade para balanceamento hierárquico.
def _v6311_norm(s):
    return re.sub(r"\s+", " ", str(s).strip().lower())

def _v6311_get(sample, *keys, default=None):
    if isinstance(sample, dict):
        for k in keys:
            if k in sample:
                return sample[k]
    return default

def _v6311_structural_signature(sample):
    if isinstance(sample, dict):
        for k in ("structure", "template", "template_id", "pattern"):
            if sample.get(k) is not None:
                return _v6311_norm(sample[k])
        text_value = _v6311_get(sample, "text", "sentence", "utterance", default="")
    else:
        text_value = str(sample)
    s = _v6311_norm(text_value)
    s = re.sub(r"\b\d+(?:[.,]\d+)?\b", "<NUM>", s)
    # Normaliza marcadores temporais comuns sem apagar o restante da sintaxe.
    s = re.sub(r"\b(?:hoje|amanhã|ontem|agora|depois|breve|semanalmente|diariamente|mensalmente)\b", "<TEMP>", s)
    s = re.sub(r"\b(?:às?|as)\s+\d{1,2}(?::\d{2})?(?:\s*horas?)?\b", "<TEMP>", s)
    return s

def _v6311_diversity_key(sample):
    return (
        _v6311_structural_signature(sample),
        _v6311_norm(_v6311_get(sample, "device", "DEVICE", default="")),
        _v6311_norm(_v6311_get(sample, "location", "LOCATION", default="")),
        _v6311_norm(_v6311_get(sample, "value", "VALUE", "measure", "MEASURE", default="")),
        _v6311_norm(_v6311_get(sample, "temporal", "temporal_type", "TS", default="")),
        _v6311_norm(_v6311_get(sample, "operation", "op", "OP", default="")),
    )

class SyntacticGenerator:
    def __init__(self, rng: random.Random):
        self.rng = rng
        self.adverbios_tempo = ["agora", "já", "imediatamente", "neste instante",
                                "agora mesmo", "o mais rápido possível",
                                "por enquanto"]
        self.adverbios_modo = ["bem", "devagar", "rapidamente", "cuidadosamente", "sem pressa",
                               "no máximo", "com cuidado", "direitinho", "de leve", "mais rápido",
                               "sem enrolar", "do jeito certo", "bastante", "um pouco", "ligeiramente"]
        self.discourse_openers = ["por favor", "por gentileza", "se puder", "quando puder", "preciso que você",
                                  "pode", "consegue", "dá para", "tem como", "será que dá para"]

    def escolher(self, lista: List[Any]) -> Any:
        return self.rng.choice(lista)

    def talvez(self, prob: float = 0.5) -> bool:
        return self.rng.random() < prob

    def prep_de(self, loc_key: Optional[str]) -> str:
        if loc_key is None: return ""
        return "da" if LOCATIONS[loc_key] == "f" else "do"

    def frase_nominal(self, obj_key: str, loc_key: Optional[str],
                      com_artigo: bool = True, possessivo: bool = False,
                      surface_obj: Optional[str] = None,
                      surface_loc: Optional[str] = None) -> str:
        genero = OBJECTS[obj_key]["genero"]
        art = ("a" if genero == "f" else "o") if com_artigo else ""
        nome = surface_obj or obj_key
        if loc_key:
            loc_str = surface_loc or loc_key
            if possessivo and not surface_loc:
                pref = "minha" if LOCATIONS[loc_key] == "f" else "meu"
                loc_str = f"{pref} {loc_key}"
            prep = "da" if LOCATIONS[loc_key] == "f" else "do"
            return f"{art} {nome} {prep} {loc_str}".strip() if com_artigo else f"{nome} {prep} {loc_str}"
        return f"{art} {nome}".strip() if com_artigo else nome

    def atributo(self, obj_key: str, loc_key: Optional[str],
                 surface_obj: Optional[str] = None, surface_loc: Optional[str] = None) -> str:
        attr = OBJECTS[obj_key]["atributo"]
        options = ATTR_SYNONYMS[attr]
        noun, gender = self.escolher(options)
        if noun.lower() == obj_key.lower():
            noun, gender = "estado", "m"
        art = "a" if gender == "f" else "o"
        nome = surface_obj or obj_key
        if loc_key:
            prep_loc = self.prep_de(loc_key)
            prep_obj = "da" if OBJECTS[obj_key]["genero"] == "f" else "do"
            return f"{art} {noun} {prep_obj} {nome} {prep_loc} {surface_loc or loc_key}"
        else:
            prep = "da" if OBJECTS[obj_key]["genero"] == "f" else "do"
            return f"{art} {noun} {prep} {nome}"

    def artigo_atributo(self, obj_key: str) -> str:
        genero = OBJECTS[obj_key]["genero"]
        return "a" if genero == "f" else "o"

    def atributo_intent(self, obj_key: str, loc_key: Optional[str],
                        intent: Optional[str] = None,
                        surface_obj: Optional[str] = None,
                        surface_loc: Optional[str] = None) -> str:
        attr = OBJECTS[obj_key]["atributo"]
        options = {
            "brilho": [("brilho", "m"), ("intensidade", "f"), ("luminosidade", "f"), ("claridade", "f")],
            "velocidade": [("velocidade", "f"), ("rotação", "f"), ("ritmo", "m")],
            "temperatura": [("temperatura", "f"), ("grau de temperatura", "m")],
            "voltagem": [("voltagem", "f"), ("tensão", "f")],
            "volume": [("volume", "m"), ("som", "m"), ("áudio", "m")],
        }.get(attr, ATTR_SYNONYMS.get(attr, [("estado", "m")]))
        noun, gender = self.escolher(options)
        art = "a" if gender == "f" else "o"
        nome = surface_obj or obj_key
        if loc_key:
            prep_obj = "da" if OBJECTS[obj_key]["genero"] == "f" else "do"
            prep_loc = self.prep_de(loc_key)
            return f"{art} {noun} {prep_obj} {nome} {prep_loc} {surface_loc or loc_key}"
        prep = "da" if OBJECTS[obj_key]["genero"] == "f" else "do"
        return f"{art} {noun} {prep} {nome}"

    def valor_str(self, obj_key: str) -> str:
        attr = OBJECTS[obj_key]["atributo"]
        if attr == "temperatura":
            v = self.rng.randint(16, 30)
            return self.rng.choice([f"{v}°C", f"{v} graus", f"{v} graus Celsius"])
        elif attr == "voltagem":
            # Limites plausíveis por classe de equipamento. Isso evita, por
            # exemplo, gerar 110/220 V para uma bateria de 12/24 V.
            low = obj_key.casefold()
            if low in {"bateria", "banco de baterias"}:
                vals = [5, 9, 12, 24]
            elif low == "carregador":
                vals = [5, 9, 12, 20, 24]
            elif low in {"nobreak", "inversor", "estabilizador", "fonte de bancada", "gerador elétrico", "painel solar"}:
                vals = [12, 24, 48, 110, 127, 220]
            else:
                vals = [5, 9, 12, 24, 48, 110, 127, 220]
            v = self.rng.choice(vals)
            return self.rng.choice([f"{v}V", f"{v} volts", f"{v} Volts"])
        else:
            v = self.rng.randint(1, 99)
            return self.rng.choice([f"{v}%", f"{v} por cento"])

    def cor_concordada(self, obj_key: str) -> str:
        color = self.escolher(COLORS)
        if OBJECTS[obj_key]["genero"] == "f":
            return COLOR_FEM[color]
        return color

    # ========== V63.1: TEMPLATES EXCLUSIVOS POR INTENÇÃO ==========

    def gerar_acao_contextual(self, intent: str, obj_key: str, obj_nom: str) -> Optional[str]:
        fem = OBJECTS[obj_key]["genero"] == "f"
        ligado = "ligada" if fem else "ligado"
        desligado = "desligada" if fem else "desligado"
        aberto = "aberta" if fem else "aberto"
        fechado = "fechada" if fem else "fechado"
        aceso = "acesa" if fem else "aceso"
        banks = {
            "TURN_ON": [
                f"liga {obj_nom}.",
                f"acende {obj_nom}." if obj_key in {"luz", "lâmpada", "luminária", "iluminação", "abajur", "led"} else f"liga {obj_nom}.",
                f"deixa {obj_nom} {ligado}.",
                f"quero {obj_nom} {ligado}.",
                f"quero deixar {obj_nom} {ligado}.",
                f"pode ligar {obj_nom}.",
                f"energiza {obj_nom}.",
                f"ativa {obj_nom}.",
                f"quero que {obj_nom} fique {ligado}.",
                f"deixa {obj_nom} {aceso}.",
                f"quero {obj_nom} {aceso}.",
                f"preciso que {obj_nom} fique {ligado}.",
                f"vou ligar {obj_nom}.",
            ],
            "TURN_OFF": [
                f"desliga {obj_nom}.",
                f"deixa {obj_nom} {desligado}.",
                f"quero {obj_nom} {desligado}.",
                f"quero deixar {obj_nom} {desligado}.",
                f"tira {obj_nom} de funcionamento.",
                f"corta a alimentação de {obj_nom}.",
                f"remove a energia de {obj_nom}.",
                f"pode apagar {obj_nom}.",
                f"quero que {obj_nom} fique {desligado}.",
                f"não quero mais {obj_nom} {ligado}.",
                f"não quero {obj_nom} {aceso}.",
                f"apaga {obj_nom}.",
                f"desconecta {obj_nom}.",
                f"desativa {obj_nom}.",
                f"desliga {obj_nom} imediatamente.",
                f"tira {obj_nom} da tomada.",
                f"não mantenha {obj_nom} {aceso}.",
                f"tira {obj_nom} da tomada.",
                f"não quero mais {obj_nom} {ligado}.",
            ],
            "OPEN": [
                f"abre {obj_nom}.",
                f"deixa {obj_nom} {aberto}.",
                f"quero {obj_nom} {aberto}.",
                f"destrava {obj_nom}.",
                f"destrave {obj_nom}.",
                f"libera {obj_nom}.",
                f"escancara {obj_nom}.",
                f"quero que {obj_nom} fique {aberto}.",
                f"pode deixar {obj_nom} {aberto}.",
                f"desbloqueia {obj_nom}.",
                f"abre completamente {obj_nom}.",
                f"destranca a {obj_nom}.",
                f"abre a {obj_nom} agora.",
            ],
            "CLOSE": [
                f"fecha {obj_nom}.",
                f"deixa {obj_nom} {fechado}.",
                f"quero {obj_nom} {fechado}.",
                f"trava {obj_nom}.",
                f"tranque {obj_nom}.",
                f"bloqueia {obj_nom}.",
                f"sela {obj_nom}.",
                f"quero que {obj_nom} fique {fechado}.",
                f"pode deixar {obj_nom} {fechado}.",
                f"não quero {obj_nom} {aberto}.",
                f"fecha bem {obj_nom}.",
                f"tranca {obj_nom} agora.",
                f"fecha a {obj_nom}.",
            ],
            "START": [
                f"inicia {obj_nom}.",
                f"começa {obj_nom}.",
                f"põe {obj_nom} para rodar.",
                f"manda {obj_nom} rodar.",
                f"faz {obj_nom} começar.",
                f"dá partida em {obj_nom}.",
                f"começa a operação de {obj_nom}.",
                f"pode iniciar {obj_nom}.",
                f"dispara {obj_nom}.",
                f"executa {obj_nom}.",
                f"roda {obj_nom}.",
            ],
            "STOP": [
                f"para {obj_nom}.",
                f"interrompe {obj_nom}.",
                f"pausa {obj_nom}.",
                f"faz {obj_nom} parar.",
                f"manda {obj_nom} parar.",
                f"cessa o funcionamento de {obj_nom}.",
                f"interrompe a operação de {obj_nom}.",
                f"para de rodar {obj_nom}.",
                f"pode parar {obj_nom}.",
                f"não deixe {obj_nom} rodando.",
                f"aborta {obj_nom}.",
                f"cancela a execução de {obj_nom}.",
                f"interrompa {obj_nom} agora.",
                f"pare imediatamente {obj_nom}.",
            ],
        }
        choices = banks.get(intent)
        return self.escolher(choices) if choices else None

    def gerar_temporal(self) -> Tuple[str, str, str]:
        kind = self.escolher(["time", "date", "relative", "recurrence"])
        if kind == "time":
            h = self.rng.randint(1, 23)
            m = self.rng.choice([0, 15, 30, 45])
            if self.talvez(0.5):
                v = f"às {h:02d}:{m:02d}"
                return v, "TIME", v.split("às ", 1)[1]
            h12 = h if h <= 12 else h - 12
            period = "da manhã" if h <= 11 else ("da tarde" if h <= 18 else "da noite")
            if m == 30:
                prep_hora = "à" if h12 == 1 else "às"
                v = f"{prep_hora} {_NUM_WORDS.get(h12, str(h12))} e meia {period}"
                return v, "TIME", v
            v = f"às {_NUM_WORDS.get(h12, str(h12))} {'hora' if h12 == 1 else 'horas'} {period}"
            return v, "TIME", v
        if kind == "date":
            v = self.escolher(["amanhã", "depois de amanhã", "na segunda-feira", "na terça-feira",
                               "na quarta-feira", "na quinta-feira", "na sexta-feira", "no sábado",
                               "no domingo", "no dia 5 de novembro", "no dia 15 de setembro",
                               "na próxima semana"])
            return v, "DATE", v
        if kind == "relative":
            # ===== ALTERAÇÃO: mais diversidade de RELATIVE_TIME =====
            units = ["segundo", "minuto", "hora"]
            nums = [5, 10, 15, 20, 30, 45, 60]
            unit = self.escolher(units)
            num = self.escolher(nums)
            variants = [
                f"daqui a {num} {unit}{'s' if num > 1 else ''}",
                f"em {num} {unit}{'s' if num > 1 else ''}",
                f"daqui a {_NUM_WORDS.get(num, str(num))} {unit}{'s' if num > 1 else ''}",
                f"dentro de {num} {unit}{'s' if num > 1 else ''}",
                # Somente expressões inequivocamente temporais.
                # Evitamos "mais tarde", "logo mais" e "em breve" porque
                # "mais" aparece em comandos de aumento de brilho/volume/etc.
            ]
            v = self.escolher(variants)
            return v, "RELATIVE_TIME", v
        # recurrence
        v = self.escolher(["todos os dias", "toda semana", "todos os dias úteis",
                           "toda segunda-feira", "todo sábado", "todo domingo",
                           "de segunda a sexta", "diariamente", "semanalmente",
                           "toda noite", "toda manhã", "todos os finais de semana"])
        return v, "RECURRENCE", v

    def adicionar_temporal(self, frase: str, intent: Optional[str] = None, obj_nom: Optional[str] = None, obj_key: Optional[str] = None) -> Tuple[str, str, str]:
        temp, temp_type, temp_val = self.gerar_temporal()
        if intent in ACTION_INTENTS and obj_nom and self.talvez(0.25):
            vf = VERB_FORMS[intent]
            inf = vf["inf"]
            explicit = self.escolher([
                f"Programe {obj_nom} para {inf} {temp}.",
                f"Agende {obj_nom} para {inf} {temp}.",
                f"Quero programar {obj_nom} para {inf} {temp}.",
                f"Deixe {obj_nom} {'programada' if obj_key and OBJECTS[obj_key]['genero']=='f' else 'programado'} para {inf} {temp}.",
            ])
            return explicit, temp_type, temp_val
        for imediato in (" agora mesmo", " imediatamente", " neste instante", " agora", " já"):
            frase = re.sub(re.escape(imediato) + r"(?=[?.!]?$)", "", frase, flags=re.I).strip()
        if frase.endswith("?"):
            frase = frase[:-1].rstrip() + f" {temp}?"
        elif frase.endswith("."):
            frase = frase[:-1].rstrip() + f" {temp}."
        else:
            frase = frase.rstrip() + f" {temp}."
        return frase, temp_type, temp_val

    def gerar_acao(self, intent: str, obj_key: str, loc_key: Optional[str],
                   surface_obj: Optional[str] = None, surface_loc: Optional[str] = None) -> Tuple[str, List[Tuple[str, str]]]:
        vf = VERB_FORMS[intent]
        inf, imp, fin = vf["inf"], self.escolher(vf["imp"]), self.escolher(vf["fin"])
        subj = self.escolher(vf.get("subj", [fin]))
        obj_nom = self.frase_nominal(obj_key, loc_key, com_artigo=True,
                                     surface_obj=surface_obj, surface_loc=surface_loc)

        if self.talvez(0.60):
            contextual = self.gerar_acao_contextual(intent, obj_key, obj_nom)
            if contextual:
                return contextual, [("DEVICE", surface_obj or obj_key)] + ([("LOCATION", surface_loc or loc_key)] if loc_key else [])

        if intent == "TURN_ON":
            style_pool = ["imperative", "declarative", "polite", "colloquial"]
        elif intent == "TURN_OFF":
            style_pool = ["imperative", "declarative", "polite", "colloquial", "negative"]
        elif intent == "OPEN":
            style_pool = ["imperative", "declarative", "polite", "colloquial"]
        elif intent == "CLOSE":
            style_pool = ["imperative", "declarative", "polite", "colloquial", "negative"]
        elif intent == "START":
            style_pool = ["imperative", "declarative", "polite", "colloquial"]
        elif intent == "STOP":
            style_pool = ["imperative", "declarative", "polite", "colloquial", "negative"]
        else:
            style_pool = ["imperative", "declarative", "polite", "colloquial"]

        style = self.escolher(style_pool)
        t = self.escolher(self.adverbios_tempo) if self.talvez(0.25) else ""
        opener = self.escolher(self.discourse_openers) if self.talvez(0.3) else ""

        if style == "imperative":
            if intent == "TURN_ON":
                frase = self.escolher([
                    f"{imp.capitalize()} {obj_nom}.",
                    f"{imp.capitalize()} {obj_nom} agora, por favor.",
                    f"{imp.capitalize()} {obj_nom}, por favor.",
                    f"{imp.capitalize()} {obj_nom} imediatamente, por favor.",
                    f"Vamos, {imp} {obj_nom}.",
                    f"Ative {obj_nom}.",
                    f"Acione {obj_nom}.",
                    f"Deixe {obj_nom} ligado.",
                ])
            elif intent == "TURN_OFF":
                frase = self.escolher([
                    f"{imp.capitalize()} {obj_nom}.",
                    f"{imp.capitalize()} {obj_nom} agora, por favor.",
                    f"{imp.capitalize()} {obj_nom}, por favor.",
                    f"{imp.capitalize()} {obj_nom} imediatamente, por favor.",
                    f"Apague {obj_nom}.",
                    f"Desative {obj_nom}.",
                    f"Tire {obj_nom} da tomada.",
                    f"Tira {obj_nom}.",
                    f"Desliga {obj_nom}.",
                ])
            elif intent == "OPEN":
                frase = self.escolher([
                    f"{imp.capitalize()} {obj_nom}.",
                    f"{imp.capitalize()} {obj_nom} agora, por favor.",
                    f"{imp.capitalize()} {obj_nom}, por favor.",
                    f"Destrave {obj_nom}.",
                    f"Libere {obj_nom}.",
                    f"Desbloqueie {obj_nom}.",
                    f"Abra {obj_nom}.",
                ])
            elif intent == "CLOSE":
                frase = self.escolher([
                    f"{imp.capitalize()} {obj_nom}.",
                    f"{imp.capitalize()} {obj_nom} agora, por favor.",
                    f"{imp.capitalize()} {obj_nom}, por favor.",
                    f"Trave {obj_nom}.",
                    f"Bloqueie {obj_nom}.",
                    f"Tranque {obj_nom}.",
                    f"Feche {obj_nom}.",
                ])
            elif intent == "START":
                frase = self.escolher([
                    f"{imp.capitalize()} {obj_nom}.",
                    f"{imp.capitalize()} {obj_nom} agora, por favor.",
                    f"{imp.capitalize()} {obj_nom}, por favor.",
                    f"Dê partida em {obj_nom}.",
                    f"Execute {obj_nom}.",
                    f"Dispare {obj_nom}.",
                    f"Inicie {obj_nom}.",
                ])
            elif intent == "STOP":
                frase = self.escolher([
                    f"{imp.capitalize()} {obj_nom}.",
                    f"{imp.capitalize()} {obj_nom} agora, por favor.",
                    f"{imp.capitalize()} {obj_nom}, por favor.",
                    f"Pare {obj_nom} imediatamente.",
                    f"Interrompa {obj_nom}.",
                    f"Cesse {obj_nom}.",
                    f"Aborte {obj_nom}.",
                ])
            else:
                frase = f"{imp.capitalize()} {obj_nom}."
        elif style == "declarative":
            if intent == "TURN_ON":
                frase = self.escolher([
                    f"Eu quero {inf} {obj_nom} agora.",
                    f"Eu quero {inf} {obj_nom}.",
                    f"Eu preciso {inf} {obj_nom} agora.",
                    f"Quero que {obj_nom} fique ligado.",
                    f"É pra {inf} {obj_nom}.",
                    f"Vou {inf} {obj_nom}.",
                    f"Preciso que {obj_nom} seja {participle_for(obj_key, intent)}.",
                ])
            elif intent == "TURN_OFF":
                frase = self.escolher([
                    f"Eu quero {inf} {obj_nom} agora.",
                    f"Eu quero {inf} {obj_nom}.",
                    f"Eu preciso {inf} {obj_nom} agora.",
                    f"Quero que {obj_nom} fique desligado.",
                    f"Não quero mais {obj_nom} ligado.",
                    f"Vou {inf} {obj_nom}.",
                    f"Já não quero mais {obj_nom} aceso.",
                ])
            elif intent == "OPEN":
                frase = self.escolher([
                    f"Eu quero {inf} {obj_nom} agora.",
                    f"Eu quero {inf} {obj_nom}.",
                    f"Eu preciso {inf} {obj_nom} agora.",
                    f"Quero que {obj_nom} fique aberto.",
                    f"Vou {inf} {obj_nom}.",
                ])
            elif intent == "CLOSE":
                frase = self.escolher([
                    f"Eu quero {inf} {obj_nom} agora.",
                    f"Eu quero {inf} {obj_nom}.",
                    f"Eu preciso {inf} {obj_nom} agora.",
                    f"Quero que {obj_nom} fique fechado.",
                    f"Vou {inf} {obj_nom}.",
                ])
            elif intent == "START":
                frase = self.escolher([
                    f"Eu quero {inf} {obj_nom} agora.",
                    f"Eu quero {inf} {obj_nom}.",
                    f"Eu preciso {inf} {obj_nom} agora.",
                    f"Quero que {obj_nom} comece.",
                    f"Vou {inf} {obj_nom}.",
                ])
            elif intent == "STOP":
                frase = self.escolher([
                    f"Eu quero {inf} {obj_nom} agora.",
                    f"Eu quero {inf} {obj_nom}.",
                    f"Eu preciso {inf} {obj_nom} agora.",
                    f"Quero que {obj_nom} pare.",
                    f"Vou {inf} {obj_nom}.",
                ])
            else:
                frase = f"Eu quero {inf} {obj_nom} agora."
        elif style == "polite":
            frase = self.escolher([
                f"Por favor, {inf} {obj_nom}.",
                f"Por gentileza, {inf} {obj_nom}.",
                f"Gostaria que você {subj} {obj_nom}.",
                f"Eu gostaria que você {subj} {obj_nom}.",
                f"Você poderia {inf} {obj_nom}, por favor?",
                f"Se puder, {inf} {obj_nom}.",
                f"Poderia {inf} {obj_nom}?",
            ])
        elif style == "colloquial":
            if intent == "TURN_ON":
                frase = self.escolher([
                    f"{fin.capitalize()} {obj_nom} aí.",
                    f"{fin.capitalize()} {obj_nom} pra mim.",
                    f"Deixa {obj_nom} ligado aí.",
                    f"Liga {obj_nom} aí.",
                    f"Acende {obj_nom} aí.",
                ])
            elif intent == "TURN_OFF":
                frase = self.escolher([
                    f"{fin.capitalize()} {obj_nom} aí.",
                    f"{fin.capitalize()} {obj_nom} pra mim.",
                    f"Tira {obj_nom} aí.",
                    f"Apaga {obj_nom} aí.",
                    f"Desliga {obj_nom} aí.",
                    f"Tira {obj_nom} da tomada aí.",
                    f"Não quero mais {obj_nom} ligado aí.",
                ])
            elif intent == "OPEN":
                frase = self.escolher([
                    f"{fin.capitalize()} {obj_nom} aí.",
                    f"{fin.capitalize()} {obj_nom} pra mim.",
                    f"Abre {obj_nom} aí.",
                    f"Destrava {obj_nom} aí.",
                ])
            elif intent == "CLOSE":
                frase = self.escolher([
                    f"{fin.capitalize()} {obj_nom} aí.",
                    f"{fin.capitalize()} {obj_nom} pra mim.",
                    f"Fecha {obj_nom} aí.",
                    f"Trava {obj_nom} aí.",
                ])
            elif intent == "START":
                frase = self.escolher([
                    f"{fin.capitalize()} {obj_nom} aí.",
                    f"{fin.capitalize()} {obj_nom} pra mim.",
                    f"Começa {obj_nom} aí.",
                    f"Põe {obj_nom} pra rodar aí.",
                ])
            elif intent == "STOP":
                frase = self.escolher([
                    f"{fin.capitalize()} {obj_nom} aí.",
                    f"{fin.capitalize()} {obj_nom} pra mim.",
                    f"Para {obj_nom} aí.",
                    f"Interrompe {obj_nom} aí.",
                ])
            else:
                frase = f"{fin.capitalize()} {obj_nom} aí."
        elif style == "negative":
            if intent in ("TURN_OFF", "CLOSE", "STOP"):
                oposto = opposite_participle_for(obj_key, intent)
                frase = self.escolher([
                    f"Não quero {obj_nom} {oposto}.",
                    f"Não deixe {obj_nom} {oposto}.",
                    f"Não mantenha {obj_nom} {oposto}.",
                    f"Não precisa {inf} {obj_nom}.",
                    f"Não {inf} {obj_nom}.",
                    f"Não deixe {obj_nom} {oposto} nunca.",
                ])
            else:
                frase = self.escolher([
                    f"Não quero {obj_nom} parado.",
                    f"Não deixe {obj_nom} parado.",
                ])
        else:  # conditional
            frase = self.escolher([
                f"Seria bom {inf} {obj_nom}.",
                f"Queria que você {subj} {obj_nom}.",
                f"Se puder, {inf} {obj_nom}.",
                f"Quando der, {inf} {obj_nom}.",
                f"Caso possa, {inf} {obj_nom}.",
            ])

        if t and style in ("imperative", "declarative", "polite", "colloquial"):
            frase = frase[:-1] + f" {t}."

        specs = [("DEVICE", surface_obj or obj_key)]
        if loc_key:
            specs.append(("LOCATION", surface_loc or loc_key))
        return frase, specs

    def gerar_status(self, obj_key: str, loc_key: Optional[str], surface_obj: Optional[str] = None, surface_loc: Optional[str] = None) -> Tuple[str, List[Tuple[str, str]]]:
        obj_nom = self.frase_nominal(obj_key, loc_key, com_artigo=True, surface_obj=surface_obj, surface_loc=surface_loc)
        attr = OBJECTS[obj_key]["atributo"]

        status_specific = {
            "temperatura": [
                f"qual é a temperatura de {obj_nom}?",
                f"qual a temperatura atual de {obj_nom}?",
                f"me diga a temperatura de {obj_nom}.",
                f"quanto está a temperatura de {obj_nom}?",
                f"{obj_nom} está a quantos graus?",
                f"consulte a temperatura de {obj_nom}.",
                f"verifique a temperatura de {obj_nom}.",
                f"qual a temperatura de {obj_nom} agora?",
                f"qual a temperatura atual de {obj_nom}?",
            ],
            "brilho": [
                f"qual é o brilho de {obj_nom}?",
                f"como está o brilho de {obj_nom}?",
                f"me diga o nível de brilho de {obj_nom}.",
                f"qual a intensidade de {obj_nom}?",
                f"consulte o brilho de {obj_nom}.",
                f"verifique o brilho de {obj_nom}.",
                f"qual a intensidade da luz de {obj_nom}?",
            ],
            "volume": [
                f"qual é o volume de {obj_nom}?",
                f"como está o volume de {obj_nom}?",
                f"qual o nível do som de {obj_nom}?",
                f"consulte o volume de {obj_nom}.",
                f"verifique o volume de {obj_nom}.",
                f"qual o volume de {obj_nom}?",
            ],
            "velocidade": [
                f"qual é a velocidade de {obj_nom}?",
                f"como está a velocidade de {obj_nom}?",
                f"a que velocidade está {obj_nom}?",
                f"consulte a velocidade de {obj_nom}.",
                f"verifique a velocidade de {obj_nom}.",
                f"qual a velocidade de {obj_nom}?",
            ],
            "estado": [
                f"qual é o estado de {obj_nom}?",
                f"como está {obj_nom}?",
                f"me diga se {obj_nom} está funcionando.",
                f"consulte o status de {obj_nom}.",
                f"verifique o status de {obj_nom}.",
                f"qual o status de {obj_nom}?",
                f"qual o estado de {obj_nom}?",
            ]
        }

        if attr in status_specific and self.talvez(0.85):
            frase = self.escolher(status_specific[attr])
        else:
            frase = self.escolher([
                f"Como está {obj_nom}?",
                f"Qual o status de {obj_nom}?",
                f"Me diga o estado de {obj_nom}.",
                f"Verifique {obj_nom}.",
                f"Consulte {obj_nom}.",
                f"Você pode me dizer como está {obj_nom}?",
                f"Poderia verificar {obj_nom}?",
                f"O que está acontecendo com {obj_nom}?",
                f"Qual a situação de {obj_nom}?",
                f"Qual o estado atual de {obj_nom}?",
            ])

        specs = [("DEVICE", obj_key)]
        if loc_key:
            specs.append(("LOCATION", loc_key))
        return frase, specs

    def gerar_cor(self, obj_key: str, loc_key: Optional[str], surface_obj: Optional[str] = None, surface_loc: Optional[str] = None) -> Tuple[str, List[Tuple[str, str]]]:
        obj_nom = self.frase_nominal(obj_key, loc_key, com_artigo=True, surface_obj=surface_obj, surface_loc=surface_loc)
        cor = self.escolher(COLORS)
        if cor in COLOR_FEM and OBJECTS[obj_key]["genero"] == "f":
            cor = COLOR_FEM[cor]

        color_frames = [
            f"mude {obj_nom} para {cor}.",
            f"coloque {obj_nom} em {cor}.",
            f"deixe {obj_nom} {cor}.",
            f"deixa {obj_nom} na cor {cor}.",
            f"quero {obj_nom} em {cor}.",
            f"quero que {obj_nom} fique {cor}.",
            f"pode colocar {obj_nom} em {cor}?",
            f"tem como deixar {obj_nom} {cor}?",
            f"bota {obj_nom} na cor {cor} aí.",
            f"muda a cor de {obj_nom} para {cor}.",
            f"altera a cor de {obj_nom} para {cor}.",
            f"define a cor de {obj_nom} como {cor}.",
            f"mude a coloração de {obj_nom} para {cor}.",
        ]
        frase = self.escolher(color_frames)
        specs = [("DEVICE", obj_key), ("COLOR", cor)]
        if loc_key:
            specs.append(("LOCATION", loc_key))
        return frase, specs

    def gerar_numerico(self, intent: str, obj_key: str, loc_key: Optional[str], operacao: str, surface_obj: Optional[str] = None, surface_loc: Optional[str] = None) -> Tuple[str, List[Tuple[str, str]]]:
        attr_phrase = self.atributo_intent(obj_key, loc_key, intent, surface_obj=surface_obj, surface_loc=surface_loc)
        valor = self.valor_str(obj_key)
        adv_modo = self.escolher(self.adverbios_modo) if self.talvez(0.3) else ""
        attr = OBJECTS[obj_key]["atributo"]

        if operacao == "SET":
            if intent == "SET_TEMPERATURE":
                frase = self.escolher([
                    f"coloque {attr_phrase} em {valor}.",
                    f"deixe {attr_phrase} em {valor}.",
                    f"ajuste {attr_phrase} para {valor}.",
                    f"configure {attr_phrase} em {valor}.",
                    f"defina {attr_phrase} em {valor}.",
                    f"pode deixar {attr_phrase} em {valor}?",
                    f"quero {attr_phrase} em {valor}.",
                    f"mude a temperatura para {valor}.",
                    f"regule a temperatura para {valor}.",
                    f"ajuste o termostato para {valor}.",
                    f"coloque a temperatura em {valor}.",
                ])
            elif intent == "SET_VOLTAGE":
                frase = self.escolher([
                    f"coloque {attr_phrase} em {valor}.",
                    f"deixe {attr_phrase} em {valor}.",
                    f"ajuste {attr_phrase} para {valor}.",
                    f"configure {attr_phrase} em {valor}.",
                    f"põe {attr_phrase} em {valor}.",
                    f"pode colocar {attr_phrase} em {valor}?",
                    f"quero {attr_phrase} em {valor}.",
                    f"defina {attr_phrase} para {valor}.",
                    f"regule a tensão para {valor}.",
                ])
            else:
                frase = self.escolher([
                    f"coloque {attr_phrase} em {valor}.",
                    f"deixe {attr_phrase} em {valor}.",
                    f"ajuste {attr_phrase} para {valor}.",
                    f"configure {attr_phrase} em {valor}.",
                    f"põe {attr_phrase} em {valor}.",
                    f"pode colocar {attr_phrase} em {valor}?",
                    f"quero {attr_phrase} em {valor}.",
                    f"quero deixar {attr_phrase} em {valor}.",
                    f"bota {attr_phrase} em {valor} aí.",
                    f"defina {attr_phrase} para {valor}.",
                ])
            specs = [("DEVICE", obj_key), ("MEASURE", valor)]
        elif operacao == "INCREASE":
            verbos = {
                "SET_TEMPERATURE": ("aumente", "suba", "eleve", "esquente"),
                "SET_SPEED": ("aumente", "acelere", "suba"),
                "SET_BRIGHTNESS": ("aumente", "suba", "eleve", "clareie"),
                "SET_VOLUME": ("aumente", "suba", "eleve"),
                "SET_VOLTAGE": ("aumente", "suba", "eleve"),
            }
            imp_bank = verbos.get(intent, ("aumente", "suba", "eleve"))
            verbo_imp = self.escolher(imp_bank)
            verbo_inf = {
                "aumente": "aumentar", "suba": "subir", "eleve": "elevar",
                "esquente": "esquentar", "acelere": "acelerar", "clareie": "clarear"
            }[verbo_imp]

            frase = self.escolher([
                f"{verbo_imp.capitalize()} {attr_phrase} em {valor}.",
                f"Pode {verbo_inf} {attr_phrase} em {valor}?",
                f"Quero {verbo_inf} {attr_phrase} em {valor}.",
                f"Vou {verbo_inf} {attr_phrase} em {valor}.",
                f"{verbo_imp.capitalize()} {attr_phrase} em {valor} aí.",
                f"{verbo_imp.capitalize()} um pouco {attr_phrase} em {valor}.",
                f"Preciso {verbo_inf} {attr_phrase} em {valor}.",
                f"{verbo_imp.capitalize()} {attr_phrase} em {valor}.",
            ])
            specs = [("DEVICE", obj_key), ("MEASURE", valor)]
        else:  # DECREASE
            verbos = {
                "SET_TEMPERATURE": ("diminua", "baixe", "reduza", "abaixe", "esfrie"),
                "SET_SPEED": ("diminua", "baixe", "reduza", "desacelere"),
                "SET_BRIGHTNESS": ("diminua", "baixe", "reduza", "abaixe", "escureça"),
                "SET_VOLUME": ("diminua", "baixe", "reduza", "abaixe"),
                "SET_VOLTAGE": ("diminua", "baixe", "reduza", "abaixe"),
            }
            imp_bank = verbos.get(intent, ("diminua", "baixe", "reduza", "abaixe"))
            verbo_imp = self.escolher(imp_bank)
            verbo_inf = {
                "diminua": "diminuir", "baixe": "baixar", "reduza": "reduzir",
                "abaixe": "abaixar", "esfrie": "esfriar", "desacelere": "desacelerar",
                "escureça": "escurecer"
            }[verbo_imp]

            frase = self.escolher([
                f"{verbo_imp.capitalize()} {attr_phrase} em {valor}.",
                f"Pode {verbo_inf} {attr_phrase} em {valor}?",
                f"Quero {verbo_inf} {attr_phrase} em {valor}.",
                f"Vou {verbo_inf} {attr_phrase} em {valor}.",
                f"{verbo_imp.capitalize()} {attr_phrase} em {valor} aí.",
                f"{verbo_imp.capitalize()} um pouco {attr_phrase} em {valor}.",
                f"Preciso {verbo_inf} {attr_phrase} em {valor}.",
                f"{verbo_imp.capitalize()} {attr_phrase} em {valor}.",
            ])
            specs = [("DEVICE", obj_key), ("MEASURE", valor)]

        if loc_key:
            specs.append(("LOCATION", loc_key))
        return frase, specs

_NUM_WORDS = {1: "uma", 2: "duas", 3: "três", 4: "quatro", 5: "cinco", 6: "seis",
              7: "sete", 8: "oito", 9: "nove", 10: "dez", 11: "onze", 12: "doze"}

# ======================== GERADOR PRINCIPAL (DatasetGenerator) ========================

@dataclass
class Config:
    samples_per_intent: int = 600
    random_seed: int = 42
    output_dir: Path = field(default_factory=lambda: Path("./dataset_v63"))
    compositions: Tuple[str, ...] = ("base", "possessive", "numbered", "combined", "demonstrative")
    contrastive_ratio: float = 0.20
    seed_per_family: int = 1
    pair_fraction: float = 0.35
    pair_rounds_per_step: int = 1
    # Temporalidade moderada para evitar correlação artificial com SCHEDULED.
    temporal_ratio: float = 0.55
    # Categorias de robustez, sem criar novas intenções.
    semantic_paraphrase_ratio: float = 0.12
    colloquial_noisy_ratio: float = 0.06
    entity_negative_ratio: float = 0.08
    # V63.22: categorias raras recebem quota própria por intenção.
    # São metas de cobertura, não condições fatais.
    semantic_anchor_ratio: float = 0.01
    explicit_schedule_ratio: float = 0.01
    # V63.14: exemplos de fronteira semântica mínima entre intenções vizinhas.
    contrastive_boundary_ratio: float = 0.18
    # Novo: estratégia de balanceamento ("equal" ou "natural")
    balance_strategy: str = "equal"
    audit_strict: bool = True

def surface_device(obj_key: str, rng: random.Random, composition: str) -> str:
    if composition in ("numbered", "combined"):
        return f"{obj_key} {rng.randint(1, 4)}"
    return obj_key

def surface_location(loc_key: Optional[str], rng: random.Random, composition: str) -> Optional[str]:
    if not loc_key:
        return None
    if composition in ("possessive", "combined"):
        prefix = "minha" if LOCATIONS[loc_key] == "f" else "meu"
        return f"{prefix} {loc_key}"
    if composition == "demonstrative":
        return f"{loc_key} aqui"
    return loc_key

def valid_combo(obj_key: str, loc_key: Optional[str]) -> bool:
    if loc_key is None:
        return True
    allowed = ALLOWED_LOCATIONS.get(obj_key)
    if allowed is not None:
        return loc_key in allowed
    return True

def combos_for(obj_key: str) -> List[Optional[str]]:
    return [None] + [loc for loc in LOCATIONS if valid_combo(obj_key, loc)]

def objects_for_intent(intent: str) -> List[str]:
    if intent == "SET_COLOR":
        return [o for o, info in OBJECTS.items() if "cor" in info["acoes"]]
    if intent in NUMERIC_INTENTS:
        attr = NUMERIC_ATTR[intent]
        return [o for o, info in OBJECTS.items() if "valor" in info["acoes"] and info["atributo"] == attr]

    act = None
    if intent in ("TURN_ON", "TURN_OFF"):
        act = "ligar"
    elif intent in ("OPEN", "CLOSE"):
        act = "abrir"
    elif intent in ("START", "STOP"):
        act = "iniciar"
    elif intent == "GET_STATUS":
        act = "status"

    if act:
        return [o for o, info in OBJECTS.items() if act in info["acoes"]]
    
    return list(OBJECTS)

def normalize(text: str) -> str:
    text = re.sub(r"\s+([?.!,])", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    contractions = [
        (r"\bde o\b", "do"), (r"\bde a\b", "da"),
        (r"\bde os\b", "dos"), (r"\bde as\b", "das"),
        (r"\bem o\b", "no"), (r"\bem a\b", "na"),
        (r"\bem os\b", "nos"), (r"\bem as\b", "nas"),
        (r"\bpara o\b", "para o"), (r"\bpara a\b", "para a"),
        (r"\bdo o\b", "do"), (r"\bda a\b", "da"),
        (r"\bno o\b", "no"), (r"\bna a\b", "na"),
    ]
    for pat, repl in contractions:
        text = re.sub(pat, repl, text, flags=re.I)
    fixes = [
        (r"\bque tal\s+aumente\b", "Que tal aumentar"),
        (r"\bque tal\s+suba\b", "Que tal subir"),
        (r"\bque tal\s+eleve\b", "Que tal elevar"),
        (r"\bque tal\s+esquente\b", "Que tal esquentar"),
        (r"\bque tal\s+acelere\b", "Que tal acelerar"),
        (r"\bque tal\s+clareie\b", "Que tal clarear"),
        (r"\bque tal\s+diminua\b", "Que tal diminuir"),
        (r"\bque tal\s+baixe\b", "Que tal baixar"),
        (r"\bque tal\s+reduza\b", "Que tal reduzir"),
        (r"\bque tal\s+abaixe\b", "Que tal abaixar"),
        (r"\bque tal\s+esfrie\b", "Que tal esfriar"),
        (r"\bque tal\s+desacelere\b", "Que tal desacelerar"),
        (r"\bque tal\s+escureça\b", "Que tal escurecer"),
        (r"\bporcento\b", "por cento"),
        (r"^Faz (.+) começar([.!?])$", r"Faça \1 iniciar\2"),
    ]
    for pat, repl in fixes:
        text = re.sub(pat, repl, text, flags=re.I)
    return text[0].upper() + text[1:] if text else ""

def find_span(text: str, value: str, used: List[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
    if not value:
        return None
    pat = r"(?<![A-Za-zÀ-ÿ0-9])" + re.escape(value) + r"(?![A-Za-zÀ-ÿ0-9])"
    for m in re.finditer(pat, text, re.I):
        s, e = m.span()
        if not any(s < be and bs < e for bs, be in used):
            return s, e
    return None

def annotate(text: str, specs: List[Tuple[str, Optional[str]]]) -> List[Dict]:
    valid_entities = {
        "DEVICE": set(OBJECTS.keys()),
        "LOCATION": set(LOCATIONS.keys()),
        "COLOR": set(COLORS + list(COLOR_FEM.values())),
        "MEASURE": set(),
        "TIME": set(),
        "RELATIVE_TIME": set(),
        "DATE": set(),
        "RECURRENCE": set(),
    }
    valid_entities["DEVICE"].update(COMPOUND_ENTITIES["DEVICE"])
    valid_entities["LOCATION"].update(COMPOUND_ENTITIES["LOCATION"])
    valid_entities["COLOR"].update(COMPOUND_ENTITIES["COLOR"])
    
    ents = []
    used: List[Tuple[int, int]] = []

    candidates = [(str(v), t) for t, v in specs if v]
    candidates.sort(key=lambda x: len(x[0]), reverse=True)

    for value, typ in candidates:
        if typ in {"DEVICE", "LOCATION", "COLOR"}:
            valid = value in valid_entities[typ]
            if not valid and typ == "DEVICE":
                base = re.sub(r"\s+[1-4]$", "", value.casefold()).strip()
                valid = base in {x.casefold() for x in valid_entities["DEVICE"]}
            if not valid and typ == "LOCATION":
                base = re.sub(r"^(minha|meu|minhas|meus)\s+", "", value.casefold()).strip()
                base = re.sub(r"\s+aqui$", "", base).strip()
                valid = base in {x.casefold() for x in valid_entities["LOCATION"]}
            if not valid and typ == "COLOR":
                valid = value.casefold() in {x.casefold() for x in COLOR_FEM.values()}
            if not valid:
                continue
        pos = find_span(text, value, used)
        if pos:
            s, e = pos
            ents.append({"start": s, "end": e, "type": typ, "value": text[s:e]})
            used.append((s, e))

    for m in re.finditer(r"\b(\d+(?:[.,]\d+)?)\s*(°C|graus|V|volts|%|por cento|porcento)(?!\w)", text, re.I):
        s, e = m.span()
        if not any(s < be and bs < e for bs, be in used):
            ents.append({"start": s, "end": e, "type": "MEASURE", "value": text[s:e]})
            used.append((s, e))

    ents.sort(key=lambda e: e["start"])
    return ents

def grammar_valid(text: str) -> bool:
    if not text or len(text.split()) < 3 or len(text) > 180:
        return False
    for pat in _BAD_PATTERNS:
        if pat.search(text):
            return False
    low = text.casefold()
    if re.search(r"\bpra ontem\b", low):
        return False
    if re.search(r"\bque tal\s+(?:aumente|suba|eleve|esquente|acelere|clareie|diminua|baixe|reduza|abaixe|esfrie|desacelere|escureça)\b", low):
        return False
    if re.search(r"\b(?:do|da|no|na)\s+(?:o|a)\b", low):
        return False
    return True

def validate_contextual_composition(text: str, entities: List[Dict]) -> bool:
    lower = text.casefold()
    for m in re.finditer(r"\bmakita\s+skil\b", lower):
        between = lower[m.start():m.end()]
        has_coord = bool(re.search(r"\b(e|ou|nem)\b", between))
        spans = [e for e in entities
                 if e["start"] <= m.start() and e["end"] >= m.end()
                 and e["type"] == "DEVICE"]
        if not has_coord and not spans:
            return False
        if not has_coord and len(spans) != 1:
            return False
    for m in re.finditer(r"\bmakita\s+(e|ou|nem)\s+skil\b", lower):
        covering = [e for e in entities
                    if e["start"] < m.end() and e["end"] > m.start()
                    and e["type"] == "DEVICE"]
        if any(e["value"].casefold() == "makita skil" for e in covering):
            return False
        vals = {e["value"].casefold() for e in covering}
        if not {"makita", "skil"}.issubset(vals):
            return False
    for typ, compounds in COMPOUND_ENTITIES.items():
        for compound in compounds:
            for m in re.finditer(r"(?<![A-Za-zÀ-ÿ0-9])" + re.escape(compound) +
                                 r"(?![A-Za-zÀ-ÿ0-9])", lower, re.I):
                covering = [e for e in entities if e["start"] <= m.start()
                            and e["end"] >= m.end() and e["type"] == typ]
                if not covering:
                    return False
    return True

# V63.16 — FIREWALL SEMÂNTICO
# Regras deliberadamente rígidas para impedir que o próprio gerador crie
# exemplos de fronteira que ensinam a mesma ação em duas intenções.
POWER_ON_TERMS = (
    "ligar", "liga", "ligue", "acender", "acende", "acenda",
    "energizar", "energiza", "energize", "ativar", "ativa", "ative",
)
POWER_OFF_TERMS = (
    "desligar", "desliga", "desligue", "apagar", "apaga", "apague",
    "desativar", "desativa", "desative", "desconectar", "desconecta",
)
PROCESS_START_TERMS = (
    "iniciar", "inicia", "inicie", "começar", "começa", "comece",
    "dar partida", "dá partida", "rodar", "roda", "ponha para rodar",
    "põe para rodar", "coloca para rodar", "manda rodar",
)
PROCESS_STOP_TERMS = (
    "parar", "pare", "interromper", "interrompe", "interrompa",
    "pausar", "pausa", "pause", "cessar", "cessa", "abortar", "aborta",
    "deixar de rodar", "deixa de rodar",
)

BRIGHTNESS_EXPLICIT_TERMS = (
    "brilho", "luminosidade", "intensidade", "claridade", "iluminação",
    "iluminacao", "mais luz", "menos luz", "mais claro", "menos claro", "mais escuro", "mais escura", "mais escuros", "mais escuras",
    "mais iluminado", "mais iluminada", "menos iluminado", "menos iluminada",
    "mais forte", "menos forte", "mais fraco", "mais fraca",
    "clarear", "clareia", "clareie", "escurecer", "escurece", "escureça",
    "%", "por cento", "porcento",
)

COLOR_ACTION_TERMS = (
    "mude a cor", "mudar a cor", "muda a cor", "altere a cor",
    "alterar a cor", "altera a cor", "defina a cor", "definir a cor",
    "coloque", "colocar", "coloca", "deixe", "deixar", "deixa",
    "ponha", "põe", "mude", "mudar", "ficar em", "ficar na cor", "quero", "configure",
)

def _has_any_term(text: str, terms: Tuple[str, ...]) -> bool:
    return any(_contains_lexeme(text, term) for term in terms)

def validate_intent_lexical_purity(sample: Dict) -> bool:
    """V63.16: firewall semântico rígido contra vazamento entre intenções."""
    t = sample["text"].casefold()
    intent = sample["intent"]

    # 1) Energia != início de processo.
    if intent == "TURN_ON":
        if _has_any_term(t, PROCESS_START_TERMS) or _has_any_term(t, PROCESS_STOP_TERMS):
            return False
    elif intent == "TURN_OFF":
        if _has_any_term(t, PROCESS_START_TERMS) or _has_any_term(t, PROCESS_STOP_TERMS):
            return False
    elif intent == "START":
        if _has_any_term(t, POWER_ON_TERMS) or _has_any_term(t, POWER_OFF_TERMS):
            return False
    elif intent == "STOP":
        if _has_any_term(t, POWER_ON_TERMS) or _has_any_term(t, POWER_OFF_TERMS):
            return False

    # 2) Brilho não pode ser ensinado apenas pela palavra "luz".
    # Precisa existir um núcleo de ajuste, atributo ou medida.
    if intent == "SET_BRIGHTNESS":
        has_measure = bool(re.search(
            r"\b\d+(?:[.,]\d+)?\s*(?:%|por cento|porcento)(?!\w)", t, re.I))
        if not _has_any_term(t, BRIGHTNESS_EXPLICIT_TERMS) and not has_measure:
            return False

    # 3) Cor exige uma ação de configuração + cor explícita.
    # "Cor da luz: azul" / "cor da luz" não deve ser usado como comando SET_COLOR.
    if intent == "SET_COLOR":
        has_color = _has_any_term(t, tuple(COLORS))
        has_action = _has_any_term(t, COLOR_ACTION_TERMS)
        if not has_color or not has_action:
            return False

    forbidden = {
        "TURN_ON": [r"\b(iniciar|inicie|inicia|começar|começa|comece|dar partida|parar|pare|interromper|interrompa|pausar|pause)\b",
                     r"\b(desligar|desliga|desligue|apagar|apaga|apague)\b",
                     r"\b(fazer funcionar|faz funcionar|pôr para funcionar|põe para funcionar|colocar para funcionar|coloca para funcionar)\b"],
        "TURN_OFF": [r"\b(iniciar|inicie|inicia|começar|começa|comece|dar partida|parar|pare|interromper|interrompa|pausar|pause)\b",
                      r"\b(fazer parar|faz parar|parar de funcionar)\b"],
        "START": [r"\b(desligar|desliga|desligue|apagar|apaga|apague|desativar|desativa|desative)\b",
                   r"\b(parar|pare|interromper|interrompa|pausar|pause)\b",
                   r"\b(ligar|liga|ligue|acender|acende|acenda)\b"],
        "STOP": [r"\b(desligar|desliga|desligue|apagar|apaga|apague|desativar|desativa|desative)\b",
                  r"\b(ligar|liga|ligue|acender|acende|acenda|iniciar|inicia|inicie|começar|começa|comece)\b"],
        "OPEN": [r"\b(fechar|fecha|feche|travar|trava|trave|bloquear|bloqueia|bloqueie)\b"],
        "CLOSE": [r"\b(abrir|abre|abra|destravar|destrava|destrave|liberar|libera|libere)\b"],
    }
    for pat in forbidden.get(intent, []):
        if re.search(pat, t):
            return False
    if intent in NUMERIC_INTENTS and re.search(r"\b(desligar|desliga|desligue|ligar|liga|ligue|parar|pare|interromper|interrompa)\b", t):
        return False
    return True

def validate_sample(s: Dict) -> bool:
    if not grammar_valid(s["text"]):
        return False
    text = s["text"]
    for e in s["entities"]:
        if text[e["start"]:e["end"]] != e["value"]:
            return False

    if not temporal_entities_are_unambiguous(text, s["entities"]):
        return False

    if s["intent"] in NUMERIC_INTENTS and s["operation"] == "SET":
        if not any(e["type"] == "MEASURE" for e in s["entities"]):
            return False
    if s["intent"] == "SET_COLOR" and not any(e["type"] == "COLOR" for e in s["entities"]):
        return False
    obj = None
    loc = None
    for e in s["entities"]:
        if e["type"] == "DEVICE":
            obj = e["value"]
        elif e["type"] == "LOCATION":
            loc = e["value"]
    erros = validar_frase(s["text"], obj, loc)
    if erros:
        return False
    if not validate_contextual_composition(s["text"], s["entities"]):
        return False
    if not validate_intent_lexical_purity(s):
        return False
    if not intent_boundary_ok(s["text"], s.get("intent", "")):
        return False

    op = s.get("operation", "NONE")
    if s["intent"] in NUMERIC_INTENTS:
        if infer_operation_from_text(s["text"], s["intent"]) != op:
            return False
    allowed = COMPATIBLE_OPERATIONS.get(s["intent"], {"NONE"})
    if op not in allowed:
        return False

    if s["intent"] in NUMERIC_INTENTS and op != "NONE":
        # SET exige medida explícita; INCREASE/DECREASE podem ser relativos
        # ("mais luz", "menos volume") e, portanto, não ter MEASURE.
        if op == "SET" and not any(e["type"] == "MEASURE" for e in s["entities"]):
            return False

    return True

LINGUISTIC_FAMILIES = (
    "SHORT",
    "INFINITIVE",
    "IMPERATIVE",
    "INTERROGATIVE",
    "POLITE",
    "DECLARATIVE",
    "COLLOQUIAL",
    "NEGATIVE",
    "CONDITIONAL",
    "SUGGESTION",
)

def detect_linguistic_family(text: str) -> str:
    """Classificador determinístico das famílias, preservando SHORT após temporais."""
    t = text.casefold().strip()
    if t.startswith(("não ", "nao ")):
        return "NEGATIVE"
    if t.startswith("que tal "):
        return "SUGGESTION"
    if t.startswith(("seria bom ", "caso ", "se ", "quando puder")):
        return "CONDITIONAL"
    if t.startswith(("por favor", "por gentileza", "você poderia", "gostaria que", "eu gostaria que")):
        return "POLITE"
    if any(x in t for x in (" aí", " pra ", "bota ", "manda ", "deixa ")):
        return "COLLOQUIAL"
    if t.endswith("?") or t.startswith(("você pode", "pode ", "tem como", "dá para", "será que", "quer ")):
        return "INTERROGATIVE"
    # INFINITIVE_COMMAND: comando elíptico/nominalizado iniciado diretamente
    # pelo infinitivo. Detectamos antes de SHORT para não confundir
    # "Ligar a luz da sala." com um comando curto imperativo.
    infinitive_starts = (
        "ligar ", "desligar ", "abrir ", "fechar ", "iniciar ", "parar ",
        "ativar ", "desativar ", "começar ", "interromper ",
        "ajustar ", "definir ", "regular ", "configurar ", "colocar ",
        "mudar ", "alterar ", "aumentar ", "diminuir ", "reduzir ",
        "abaixar ", "clarear ", "escurecer ", "verificar ", "consultar ",
        "saber ", "deixar ", "programar ", "agendar ", "configurar ",
    )
    if t.startswith(infinitive_starts):
        return "INFINITIVE"
    base = t.rstrip(" .?!")
    temporal_patterns = (
        r"\s+depois de amanhã$", r"\s+amanhã$", r"\s+na próxima semana$",
        r"\s+na (?:segunda|terça|quarta|quinta|sexta)-feira$",
        r"\s+no (?:sábado|domingo)$", r"\s+às?\s+\d{1,2}:\d{2}$",
        r"\s+daqui a .+$", r"\s+em .+ (?:minuto|minutos|hora|horas|dia|dias)$",
        r"\s+(?:todos os dias úteis|todos os dias|diariamente|semanalmente|toda semana)$",
    )
    stripped = base
    for pat in temporal_patterns:
        candidate = re.sub(pat, "", stripped, flags=re.I)
        if candidate != stripped:
            stripped = candidate
            break
    # Agendamento explícito é sempre IMPERATIVE nesta ontologia.
    if t.startswith(("programe ", "agende ")):
        return "IMPERATIVE"
    # Declarações de intenção têm prioridade sobre SHORT mesmo quando curtas
    # (ex.: "Quero mais luz na sala").
    if t.startswith(("quero ", "vou ", "gostaria ", "preciso ", "eu quero ", "eu preciso ")):
        return "DECLARATIVE"
    # Imperativos explícitos também têm prioridade sobre SHORT.
    if t.startswith(("ligue", "liga", "desligue", "desliga", "abra", "abre", "fecha", "feche",
                     "inicie", "inicia", "pare", "pause", "interrompa", "ative", "aumente",
                     "diminua", "reduza", "mude", "coloque", "baixe", "verifique", "consulte",
                     "ajuste", "defina", "configure", "regule", "programe", "agende",
                     "comece", "destrave", "bloqueie", "trave", "tranque",
                     "clareia", "clareie", "escurece", "escureça", "ilumina", "ilumine",
                     "deixe", "aumente", "diminua", "reduza", "abaixe", "suba", "eleve")):
        return "IMPERATIVE"
    if len(stripped.split()) <= 5 and stripped:
        return "SHORT"
    return "DECLARATIVE"

def coordinated_device_examples() -> List[Dict[str, Any]]:
    return [
        {"text": "Ligue a Makita Skil.", "entities": [
            {"type": "DEVICE", "value": "Makita Skil"}]},
        {"text": "Ligue a Makita e a Skil.", "entities": [
            {"type": "DEVICE", "value": "Makita"},
            {"type": "DEVICE", "value": "Skil"}]},
        {"text": "Pode ligar a Makita Skil da garagem?", "entities": [
            {"type": "DEVICE", "value": "Makita Skil"},
            {"type": "LOCATION", "value": "garagem"}]},
        {"text": "Pode ligar a Makita da garagem e a Skil da bancada?", "entities": [
            {"type": "DEVICE", "value": "Makita"},
            {"type": "LOCATION", "value": "garagem"},
            {"type": "DEVICE", "value": "Skil"},
            {"type": "LOCATION", "value": "bancada"}]},
    ]


class GenerationInvariantError(RuntimeError):
    pass


def _balanced_targets(total: int, labels: Tuple[str, ...]) -> Dict[str, int]:
    """Distribui exatamente *total* entre os rótulos, diferença máxima 1."""
    base, rem = divmod(total, len(labels))
    return {label: base + (1 if i < rem else 0) for i, label in enumerate(labels)}


class DatasetGenerator:
    CONTRAST_PAIRS = [
        ("TURN_ON", "TURN_OFF"),
        ("START", "STOP"),
        ("OPEN", "CLOSE"),
        ("TURN_ON", "START"),
        ("TURN_OFF", "STOP"),
        ("GET_STATUS", "SET_TEMPERATURE"),
        ("GET_STATUS", "SET_SPEED"),
        ("GET_STATUS", "SET_BRIGHTNESS"),
        ("GET_STATUS", "SET_VOLUME"),
        ("GET_STATUS", "SET_VOLTAGE"),
        ("GET_STATUS", "SET_COLOR"),
        ("SET_BRIGHTNESS", "SET_COLOR"),
        ("SET_BRIGHTNESS", "TURN_ON"),
        ("SET_BRIGHTNESS", "TURN_OFF"),
        ("STOP", "GET_STATUS"),
        ("CLOSE", "GET_STATUS"),
        ("OPEN", "GET_STATUS"),
        ("TURN_ON", "GET_STATUS"),
        ("TURN_OFF", "GET_STATUS"),
        ("START", "GET_STATUS"),
        ("SET_TEMPERATURE", "TURN_OFF"),
        ("SET_SPEED", "STOP"),
        ("SET_VOLUME", "TURN_OFF"),
    ]
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.rng = random.Random(cfg.random_seed)
        self.synt = SyntacticGenerator(self.rng)
        self.global_seen: set = set()
        self.intent_seen: Dict[str, set] = {i: set() for i in INTENT_MAP}
        self.audit = {
            "seed_matrix": Counter(),
            "pair_blocks": Counter(),
            "family_counts": Counter(),
            "rejections": Counter(),
            "duplicates": 0,
            "primary_seeds": 0,
            "semantic_anchors": 0,
            "explicit_schedules": 0,
            # V63.21: déficits são informativos e nunca bloqueiam as próximas intenções.
            "shortage_intent": Counter(),
            "shortage_family": Counter(),
            "shortage_temporal": Counter(),
            "shortage_schedule": Counter(),
            "missing_family": Counter(),
            "partial_families": Counter(),
        }
        self._op_counts: Dict[str, Counter] = {i: Counter() for i in INTENT_MAP}
        self._composition_counts: Dict[str, Counter] = {i: Counter() for i in INTENT_MAP}
        self._family_counts_by_intent: Dict[str, Counter] = {i: Counter() for i in INTENT_MAP}
        self._op_targets: Dict[str, Dict[str, int]] = {i: {} for i in INTENT_MAP}
        self._op_schedule: Dict[str, Any] = {}
        self._special_counts: Dict[str, Counter] = {i: Counter() for i in INTENT_MAP}
        self._special_targets: Dict[str, Dict[str, int]] = {i: {} for i in INTENT_MAP}

    def _generate_for_intent(self, intent: str, obj_key: str, loc_key: Optional[str],
                             op: str = "NONE") -> Optional[Dict]:
        comp = self.rng.choice(self.cfg.compositions)
        device = surface_device(obj_key, self.rng, comp)
        location = surface_location(loc_key, self.rng, comp) if loc_key else None

        if intent in NUMERIC_INTENTS:
            if op == "NONE" or op not in COMPATIBLE_OPERATIONS[intent]:
                valid_ops = get_valid_operations(intent)
                weights = [50 if o == "SET" else 25 for o in valid_ops]
                op = self.rng.choices(valid_ops, weights=weights, k=1)[0]
        else:
            op = "NONE"

        if intent in ACTION_INTENTS and self.rng.random() < 0.12 and DEVICE_COMMAND_GROUPS:
            pair = self.rng.choice(DEVICE_COMMAND_GROUPS)
            if all(valid_combo(d, loc_key) for d in pair):
                first, second = pair
                combined_surface = f"{first} e a {second}"
                text, raw_specs = self.synt.gerar_acao(
                    intent, first, loc_key, combined_surface, location
                )
                raw_specs = [("DEVICE", first), ("DEVICE", second)] + [
                    x for x in raw_specs if x[0] != "DEVICE"
                ]
            else:
                text, raw_specs = self.synt.gerar_acao(
                    intent, obj_key, loc_key, device, location
                )
        elif intent in ACTION_INTENTS:
            text, raw_specs = self.synt.gerar_acao(
                intent, obj_key, loc_key, device, location
            )
        elif intent == "GET_STATUS":
            text, raw_specs = self.synt.gerar_status(
                obj_key, loc_key, device, location
            )
        elif intent == "SET_COLOR":
            # Cor é sempre uma configuração explícita: SET_COLOR -> SET.
            op = "SET"
            text, raw_specs = self.synt.gerar_cor(
                obj_key, loc_key, device, location
            )
        elif intent in NUMERIC_INTENTS:
            text, raw_specs = self.synt.gerar_numerico(
                intent, obj_key, loc_key, op, device, location
            )
        else:
            return None

        if intent in NUMERIC_INTENTS:
            detected_op = infer_operation_from_text(text, intent)
            if detected_op != op:
                return None
        temporal_spec = None
        if self.rng.random() < self.cfg.temporal_ratio:
            text, temp_type, temp_val = self.synt.adicionar_temporal(
                text, intent=intent,
                obj_nom=self.synt.frase_nominal(obj_key, loc_key, com_artigo=True,
                                                surface_obj=device, surface_loc=location),
                obj_key=obj_key
            )
            temporal_spec = (temp_type, temp_val)
            raw_specs = list(raw_specs) + [temporal_spec]

        text = normalize(text)
        specs = []
        for typ, val in raw_specs:
            if typ == "DEVICE":
                specs.append((typ, val))
            elif typ == "LOCATION" and location:
                specs.append((typ, val))
            else:
                specs.append((typ, val))
        if not location:
            specs = [(t, v) for t, v in specs if t != "LOCATION"]

        ents = annotate(text, specs)

        if not any(e["type"] == "DEVICE" for e in ents) and device in text:
            pos = text.find(device)
            if pos >= 0:
                ents.append({
                    "start": pos, "end": pos + len(device),
                    "type": "DEVICE", "value": device
                })

        if intent == "SET_COLOR" and not any(e["type"] == "COLOR" for e in ents):
            for c in COLORS:
                if c in text or COLOR_FEM[c] in text:
                    cor = c if c in text else COLOR_FEM[c]
                    pos = text.find(cor)
                    if pos >= 0:
                        ents.append({
                            "start": pos, "end": pos + len(cor),
                            "type": "COLOR", "value": cor
                        })
                        break

        if intent in NUMERIC_INTENTS and op == "SET" and not any(
            e["type"] == "MEASURE" for e in ents
        ):
            m = re.search(r"\d+(?:[.,]\d+)?(?:°C|graus|V|volts|%|por cento|porcento)", text, re.I)
            if m:
                val = m.group(0)
                ents.append({
                    "start": m.start(), "end": m.end(),
                    "type": "MEASURE", "value": val
                })

        ents.sort(key=lambda e: e["start"])
        sample = {
            "text": text, "intent": intent, "operation": op,
            "entities": ents, "category": "generative"
        }
        return enrich_sample(sample)

    def _candidate(self, intent: str, family: Optional[str] = None,
                   force_obj: Optional[str] = None,
                   force_loc: Optional[str] = None,
                   force_op: Optional[str] = None,
                   max_attempts: int = 50) -> Optional[Dict]:
        objs = objects_for_intent(intent)
        if not objs:
            return None
        for _ in range(max_attempts):
            obj_key = force_obj or self.rng.choice(objs)
            loc_key = force_loc if force_loc is not None else self.rng.choice(combos_for(obj_key))
            op = force_op if force_op is not None else "NONE"

            s = self._generate_for_intent(intent, obj_key, loc_key, op)
            if not s:
                self.audit["rejections"]["empty"] += 1
                continue
            if not validate_sample(s):
                self.audit["rejections"]["validation"] += 1
                continue

            detected = detect_linguistic_family(s["text"])
            if family is not None and detected != family:
                self.audit["rejections"][f"family:{family}"] += 1
                continue

            s["family"] = detected
            return s
        return None

    def _primary_command_seed(self, intent: str) -> Optional[Dict]:
        preferred = {
            "TURN_ON": ("luz", "sala"),
            "TURN_OFF": ("luz", "sala"),
            "OPEN": ("porta", "sala"),
            "CLOSE": ("porta", "sala"),
            "START": ("ventilador", "sala"),
            "STOP": ("ventilador", "sala"),
        }

        if intent in preferred:
            obj, loc = preferred[intent]
            if obj not in objects_for_intent(intent):
                obj = objects_for_intent(intent)[0]
            combos = combos_for(obj)
            if loc not in combos:
                loc = combos[0] if combos else None
            device = surface_device(obj, self.rng, "base")
            location = surface_location(loc, self.rng, "base") if loc else None
            forms = {
                "TURN_ON": "Ligue",
                "TURN_OFF": "Desligue",
                "OPEN": "Abra",
                "CLOSE": "Feche",
                "START": "Inicie",
                "STOP": "Pare",
            }
            phrase = normalize(f"{forms[intent]} {self.synt.frase_nominal(obj, loc, True, False, device, location)}.")
            ents = annotate(phrase, [("DEVICE", device), ("LOCATION", location)])
            sample = {"text": phrase, "intent": intent, "operation": "NONE",
                      "entities": ents, "category": "primary", "family": "IMPERATIVE", "protected_primary": True}
            return enrich_sample(sample) if validate_sample(sample) else None

        if intent == "GET_STATUS":
            obj = objects_for_intent(intent)[0]
            locs = combos_for(obj)
            loc = locs[0] if locs else None
            device = surface_device(obj, self.rng, "base")
            location = surface_location(loc, self.rng, "base") if loc else None
            obj_nom = self.synt.frase_nominal(obj, loc, True, False, device, location)
            phrase = normalize(f"Verifique {obj_nom}.")
            ents = annotate(phrase, [("DEVICE", device), ("LOCATION", location)])
            sample = {"text": phrase, "intent": intent, "operation": "NONE",
                      "entities": ents, "category": "primary", "family": "IMPERATIVE", "protected_primary": True}
            return enrich_sample(sample) if validate_sample(sample) else None

        if intent == "SET_COLOR":
            obj = "luz" if "luz" in objects_for_intent(intent) else objects_for_intent(intent)[0]
            locs = combos_for(obj)
            loc = "sala" if "sala" in locs else (locs[0] if locs else None)
            device = surface_device(obj, self.rng, "base")
            location = surface_location(loc, self.rng, "base") if loc else None
            color = "branco"
            obj_nom = self.synt.frase_nominal(obj, loc, True, False, device, location)
            phrase = normalize(f"Coloque {obj_nom} em {color}.")
            ents = annotate(phrase, [("DEVICE", device), ("LOCATION", location), ("COLOR", color)])
            sample = {"text": phrase, "intent": intent, "operation": "SET",
                      "entities": ents, "category": "primary", "family": "IMPERATIVE", "protected_primary": True}
            return enrich_sample(sample) if validate_sample(sample) else None

        if intent in NUMERIC_INTENTS:
            obj = objects_for_intent(intent)[0]
            locs = combos_for(obj)
            loc = locs[0] if locs else None
            device = surface_device(obj, self.rng, "base")
            location = surface_location(loc, self.rng, "base") if loc else None
            attr_phrase = self.synt.atributo_intent(obj, loc, intent, device, location)
            value = self.synt.valor_str(obj)
            phrase = normalize(f"Defina {attr_phrase} em {value}.")
            ents = annotate(phrase, [("DEVICE", device), ("LOCATION", location), ("MEASURE", None)])
            for m in re.finditer(r"\b(\d+(?:[.,]\d+)?)\s*(°C|graus|V|volts|%|por cento|porcento)(?!\w)", phrase, re.I):
                if not any(e["start"] == m.start() and e["end"] == m.end() for e in ents):
                    ents.append({"start": m.start(), "end": m.end(), "type": "MEASURE", "value": phrase[m.start():m.end()]})
            sample = {"text": phrase, "intent": intent, "operation": "SET",
                      "entities": sorted(ents, key=lambda e: e["start"]),
                      "category": "primary", "family": "IMPERATIVE", "protected_primary": True}
            return enrich_sample(sample) if validate_sample(sample) and infer_operation_from_text(phrase, intent) == "SET" else None

        return None

    def _secondary_primary_seed(self, intent: str) -> Optional[Dict]:
        preferred = {
            "TURN_ON": ("luz", "sala", "Liga"),
            "TURN_OFF": ("luz", "sala", "Desliga"),
            "OPEN": ("porta", "sala", "Abre"),
            "CLOSE": ("porta", "sala", "Fecha"),
            "START": ("ventilador", "sala", "Inicia"),
            "STOP": ("ventilador", "sala", "Para"),
        }
        if intent not in preferred:
            return None
        obj, loc, verb = preferred[intent]
        if obj not in objects_for_intent(intent):
            obj = objects_for_intent(intent)[0]
        combos = combos_for(obj)
        if loc not in combos:
            loc = combos[0] if combos else None
        device = surface_device(obj, self.rng, "base")
        location = surface_location(loc, self.rng, "base") if loc else None
        obj_nom = self.synt.frase_nominal(obj, loc, True, False, device, location)
        phrase = normalize(f"{verb} {obj_nom}.")
        ents = annotate(phrase, [("DEVICE", device), ("LOCATION", location)])
        sample = {"text": phrase, "intent": intent, "operation": "NONE",
                  "entities": ents, "category": "primary", "family": "IMPERATIVE",
                  "protected_primary": True}
        return enrich_sample(sample) if validate_sample(sample) else None

    def _choose_operation_balanced(self, intent: str, family: str) -> str:
        if intent not in NUMERIC_INTENTS:
            return "NONE"
        allowed = {"SET", "INCREASE", "DECREASE"}
        if family == "NEGATIVE":
            allowed.discard("SET")
        queue = self._op_schedule.setdefault(intent, [])
        for op in queue:
            if op in allowed:
                return op
        return "INCREASE" if family == "NEGATIVE" else "SET"

    def _choose_composition_balanced(self, intent: str) -> str:
        choices = list(self.cfg.compositions)
        counts = self._composition_counts[intent]
        comp = min(choices, key=lambda x: (counts[x], x))
        counts[comp] += 1
        return comp

    def _make_sample(self, text: str, intent: str, operation: str,
                     specs: List[Tuple[str, Optional[str]]], family: str,
                     category: str = "generative") -> Optional[Dict]:
        text = normalize(text)
        ents = annotate(text, specs)
        sample = {"text": text, "intent": intent, "operation": operation,
                  "entities": sorted(ents, key=lambda e: e["start"]),
                  "category": category, "family": family, "_validated": False}
        if not validate_sample(sample):
            self.audit["rejections"][f"direct_invalid:{intent}:{family}"] += 1
            return None
        detected = detect_linguistic_family(text)
        if detected != family:
            self.audit["rejections"][f"direct_family:{intent}:{family}:{detected}"] += 1
            return None
        if intent in NUMERIC_INTENTS:
            self._op_counts[intent][operation] += 1
            queue = self._op_schedule.setdefault(intent, [])
            if operation in queue:
                queue.remove(operation)
        sample["_validated"] = True
        return enrich_sample(sample)

    def _temporalize_sample(self, sample: Dict) -> Optional[Dict]:
        """Adiciona um slot temporal preservando intenção, operação e família."""
        temp, temp_type, temp_val = self.synt.gerar_temporal()
        text = sample["text"]
        if text.endswith("?"):
            text = text[:-1].rstrip() + f" {temp}?"
        elif text.endswith("."):
            text = text[:-1].rstrip() + f" {temp}."
        else:
            text = text.rstrip() + f" {temp}."
        specs = []
        for e in sample.get("entities", []):
            if e["type"] in {"DEVICE", "LOCATION", "COLOR"}:
                specs.append((e["type"], e["value"]))
        specs.append((temp_type, temp_val))
        candidate = self._make_sample(text, sample["intent"], sample["operation"],
                                      specs, sample.get("family", detect_linguistic_family(text)),
                                      sample.get("category", "generative"))
        return candidate

    def _explicit_schedule_seed(self, intent: str, force_op: Optional[str] = None) -> Optional[Dict]:
        """Cria UMA semente obrigatória de programação por intenção.

        Esta semente não é uma família nova: ela pertence à família linguística
        correspondente (normalmente IMPERATIVE). O objetivo é garantir que a
        geração real contenha pelo menos uma forma explícita de agendamento por
        intenção, em vez de depender de uma probabilidade de 25%.
        """
        if self.cfg.temporal_ratio <= 0:
            return None

        temp, temp_type, temp_val = self.synt.gerar_temporal()
        objs = objects_for_intent(intent)
        if not objs:
            return None
        obj = objs[self.rng.randrange(len(objs))]
        locs = combos_for(obj)
        loc = locs[self.rng.randrange(len(locs))] if locs else None
        comp = self._choose_composition_balanced(intent)
        device = surface_device(obj, self.rng, comp)
        location = surface_location(loc, self.rng, comp) if loc else None
        obj_nom = self.synt.frase_nominal(obj, loc, True, False, device, location)

        if intent in ACTION_INTENTS:
            inf = VERB_FORMS[intent]["inf"]
            phrase = f"Programe {obj_nom} para {inf} {temp}."
            return self._make_sample(
                phrase, intent, "NONE",
                [("DEVICE", device)] + ([('LOCATION', location)] if location else []) + [(temp_type, temp_val)],
                "IMPERATIVE", "explicit_schedule"
            )

        if intent == "SET_COLOR":
            color = self.rng.choice(COLORS_SINGLE)
            if OBJECTS[obj]["genero"] == "f" and color in COLOR_FEM:
                color = COLOR_FEM[color]
            phrase = f"Programe {obj_nom} para ficar em {color} {temp}."
            # SET_COLOR é uma configuração explícita: mesmo quando a frase
            # é agendada, a operação semântica continua sendo SET.
            return self._make_sample(
                phrase, intent, "SET",
                [("DEVICE", device), ("COLOR", color)] + ([('LOCATION', location)] if location else []) + [(temp_type, temp_val)],
                "IMPERATIVE", "explicit_schedule"
            )

        if intent == "GET_STATUS":
            attr_names = {
                "brilho": ("o", "brilho"), "volume": ("o", "volume"),
                "velocidade": ("a", "velocidade"), "temperatura": ("a", "temperatura"),
                "voltagem": ("a", "voltagem"), "estado": ("o", "estado"),
            }
            attr = OBJECTS[obj]["atributo"]
            article, word = attr_names.get(attr, ("o", "estado"))
            base = self.synt.frase_nominal(obj, loc, False, False, device, location)
            prep_attr = "do" if article == "o" else "da"
            phrase = f"Programe uma verificação {prep_attr} {word} de {base} para {temp}."
            return self._make_sample(
                phrase, intent, "NONE",
                [("DEVICE", device)] + ([('LOCATION', location)] if location else []) + [(temp_type, temp_val)],
                "IMPERATIVE", "explicit_schedule"
            )

        if intent in NUMERIC_INTENTS:
            op = force_op or self._choose_operation_balanced(intent, "IMPERATIVE")
            value = self.synt.valor_str(obj)
            canonical_attr = {
                "SET_BRIGHTNESS": "o brilho",
                "SET_SPEED": "a velocidade",
                "SET_TEMPERATURE": "a temperatura",
                "SET_VOLTAGE": "a voltagem",
                "SET_VOLUME": "o volume",
            }[intent]
            attr_phrase = f"{canonical_attr} de {obj_nom}"
            if op == "SET":
                phrase = f"Programe {attr_phrase} para {value} {temp}."
            else:
                verbs = {
                    "SET_TEMPERATURE": {"INCREASE": "aumentar", "DECREASE": "diminuir"},
                    "SET_SPEED": {"INCREASE": "aumentar", "DECREASE": "reduzir"},
                    "SET_BRIGHTNESS": {"INCREASE": "aumentar", "DECREASE": "diminuir"},
                    "SET_VOLUME": {"INCREASE": "aumentar", "DECREASE": "abaixar"},
                    "SET_VOLTAGE": {"INCREASE": "aumentar", "DECREASE": "reduzir"},
                }[intent]
                phrase = f"Programe {obj_nom} para {verbs[op]} {canonical_attr} para {value} {temp}."
            return self._make_sample(
                phrase, intent, op,
                [("DEVICE", device), ("MEASURE", value)] + ([('LOCATION', location)] if location else []) + [(temp_type, temp_val)],
                "IMPERATIVE", "explicit_schedule"
            )
        return None

    def _special_category_order(self, intent: str) -> List[str]:
        """Categorias especiais ainda deficitárias, priorizando a menor cobertura relativa."""
        targets = self._special_targets.get(intent, {})
        counts = self._special_counts.get(intent, Counter())
        return sorted(
            (c for c, target in targets.items() if counts[c] < target),
            key=lambda c: (counts[c] / max(1, targets[c]), c)
        )

    def _special_candidate(self, intent: str, family: str, obj: str, loc: Optional[str],
                           device: str, location: Optional[str], op: str,
                           category: str) -> Optional[Dict]:
        """Constrói exemplos difíceis sem alterar o esquema das 13 intenções."""
        obj_nom = self.synt.frase_nominal(obj, loc, True, False, device, location)
        loc_phrase = f"{self.synt.prep_de(loc)} {location}" if loc and location else ""

        # ================================================================
        # V63.14 — FRONTEIRA CONTRASTIVA SEMÂNTICA
        # ================================================================
        # Mantém o mesmo dispositivo/local e muda o núcleo semântico.
        # Isso aumenta a separação entre intenções vizinhas sem alterar
        # o esquema do dataset nem criar novas intenções.
        if category == "CONTRASTIVE_BOUNDARY":
            pools = {}
            if intent == "TURN_ON":
                pools = {
                    "SHORT": [f"Liga {obj_nom}.", f"Acende {obj_nom}."],
                    "INFINITIVE": [f"Ligar {obj_nom}.", f"Acender {obj_nom}."],
                    "IMPERATIVE": [f"Ligue {obj_nom}.", f"Acenda {obj_nom}.", f"Ative {obj_nom}."],
                    "INTERROGATIVE": [f"Pode ligar {obj_nom}?", f"Pode acender {obj_nom}?"],
                    "POLITE": [f"Por favor, ligue {obj_nom}."],
                    "DECLARATIVE": [f"Quero {obj_nom} ligado.", f"Quero deixar {obj_nom} ligado."],
                    "COLLOQUIAL": [f"Liga {obj_nom} aí.", f"Deixa {obj_nom} ligado aí."],
                    "NEGATIVE": [],
                    "CONDITIONAL": [f"Se puder, ligue {obj_nom}."],
                    "SUGGESTION": [f"Que tal ligar {obj_nom}?"],
                }
            elif intent == "TURN_OFF":
                pools = {
                    "SHORT": [f"Desliga {obj_nom}.", f"Apaga {obj_nom}."],
                    "INFINITIVE": [f"Desligar {obj_nom}.", f"Apagar {obj_nom}."],
                    "IMPERATIVE": [f"Desligue {obj_nom}.", f"Apague {obj_nom}.", f"Desative {obj_nom}."],
                    "INTERROGATIVE": [f"Pode desligar {obj_nom}?", f"Pode apagar {obj_nom}?"],
                    "POLITE": [f"Por favor, desligue {obj_nom}."],
                    "DECLARATIVE": [f"Quero {obj_nom} desligado.", f"Não quero {obj_nom} ligado."],
                    "COLLOQUIAL": [f"Desliga {obj_nom} aí.", f"Tira {obj_nom} aí."],
                    "NEGATIVE": [f"Não deixe {obj_nom} ligado."],
                    "CONDITIONAL": [f"Se puder, desligue {obj_nom}."],
                    "SUGGESTION": [f"Que tal desligar {obj_nom}?"],
                }
            elif intent == "START":
                pools = {
                    "SHORT": [f"Inicia {obj_nom}.", f"Começa {obj_nom}."],
                    "INFINITIVE": [f"Iniciar {obj_nom}.", f"Começar {obj_nom}."],
                    "IMPERATIVE": [f"Inicie {obj_nom}.", f"Comece {obj_nom}.", f"Dê partida em {obj_nom}."],
                    "INTERROGATIVE": [f"Pode iniciar {obj_nom}?", f"Pode começar {obj_nom}?"],
                    "POLITE": [f"Por favor, inicie {obj_nom}."],
                    "DECLARATIVE": [f"Quero iniciar {obj_nom}.", f"Quero que {obj_nom} comece."],
                    "COLLOQUIAL": [f"Inicia {obj_nom} aí.", f"Começa {obj_nom} aí."],
                    "NEGATIVE": [],
                    "CONDITIONAL": [f"Se puder, inicie {obj_nom}."],
                    "SUGGESTION": [f"Que tal iniciar {obj_nom}?"],
                }
            elif intent == "STOP":
                pools = {
                    "SHORT": [f"Para {obj_nom}.", f"Interrompe {obj_nom}."],
                    "INFINITIVE": [f"Parar {obj_nom}.", f"Interromper {obj_nom}."],
                    "IMPERATIVE": [f"Pare {obj_nom}.", f"Interrompa {obj_nom}.", f"Pause {obj_nom}."],
                    "INTERROGATIVE": [f"Pode parar {obj_nom}?", f"Pode interromper {obj_nom}?"],
                    "POLITE": [f"Por favor, pare {obj_nom}."],
                    "DECLARATIVE": [f"Quero parar {obj_nom}.", f"Quero que {obj_nom} pare."],
                    "COLLOQUIAL": [f"Para {obj_nom} aí.", f"Interrompe {obj_nom} aí."],
                    "NEGATIVE": [f"Não deixe {obj_nom} rodando."],
                    "CONDITIONAL": [f"Se puder, pare {obj_nom}."],
                    "SUGGESTION": [f"Que tal parar {obj_nom}?"],
                }
            elif intent == "OPEN":
                pools = {
                    "SHORT": [f"Abre {obj_nom}.", f"Destrava {obj_nom}."],
                    "INFINITIVE": [f"Abrir {obj_nom}.", f"Destravar {obj_nom}."],
                    "IMPERATIVE": [f"Abra {obj_nom}.", f"Destrave {obj_nom}."],
                    "INTERROGATIVE": [f"Pode abrir {obj_nom}?", f"Pode destravar {obj_nom}?"],
                    "POLITE": [f"Por favor, abra {obj_nom}."],
                    "DECLARATIVE": [f"Quero {obj_nom} aberto."],
                    "COLLOQUIAL": [f"Abre {obj_nom} aí.", f"Destrava {obj_nom} aí."],
                    "NEGATIVE": [f"Não deixe {obj_nom} fechado."],
                    "CONDITIONAL": [f"Se puder, abra {obj_nom}."],
                    "SUGGESTION": [f"Que tal abrir {obj_nom}?"],
                }
            elif intent == "CLOSE":
                pools = {
                    "SHORT": [f"Fecha {obj_nom}.", f"Trava {obj_nom}."],
                    "INFINITIVE": [f"Fechar {obj_nom}.", f"Travar {obj_nom}."],
                    "IMPERATIVE": [f"Feche {obj_nom}.", f"Trave {obj_nom}."],
                    "INTERROGATIVE": [f"Pode fechar {obj_nom}?", f"Pode travar {obj_nom}?"],
                    "POLITE": [f"Por favor, feche {obj_nom}."],
                    "DECLARATIVE": [f"Quero {obj_nom} fechado."],
                    "COLLOQUIAL": [f"Fecha {obj_nom} aí.", f"Trava {obj_nom} aí."],
                    "NEGATIVE": [f"Não deixe {obj_nom} aberto."],
                    "CONDITIONAL": [f"Se puder, feche {obj_nom}."],
                    "SUGGESTION": [f"Que tal fechar {obj_nom}?"],
                }
            elif intent == "GET_STATUS":
                word = {"brilho":"brilho", "volume":"volume", "velocidade":"velocidade",
                        "temperatura":"temperatura", "voltagem":"voltagem", "estado":"estado"}.get(OBJECTS[obj]["atributo"], "estado")
                pools = {
                    "SHORT": [f"Status de {obj_nom}?", f"Estado de {obj_nom}?"],
                    "INFINITIVE": [f"Verificar o estado de {obj_nom}."],
                    "IMPERATIVE": [f"Verifique o {word} de {obj_nom}."],
                    "INTERROGATIVE": [f"Como está {obj_nom}?", f"Qual é o {word} de {obj_nom}?"],
                    "POLITE": [f"Por favor, me diga como está {obj_nom}."],
                    "DECLARATIVE": [f"Quero saber como está {obj_nom}."],
                    "COLLOQUIAL": [f"Como tá {obj_nom} aí?"],
                    "NEGATIVE": [],
                    "CONDITIONAL": [f"Se puder, me diga como está {obj_nom}."],
                    "SUGGESTION": [f"Que tal verificar como está {obj_nom}?"],
                }
            elif intent == "SET_TEMPERATURE":
                pools = {
                    "SHORT": [f"Esfria {obj_nom}.", f"Aquece {obj_nom}."],
                    "INFINITIVE": [f"Esfriar {obj_nom}.", f"Aquecer {obj_nom}."],
                    "IMPERATIVE": [f"Esfrie {obj_nom}.", f"Aqueça {obj_nom}."],
                    "INTERROGATIVE": [f"Pode esfriar {obj_nom}?", f"Pode aquecer {obj_nom}?"],
                    "POLITE": [f"Por favor, esfrie {obj_nom}."],
                    "DECLARATIVE": [f"Quero {obj_nom} mais frio.", f"Quero {obj_nom} mais quente."],
                    "COLLOQUIAL": [f"Esfria {obj_nom} aí.", f"Aquece {obj_nom} aí."],
                    "NEGATIVE": [],
                    "CONDITIONAL": [f"Se puder, esfrie {obj_nom}."],
                    "SUGGESTION": [f"Que tal esfriar {obj_nom}?"],
                }
            elif intent in {"SET_SPEED", "SET_VOLUME", "SET_BRIGHTNESS", "SET_VOLTAGE"}:
                attr = NUMERIC_ATTR[intent]
                pools = {
                    "SHORT": [f"Mais {attr} em {obj_nom}.", f"Menos {attr} em {obj_nom}."],
                    "INFINITIVE": [f"Aumentar {attr} de {obj_nom}.", f"Diminuir {attr} de {obj_nom}."],
                    "IMPERATIVE": [f"Aumente {attr} de {obj_nom}.", f"Diminua {attr} de {obj_nom}."],
                    "INTERROGATIVE": [f"Pode aumentar {attr} de {obj_nom}?", f"Pode diminuir {attr} de {obj_nom}?"],
                    "POLITE": [f"Por favor, aumente {attr} de {obj_nom}."],
                    "DECLARATIVE": [f"Quero mais {attr} em {obj_nom}."],
                    "COLLOQUIAL": [f"Aumenta {attr} de {obj_nom} aí.", f"Diminui {attr} de {obj_nom} aí."],
                    "NEGATIVE": [],
                    "CONDITIONAL": [f"Se puder, aumente {attr} de {obj_nom}."],
                    "SUGGESTION": [f"Que tal aumentar {attr} de {obj_nom}?"],
                }
            elif intent == "SET_COLOR":
                color = self.rng.choice(COLORS_SINGLE)
                if OBJECTS[obj]["genero"] == "f" and color in COLOR_FEM:
                    color = COLOR_FEM[color]
                pools = {
                    "SHORT": [f"Cor de {obj_nom}: {color}."],
                    "INFINITIVE": [f"Mudar a cor de {obj_nom} para {color}."],
                    "IMPERATIVE": [f"Coloque {obj_nom} em {color}."],
                    "INTERROGATIVE": [f"Pode colocar {obj_nom} em {color}?"],
                    "POLITE": [f"Por favor, coloque {obj_nom} em {color}."],
                    "DECLARATIVE": [f"Quero {obj_nom} em {color}."],
                    "COLLOQUIAL": [f"Deixa {obj_nom} em {color} aí."],
                    "NEGATIVE": [],
                    "CONDITIONAL": [f"Se puder, coloque {obj_nom} em {color}."],
                    "SUGGESTION": [f"Que tal colocar {obj_nom} em {color}?"],
                }
                phrase = self.rng.choice(pools.get(family, [])) if pools.get(family) else None
                if phrase:
                    return self._make_sample(phrase, intent, "NONE",
                        [("DEVICE", device), ("COLOR", color)] + ([('LOCATION', location)] if location else []),
                        family, category)
                return None

            phrases = pools.get(family, []) if pools else []
            if phrases:
                phrase = self.rng.choice(phrases)
                boundary_op = op if intent in NUMERIC_INTENTS else "NONE"
                specs = [("DEVICE", device)] + ([('LOCATION', location)] if location else [])
                if intent in NUMERIC_INTENTS and boundary_op == "SET":
                    return None
                return self._make_sample(phrase, intent, boundary_op, specs, family, category)

        # ====== BRILHO: fronteira crítica BRIGHTNESS x COLOR x TURN_ON ======
        if intent == "SET_BRIGHTNESS" and op in {"INCREASE", "DECREASE"}:
            inc = op == "INCREASE"
            if category == "SEMANTIC_PARAPHRASE":
                if inc:
                    pools = {
                        "SHORT": [f"Mais luz {loc_phrase}.".strip(), f"Mais claridade {loc_phrase}.".strip(),
                                  f"Clareia {obj_nom}.", f"Clareia {location or obj_nom}.",
                                  f"Mais iluminação {loc_phrase}.".strip()],
                        "INFINITIVE": [f"Clarear {obj_nom}.", f"Clarear {location or obj_nom}.", f"Deixar {obj_nom} mais claro."],
                        "IMPERATIVE": [f"Clareie {obj_nom}.", f"Clareie {location or obj_nom}.", f"Deixe {obj_nom} mais iluminado."],
                        "INTERROGATIVE": [f"Pode deixar {obj_nom} mais iluminado?", f"Dá para deixar {obj_nom} mais claro?"],
                        "POLITE": [f"Por favor, deixe {obj_nom} mais claro.", f"Por gentileza, clareie {obj_nom}."],
                        "DECLARATIVE": [f"Quero mais luz {loc_phrase}.".strip(), f"Quero mais claridade {loc_phrase}.".strip(),
                                        f"Preciso de mais iluminação {loc_phrase}.".strip()],
                        "COLLOQUIAL": [f"Deixa {obj_nom} mais forte aí.", f"Bota mais luz {loc_phrase} aí.".strip()],
                        "NEGATIVE": [],
                        "CONDITIONAL": [f"Se puder, deixe {obj_nom} mais claro."],
                        "SUGGESTION": [f"Que tal deixar {obj_nom} mais claro?"],
                    }
                else:
                    pools = {
                        "SHORT": [f"Menos luz {loc_phrase}.".strip(), f"Menos claridade {loc_phrase}.".strip(),
                                  f"Escurece {obj_nom}.", f"Escurece {location or obj_nom}.",
                                  f"Menos iluminação {loc_phrase}.".strip()],
                        "INFINITIVE": [f"Escurecer {obj_nom}.", f"Escurecer {location or obj_nom}.", f"Deixar {obj_nom} menos claro."],
                        "IMPERATIVE": [f"Escureça {obj_nom}.", f"Escureça {location or obj_nom}.", f"Deixe {obj_nom} menos iluminado."],
                        "INTERROGATIVE": [f"Pode deixar {obj_nom} menos iluminado?", f"Dá para deixar {obj_nom} mais fraco?"],
                        "POLITE": [f"Por favor, deixe {obj_nom} menos claro.", f"Por gentileza, escureça {obj_nom}."],
                        "DECLARATIVE": [f"Quero menos luz {loc_phrase}.".strip(), f"Quero menos claridade {loc_phrase}.".strip(),
                                        f"Preciso de menos iluminação {loc_phrase}.".strip()],
                        "COLLOQUIAL": [f"Deixa {obj_nom} mais fraco aí.", f"Bota menos luz {loc_phrase} aí.".strip()],
                        "NEGATIVE": [],
                        "CONDITIONAL": [f"Se puder, deixe {obj_nom} menos claro."],
                        "SUGGESTION": [f"Que tal deixar {obj_nom} menos claro?"],
                    }
                phrases = pools.get(family, [])
                if phrases:
                    phrase = self.rng.choice(phrases)
                    return self._make_sample(
                        phrase, intent, op,
                        [("DEVICE", device)] + ([('LOCATION', location)] if location else []),
                        family, category
                    )

            if category == "COLLOQUIAL_NOISY" and family in {"SHORT", "COLLOQUIAL"}:
                prefix = "mais" if inc else "menos"
                verb = "aumenta" if inc else "diminui"
                phrases = [
                    f"{prefix} {device} {location or ''}.".strip(),
                    f"{verb} {device} {location or ''}.".strip(),
                    f"{device} {location or ''} {('mais forte' if inc else 'mais fraca')}.".strip(),
                ]
                phrase = self.rng.choice(phrases)
                return self._make_sample(
                    phrase, intent, op,
                    [("DEVICE", device)] + ([('LOCATION', location)] if location else []),
                    family, category
                )

            if category == "ENTITY_NEGATIVE" and family in {"SHORT", "COLLOQUIAL", "IMPERATIVE", "DECLARATIVE", "INTERROGATIVE", "POLITE", "CONDITIONAL", "SUGGESTION", "INFINITIVE"}:
                phrase = self.rng.choice([
                    f"Deixa {obj_nom} {'mais forte' if inc else 'mais fraco'}.",
                    f"{'Aumenta' if inc else 'Diminui'} um pouco a luz {loc_phrase}.".strip(),
                    f"{'Clareia' if inc else 'Escurece'} {obj_nom} um pouco.",
                ])
                return self._make_sample(
                    phrase, intent, op,
                    [("DEVICE", device)] + ([('LOCATION', location)] if location else []),
                    family, category
                )

        # ====== COR: contraste explícito com brilho ======
        if intent == "SET_COLOR" and category in {"SEMANTIC_PARAPHRASE", "ENTITY_NEGATIVE"}:
            color = self.rng.choice(COLORS_SINGLE)
            if OBJECTS[obj]["genero"] == "f" and color in COLOR_FEM:
                color = COLOR_FEM[color]
            if family == "SHORT":
                # V63.16: fragmento curto ainda precisa carregar o verbo de configuração.
                # Frases nominais como "Cor da luz: azul" ficam reservadas a GET_STATUS.
                phrases = [f"{device}: colocar em {color}.",
                           f"{device}: mudar para {color}.",
                           f"Cor da {device}: mudar para {color}."]
            elif family == "INFINITIVE":
                phrases = [f"Mudar a cor de {obj_nom} para {color}.", f"Deixar {obj_nom} na cor {color}."]
            elif family == "INTERROGATIVE":
                phrases = [f"Pode deixar {obj_nom} em {color}?", f"Dá para mudar a cor de {obj_nom} para {color}?"]
            elif family == "POLITE":
                phrases = [f"Por favor, mude a cor de {obj_nom} para {color}."]
            elif family == "DECLARATIVE":
                phrases = [f"Quero {obj_nom} na cor {color}."]
            elif family == "COLLOQUIAL":
                phrases = [f"Deixa {obj_nom} na cor {color} aí."]
            elif family == "CONDITIONAL":
                phrases = [f"Se puder, mude a cor de {obj_nom} para {color}."]
            elif family == "SUGGESTION":
                phrases = [f"Que tal deixar {obj_nom} na cor {color}?"]
            elif family == "NEGATIVE":
                phrases = [f"Não deixe de colocar {obj_nom} em {color}."]
            else:
                phrases = [f"Muda a cor de {obj_nom} para {color}.", f"Deixa {obj_nom} na cor {color}.", f"Coloca {obj_nom} em {color}."]
            phrase = self.rng.choice(phrases)
            return self._make_sample(
                phrase, intent, "NONE",
                [("DEVICE", device), ("COLOR", color)] + ([('LOCATION', location)] if location else []),
                family, category
            )

        # ====== AÇÕES: paráfrases de estado ======
        if intent in ACTION_INTENTS and category == "SEMANTIC_PARAPHRASE":
            state = {
                "TURN_ON": ("ligado", "aceso"), "TURN_OFF": ("desligado", "apagado"),
                "OPEN": ("aberto", "destravado"), "CLOSE": ("fechado", "travado"),
                "START": ("funcionando", "rodando"), "STOP": ("parado", "interrompido"),
            }[intent]
            desired, alt = state
            if family == "SHORT":
                phrase = f"{obj_nom.capitalize()} {desired}."
            elif family == "DECLARATIVE":
                phrase = f"Quero {obj_nom} {desired}."
            elif family == "INTERROGATIVE":
                phrase = f"Pode deixar {obj_nom} {desired}?"
            elif family == "POLITE":
                phrase = f"Por favor, deixe {obj_nom} {desired}."
            elif family == "COLLOQUIAL":
                phrase = f"Deixa {obj_nom} {desired} aí."
            elif family == "CONDITIONAL":
                phrase = f"Se puder, deixe {obj_nom} {desired}."
            elif family == "SUGGESTION":
                phrase = f"Que tal deixar {obj_nom} {desired}?"
            elif family == "IMPERATIVE":
                phrase = f"Deixe {obj_nom} {desired}."
            elif family == "INFINITIVE":
                phrase = f"Deixar {obj_nom} {desired}."
            elif family == "NEGATIVE":
                opposite = {"TURN_ON":"desligado", "TURN_OFF":"ligado", "OPEN":"fechado", "CLOSE":"aberto", "START":"parado", "STOP":"rodando"}[intent]
                phrase = f"Não deixe {obj_nom} {opposite}."
            else:
                return None
            return self._make_sample(
                phrase, intent, "NONE",
                [("DEVICE", device)] + ([('LOCATION', location)] if location else []),
                family, category
            )

        # ====== NEGATIVA DE ENTIDADE: ação/status ======
        if category == "ENTITY_NEGATIVE" and intent in ACTION_INTENTS:
            states = {
                "TURN_ON": ("ligado", "aceso"), "TURN_OFF": ("desligado", "apagado"),
                "OPEN": ("aberto", "destravado"), "CLOSE": ("fechado", "travado"),
                "START": ("funcionando", "rodando"), "STOP": ("parado", "interrompido"),
            }
            desired, opposite = states[intent]
            if family == "SHORT": phrase = f"{obj_nom.capitalize()} {desired}."
            elif family == "INFINITIVE": phrase = f"Deixar {obj_nom} {desired}."
            elif family == "IMPERATIVE": phrase = f"Deixe {obj_nom} {desired}."
            elif family == "INTERROGATIVE": phrase = f"Pode deixar {obj_nom} {desired}?"
            elif family == "POLITE": phrase = f"Por favor, deixe {obj_nom} {desired}."
            elif family == "DECLARATIVE": phrase = f"Quero {obj_nom} {desired}."
            elif family == "COLLOQUIAL": phrase = f"Deixa {obj_nom} {desired} aí."
            elif family == "CONDITIONAL": phrase = f"Se puder, deixe {obj_nom} {desired}."
            elif family == "SUGGESTION": phrase = f"Que tal deixar {obj_nom} {desired}?"
            elif family == "NEGATIVE": phrase = f"Não deixe {obj_nom} {opposite}."
            else: return None
            return self._make_sample(phrase, intent, "NONE",
                                     [("DEVICE", device)] + ([('LOCATION', location)] if location else []),
                                     family, category)

        if category == "SEMANTIC_PARAPHRASE" and intent == "GET_STATUS":
            if family == "SHORT": phrase = f"Status da {device}." if OBJECTS[obj]["genero"] == "f" else f"Status do {device}."
            elif family == "INFINITIVE": phrase = f"Saber como está {obj_nom}."
            elif family == "IMPERATIVE": phrase = f"Verifique como está {obj_nom}."
            elif family == "INTERROGATIVE": phrase = f"Como está {obj_nom}?"
            elif family == "POLITE": phrase = f"Por favor, me diga como está {obj_nom}."
            elif family == "DECLARATIVE": phrase = f"Quero saber como está {obj_nom}."
            elif family == "COLLOQUIAL": phrase = f"Como tá {obj_nom} aí?"
            elif family == "CONDITIONAL": phrase = f"Se puder, me diga como está {obj_nom}."
            elif family == "SUGGESTION": phrase = f"Que tal verificar como está {obj_nom}?"
            elif family == "NEGATIVE": phrase = f"Não deixe de verificar como está {obj_nom}."
            else: return None
            return self._make_sample(phrase, intent, "NONE",
                                     [("DEVICE", device)] + ([('LOCATION', location)] if location else []),
                                     family, category)

        if category == "ENTITY_NEGATIVE" and intent == "GET_STATUS":
            if family == "SHORT": phrase = f"Situação atual: {device}."
            elif family == "INFINITIVE": phrase = f"Saber o estado atual de {obj_nom}."
            elif family == "IMPERATIVE": phrase = f"Confira o estado atual de {obj_nom}."
            elif family == "INTERROGATIVE": phrase = f"Está tudo certo com {obj_nom}?"
            elif family == "POLITE": phrase = f"Por favor, confira o estado atual de {obj_nom}."
            elif family == "DECLARATIVE": phrase = f"Quero saber o estado atual de {obj_nom}."
            elif family == "COLLOQUIAL": phrase = f"E aí, como tá {obj_nom} agora?"
            elif family == "CONDITIONAL": phrase = f"Se puder, confira o estado atual de {obj_nom}."
            elif family == "SUGGESTION": phrase = f"Que tal conferir o estado atual de {obj_nom}?"
            elif family == "NEGATIVE": phrase = f"Não deixe de conferir o estado atual de {obj_nom}."
            else: return None
            return self._make_sample(phrase, intent, "NONE",
                                     [("DEVICE", device)] + ([('LOCATION', location)] if location else []),
                                     family, category)

        # ====== COLLOQUIAL_NOISY: ações ======
        if category == "COLLOQUIAL_NOISY" and intent in ACTION_INTENTS and family in {"SHORT", "COLLOQUIAL"}:
            verbs = {"TURN_ON":"liga", "TURN_OFF":"desliga", "OPEN":"abre", "CLOSE":"fecha", "START":"inicia", "STOP":"para"}
            phrase = self.rng.choice([
                f"{verbs[intent]} {device} {location or ''}.".strip(),
                f"{verbs[intent]} {obj_nom} aí.",
            ])
            return self._make_sample(phrase, intent, "NONE",
                                     [("DEVICE", device)] + ([('LOCATION', location)] if location else []),
                                     family, category)

        # ====== NUMÉRICOS: aumento/redução sem medida ======
        if intent in NUMERIC_INTENTS and category == "SEMANTIC_PARAPHRASE" and op in {"INCREASE", "DECREASE"}:
            inc = op == "INCREASE"
            words = {
                "SET_BRIGHTNESS": ("luz", "claridade", "luminosidade"),
                "SET_SPEED": ("velocidade", "ritmo"),
                "SET_TEMPERATURE": ("temperatura", "calor"),
                "SET_VOLUME": ("volume", "som"),
                "SET_VOLTAGE": ("voltagem", "tensão"),
            }[intent]
            word = self.rng.choice(words)
            prefix = "mais" if inc else "menos"
            if family == "SHORT":
                phrase = f"{prefix} {word} {location or device}."
            elif family == "INFINITIVE":
                phrase = f"Deixar {obj_nom} com {prefix} {word}."
            elif family == "IMPERATIVE":
                phrase = f"Deixe {obj_nom} com {prefix} {word}."
            elif family == "INTERROGATIVE":
                phrase = f"Pode deixar {obj_nom} com {prefix} {word}?"
            elif family == "POLITE":
                phrase = f"Por favor, deixe {obj_nom} com {prefix} {word}."
            elif family == "DECLARATIVE":
                phrase = f"Quero {prefix} {word} em {obj_nom}."
            elif family == "COLLOQUIAL":
                phrase = f"Deixa {obj_nom} com {prefix} {word} aí."
            elif family == "CONDITIONAL":
                phrase = f"Se puder, deixe {obj_nom} com {prefix} {word}."
            elif family == "SUGGESTION":
                phrase = f"Que tal {prefix} {word} em {obj_nom}?"
            elif family == "NEGATIVE":
                return None
            else:
                return None
            return self._make_sample(
                phrase, intent, op,
                [("DEVICE", device)] + ([('LOCATION', location)] if location else []),
                family, category
            )

        if category == "ENTITY_NEGATIVE" and intent in NUMERIC_INTENTS and op == "SET":
            value = self.synt.valor_str(obj)
            if family == "SHORT": phrase = f"{NUMERIC_ATTR[intent].capitalize()} {device}: {value}."
            elif family == "INFINITIVE": phrase = f"Definir {NUMERIC_ATTR[intent]} de {obj_nom} em {value}."
            elif family == "IMPERATIVE": phrase = f"Ajuste {NUMERIC_ATTR[intent]} de {obj_nom} para {value}."
            elif family == "INTERROGATIVE": phrase = f"Pode deixar {NUMERIC_ATTR[intent]} de {obj_nom} em {value}?"
            elif family == "POLITE": phrase = f"Por favor, ajuste {NUMERIC_ATTR[intent]} de {obj_nom} para {value}."
            elif family == "DECLARATIVE": phrase = f"Quero {NUMERIC_ATTR[intent]} de {obj_nom} em {value}."
            elif family == "COLLOQUIAL": phrase = f"Bota {NUMERIC_ATTR[intent]} de {obj_nom} em {value} aí."
            elif family == "CONDITIONAL": phrase = f"Se puder, ajuste {NUMERIC_ATTR[intent]} de {obj_nom} para {value}."
            elif family == "SUGGESTION": phrase = f"Que tal deixar {NUMERIC_ATTR[intent]} de {obj_nom} em {value}?"
            elif family == "NEGATIVE": return None
            else: return None
            return self._make_sample(phrase, intent, "SET",
                                     [("DEVICE", device), ("MEASURE", value)] + ([('LOCATION', location)] if location else []),
                                     family, category)

        # ====== COLLOQUIAL_NOISY genérico ======
        if category == "COLLOQUIAL_NOISY" and family in {"SHORT", "COLLOQUIAL"}:
            if intent == "GET_STATUS":
                phrase = self.rng.choice([f"Status {device}.", f"Como tá {device}?", f"E {device}, como tá?"])
                return self._make_sample(phrase, intent, "NONE",
                                         [("DEVICE", device)] + ([('LOCATION', location)] if location else []),
                                         family, category)
            if intent in NUMERIC_INTENTS and op in {"INCREASE", "DECREASE"}:
                word = NUMERIC_ATTR[intent]
                prefix = "mais" if op == "INCREASE" else "menos"
                phrase = f"{prefix} {word} {location or device}."
                return self._make_sample(phrase, intent, op,
                                         [("DEVICE", device)] + ([('LOCATION', location)] if location else []),
                                         family, category)
            if intent == "SET_COLOR":
                color = self.rng.choice(COLORS_SINGLE)
                phrase = f"{device} {color}."
                return self._make_sample(phrase, intent, "SET",
                                         [("DEVICE", device), ("COLOR", color)] + ([('LOCATION', location)] if location else []),
                                         family, category)
        return None

    def _direct_family_seed(self, intent: str, family: str, force_op: Optional[str] = None, special_category: Optional[str] = None) -> Optional[Dict]:
        objs = objects_for_intent(intent)
        if not objs or family not in LINGUISTIC_FAMILIES:
            return None

        # SHORT exige concisão real. Escolhemos dispositivos de nome curto
        # quando houver essa opção; as demais famílias cobrem os compostos.
        if family == "SHORT":
            short_objs = [o for o in objs if len(o.split()) == 1]
            objs = short_objs or objs

        # Balanceamento determinístico de dispositivo/local/composição.
        obj = objs[self.rng.randrange(len(objs))]
        locs = combos_for(obj)
        loc = locs[self.rng.randrange(len(locs))] if locs else None
        comp = self._choose_composition_balanced(intent)
        device = surface_device(obj, self.rng, comp)
        location = surface_location(loc, self.rng, comp) if loc else None
        obj_nom = self.synt.frase_nominal(obj, loc, True, False, device, location)
        op = force_op if force_op is not None else self._choose_operation_balanced(intent, family)

        if special_category:
            special = self._special_candidate(intent, family, obj, loc, device, location, op, special_category)
            if special:
                return special

        # Ações: templates semanticamente separados para evitar mistura de intenções.
        if intent in ACTION_INTENTS:
            forms = {
                "TURN_ON": ("ligue", "ligar", "ligado", "desligado"),
                "TURN_OFF": ("desligue", "desligar", "desligado", "ligado"),
                "OPEN": ("abra", "abrir", "aberto", "fechado"),
                "CLOSE": ("feche", "fechar", "fechado", "aberto"),
                "START": ("inicie", "iniciar", "rodando", "parado"),
                "STOP": ("pare", "parar", "parado", "rodando"),
            }
            imp, inf, desired, opposite = forms[intent]
            opposite = opposite_participle_for(obj, intent)
            if family == "SHORT":
                phrase = self.rng.choice([
                    f"{imp.capitalize()} {obj_nom}.",
                    f"{imp.capitalize()} {obj_nom} agora.",
                    f"{imp.capitalize()} {obj_nom} já.",
                    f"{imp.capitalize()} {obj_nom} aí.",
                    f"{imp.capitalize()} {obj_nom}, por favor.",
                    f"{imp.capitalize()} {obj_nom} agora, por favor.",
                    f"{imp.capitalize()} {obj_nom} já, por favor.",
                ])
            elif family == "INFINITIVE":
                phrase = f"{inf.capitalize()} {obj_nom}."
            elif family == "IMPERATIVE":
                phrase = self.rng.choice([
                    f"{imp.capitalize()} {obj_nom} agora, por favor.",
                    f"{imp.capitalize()} {obj_nom}, por favor.",
                    f"{imp.capitalize()} {obj_nom} imediatamente, por favor.",
                ])
            elif family == "INTERROGATIVE":
                phrase = self.rng.choice([f"Você pode {inf} {obj_nom} agora?", f"Você pode {inf} {obj_nom} agora?"])
            elif family == "POLITE":
                phrase = self.rng.choice([f"Por favor, {inf} {obj_nom}.", f"Você poderia {inf} {obj_nom}, por favor?"])
            elif family == "DECLARATIVE":
                phrase = self.rng.choice([f"Eu quero {inf} {obj_nom} agora.", f"Eu preciso {inf} {obj_nom} agora."])
            elif family == "COLLOQUIAL":
                phrase = self.rng.choice([f"{imp.capitalize()} {obj_nom} aí pra mim.", f"{imp.capitalize()} {obj_nom} aí pra mim."])
            elif family == "NEGATIVE":
                phrase = f"Não deixe {obj_nom} {opposite}."
            elif family == "CONDITIONAL":
                phrase = self.rng.choice([f"Se puder, {inf} {obj_nom}.", f"Caso possa, {inf} {obj_nom}."])
            else:  # SUGGESTION
                phrase = f"Que tal {inf} {obj_nom} agora?"
            return self._make_sample(phrase, intent, "NONE",
                                     [("DEVICE", device)] + ([('LOCATION', location)] if location else []),
                                     family)

        if intent == "GET_STATUS":
            attr = OBJECTS[obj]["atributo"]
            attr_names = {
                "brilho": ("o", "brilho"),
                "volume": ("o", "volume"),
                "velocidade": ("a", "velocidade"),
                "temperatura": ("a", "temperatura"),
                "voltagem": ("a", "voltagem"),
                "estado": ("o", "estado"),
            }
            attr_article, attr_word = attr_names.get(attr, ("o", "estado"))
            attr_name = f"{attr_article} {attr_word}"
            base = self.synt.frase_nominal(obj, loc if family != "SHORT" else None, False, False, device, location)
            if family == "SHORT":
                phrase = self.rng.choice([
                    f"Status de {base}?",
                    f"Como está {base}?",
                    f"Estado de {base}?",
                    f"Situação de {base}?",
                    f"Status: {base}.",
                    f"Estado: {base}.",
                ])
            elif family == "INFINITIVE":
                phrase = self.rng.choice([
                    f"Verificar {attr_name} de {base}.",
                    f"Consultar o {attr_name} de {base}.",
                    f"Saber o {attr_name} de {base}.",
                ])
            elif family == "IMPERATIVE":
                phrase = f"Verifique {attr_name} de {base}."
            elif family == "INTERROGATIVE":
                phrase = f"Qual é {attr_name} de {base}?"
            elif family == "POLITE":
                phrase = f"Por favor, qual é {attr_name} de {base}?"
            elif family == "DECLARATIVE":
                phrase = f"Quero saber {attr_name} de {base}."
            elif family == "COLLOQUIAL":
                phrase = f"Como está {attr_name} de {base} aí?"
            elif family == "NEGATIVE":
                phrase = f"Não deixe de me dizer {attr_name} de {base}."
            elif family == "CONDITIONAL":
                phrase = f"Se puder, me diga {attr_name} de {base}."
            else:
                phrase = f"Que tal verificar {attr_name} de {base}?"
            return self._make_sample(phrase, intent, "NONE",
                                     [("DEVICE", device)] + ([('LOCATION', location)] if location else []), family)

        if intent == "SET_COLOR":
            color = self.rng.choice(COLORS_SINGLE)
            if OBJECTS[obj]["genero"] == "f" and color in COLOR_FEM:
                color = COLOR_FEM[color]
            if family == "SHORT":
                # Nunca gerar "Cor da luz: azul" como SET_COLOR: falta ação.
                phrase = self.rng.choice([
                    f"{device}: colocar em {color}.",
                    f"{device}: mudar para {color}.",
                    f"Cor da {device}: mudar para {color}.",
                ])
            elif family == "INFINITIVE":
                phrase = self.rng.choice([
                    f"Colocar {obj_nom} em {color}.",
                    f"Mudar {obj_nom} para {color}.",
                    f"Deixar {obj_nom} {color}.",
                ])
            elif family == "IMPERATIVE":
                phrase = f"Coloque {obj_nom} em {color}."
            elif family == "INTERROGATIVE":
                phrase = f"Pode colocar {obj_nom} em {color}?"
            elif family == "POLITE":
                phrase = f"Por favor, coloque {obj_nom} em {color}."
            elif family == "DECLARATIVE":
                phrase = f"Quero {obj_nom} em {color}."
            elif family == "COLLOQUIAL":
                phrase = f"Deixa {obj_nom} em {color} aí."
            elif family == "NEGATIVE":
                phrase = f"Não deixe de colocar {obj_nom} em {color}."
            elif family == "CONDITIONAL":
                phrase = f"Se puder, deixe {obj_nom} em {color}."
            else:
                phrase = f"Que tal deixar {obj_nom} em {color}?"
            return self._make_sample(phrase, intent, "SET",
                                     [("DEVICE", device), ("COLOR", color)] + ([('LOCATION', location)] if location else []), family)

        # Intenções numéricas. O valor é obrigatório apenas em SET/INCREASE/DECREASE.
        attr_phrase = self.synt.atributo_intent(obj, loc, intent, device, location)
        attr = NUMERIC_ATTR[intent]
        if op == "SET":
            value = self.synt.valor_str(obj)
            if family == "INFINITIVE":
                phrase = self.rng.choice([
                    f"Definir {attr_phrase} em {value}.",
                    f"Ajustar {attr_phrase} para {value}.",
                    f"Regular {attr_phrase} em {value}.",
                ])
            elif family == "SHORT":
                # Formato curto, deliberadamente <= 4 palavras.
                short_attr = {
                    "SET_BRIGHTNESS": "Brilho",
                    "SET_SPEED": "Velocidade",
                    "SET_TEMPERATURE": "Temperatura",
                    "SET_VOLTAGE": "Voltagem",
                    "SET_VOLUME": "Volume",
                }[intent]
                phrase = self.rng.choice([
                    f"{short_attr} {device}: {value}.",
                    f"{short_attr} da {device}: {value}.",
                    f"{device}: {short_attr} {value}.",
                    f"{short_attr}: {value} na {device}.",
                    f"{short_attr} em {device}: {value}.",
                ])
            elif family == "IMPERATIVE":
                phrase = f"Defina {attr_phrase} em {value}."
            elif family == "INTERROGATIVE":
                phrase = f"Pode definir {attr_phrase} em {value}?"
            elif family == "POLITE":
                phrase = f"Por favor, ajuste {attr_phrase} para {value}."
            elif family == "DECLARATIVE":
                phrase = f"Quero {attr_phrase} em {value}."
            elif family == "COLLOQUIAL":
                phrase = f"Deixa {attr_phrase} em {value} aí."
            elif family == "NEGATIVE":
                # NEGATIVE com SET seria semanticamente ambígua; o balanceador evita SET.
                return self._direct_family_seed(intent, "NEGATIVE") if False else None
            elif family == "CONDITIONAL":
                phrase = f"Se puder, deixe {attr_phrase} em {value}."
            else:
                phrase = f"Que tal colocar {attr_phrase} em {value}?"
        else:
            increase = op == "INCREASE"
            verb_map = {
                "SET_TEMPERATURE": ("aumente", "aumentar", "diminua", "diminuir"),
                "SET_SPEED": ("aumente", "aumentar", "reduza", "reduzir"),
                "SET_BRIGHTNESS": ("aumente", "aumentar", "diminua", "diminuir"),
                "SET_VOLUME": ("aumente", "aumentar", "abaixe", "abaixar"),
                "SET_VOLTAGE": ("aumente", "aumentar", "reduza", "reduzir"),
            }[intent]
            imp, inf, dimp, dinf = verb_map
            verb, infinitive = (imp, inf) if increase else (dimp, dinf)
            value = self.synt.valor_str(obj)
            if family == "INFINITIVE":
                phrase = f"{infinitive.capitalize()} {attr_phrase} para {value}."
            elif family == "SHORT":
                short_attr = {
                    "SET_BRIGHTNESS": "brilho",
                    "SET_SPEED": "velocidade",
                    "SET_TEMPERATURE": "temperatura",
                    "SET_VOLTAGE": "voltagem",
                    "SET_VOLUME": "volume",
                }[intent]
                prefix = "Mais" if increase else "Menos"
                phrase = self.rng.choice([
                    f"{prefix} {short_attr} {device}: {value}.",
                    f"{prefix} {short_attr} da {device}: {value}.",
                    f"{device}: {prefix.lower()} {short_attr} {value}.",
                    f"{prefix} {short_attr}: {value} na {device}.",
                    f"{device}: {prefix.lower()} {short_attr} em {value}.",
                ])
            elif family == "IMPERATIVE":
                phrase = f"{verb.capitalize()} {attr_phrase} em {value}."
            elif family == "INTERROGATIVE":
                phrase = f"Você pode {infinitive} {attr_phrase} em {value}?"
            elif family == "POLITE":
                phrase = f"Por favor, {verb} {attr_phrase} em {value}."
            elif family == "DECLARATIVE":
                phrase = f"Eu quero {infinitive} {attr_phrase} em {value} agora."
            elif family == "COLLOQUIAL":
                phrase = f"{verb.capitalize()} {attr_phrase} em {value} aí."
            elif family == "NEGATIVE":
                phrase = f"Não deixe de {infinitive} {attr_phrase} em {value}."
            elif family == "CONDITIONAL":
                phrase = f"Se puder, {verb} {attr_phrase} em {value}."
            else:
                phrase = f"Que tal {infinitive} {attr_phrase} em {value} agora?"
        specs = [("DEVICE", device), ("MEASURE", value)] + ([('LOCATION', location)] if location else [])
        return self._make_sample(phrase, intent, op, specs, family)

    def _seed_matrix(self, intents: List[str]) -> Dict[str, List[Dict]]:
        matrix = {i: [] for i in intents}
        for intent in intents:
            for family in LINGUISTIC_FAMILIES:
                got = None
                for _ in range(12):
                    got = self._direct_family_seed(intent, family)
                    if got:
                        break
                if not got:
                    for _ in range(20):
                        got = self._candidate(intent, family=family)
                        if got:
                            break
                if got:
                    matrix[intent].append(got)
                    self.audit["seed_matrix"][f"{intent}:{family}"] += 1
                else:
                    self.audit["rejections"][f"seed_missing:{intent}:{family}"] += 1
        return matrix

    def _append_unique(self, out: List[Dict], s: Optional[Dict], seen: set) -> bool:
        if not s:
            return False
        key = s["text"].casefold()
        if key in seen or key in self.global_seen:
            self.audit["duplicates"] += 1
            return False
        seen.add(key)
        self.global_seen.add(key)
        out.append(s)
        self.audit["family_counts"][s.get("family", detect_linguistic_family(s["text"]))] += 1
        return True

    def _pair_fill(self, outputs: Dict[str, List[Dict]], targets: Dict[str, int],
                   pair_seen: set) -> None:
        pair_index = 0
        progress = True
        while progress:
            progress = False
            for a, b in self.CONTRAST_PAIRS:
                if a not in outputs or b not in outputs:
                    continue
                if len(outputs[a]) >= targets[a] and len(outputs[b]) >= targets[b]:
                    continue

                common = set(objects_for_intent(a)).intersection(objects_for_intent(b))
                if not common:
                    continue
                obj = self.rng.choice(sorted(common))
                common_locs = [x for x in combos_for(obj)
                               if valid_combo(obj, x)]
                loc = self.rng.choice(common_locs) if common_locs else None

                for intent in (a, b):
                    if len(outputs[intent]) >= targets[intent]:
                        continue
                    op = "NONE" if intent not in NUMERIC_INTENTS else None
                    s = self._candidate(intent, force_obj=obj, force_loc=loc, force_op=op)
                    if not s:
                        continue
                    pair_key = (pair_index, a, b, obj, loc, s["text"].casefold())
                    if pair_key in pair_seen:
                        continue
                    pair_seen.add(pair_key)
                    if self._append_unique(outputs[intent], s, self.intent_seen.setdefault(intent, set())):
                        self.audit["pair_blocks"][f"{a}__{b}"] += 1
                        progress = True
                pair_index += 1

            if all(len(outputs[i]) >= targets[i] for i in outputs):
                break
            if pair_index > 5000:
                break

    def _balanced_fill(self, outputs: Dict[str, List[Dict]], target: int) -> Dict[str, int]:
        """Preenchimento tolerante: se uma intenção esgotar candidatos, registra o déficit e segue.

        A meta continua sendo ``target``; ela deixa de ser uma invariável fatal.
        Isso permite que uma intenção termine com menos amostras quando não houver
        novas combinações válidas, sem bloquear a geração das demais intenções.
        """
        shortages: Dict[str, int] = {}
        for intent, items in outputs.items():
            if len(items) >= target:
                continue
            guard = 0
            stagnant = 0
            max_guard = max(120, target * 30)
            while len(items) < target and guard < max_guard:
                guard += 1
                before = len(items)
                family = min(LINGUISTIC_FAMILIES, key=lambda f: sum(
                    1 for x in items if x.get("family") == f))
                s = self._direct_family_seed(intent, family) or self._candidate(
                    intent, family=family, max_attempts=5)
                if self._append_unique(items, s, self.intent_seen.setdefault(intent, set())):
                    stagnant = 0
                    continue
                stagnant += 1
                # Muitas tentativas consecutivas sem uma única amostra nova
                # significam que as combinações válidas desta intenção foram
                # esgotadas na prática. Não insistir indefinidamente.
                if len(items) == before and stagnant >= max(80, len(LINGUISTIC_FAMILIES) * 12):
                    break
            if len(items) < target:
                shortages[intent] = target - len(items)
                self.audit["shortage_intent"][intent] += target - len(items)
        return shortages

    def generate_intent(self, intent: str, target: int) -> List[Dict]:
        outputs = {intent: []}
        seed = self._seed_matrix([intent])[intent]
        seen = set()
        for s in seed[:target]:
            self._append_unique(outputs[intent], s, seen)
        pair_target = max(len(outputs[intent]), int(target * self.cfg.pair_fraction))
        pairs = [(a,b) for a,b in self.CONTRAST_PAIRS if intent in (a,b)]
        if pairs:
            for _ in range(max(1, len(pairs))):
                if len(outputs[intent]) >= pair_target:
                    break
                a,b = pairs[_ % len(pairs)]
                other = b if intent == a else a
                common = set(objects_for_intent(intent)).intersection(objects_for_intent(other))
                if common:
                    obj=self.rng.choice(sorted(common))
                    loc=self.rng.choice(combos_for(obj))
                    s=self._candidate(intent, force_obj=obj, force_loc=loc)
                    self._append_unique(outputs[intent],s,seen)
        self._balanced_fill(outputs, target)
        self.rng.shuffle(outputs[intent])
        return outputs[intent][:target]

    def _ensure_temporal_ratio(self, outputs: Dict[str, List[Dict]], target: int) -> None:
        desired = int(round(target * self.cfg.temporal_ratio))
        for intent, items in outputs.items():
            if desired <= 0:
                continue
            def has_temp(sample: Dict) -> bool:
                return any(e.get("type") in TEMPORAL_ENTITY_TYPES for e in sample.get("entities", []))
            current = sum(1 for x in items if has_temp(x))
            attempts = 0
            while current < desired and attempts < 500:
                attempts += 1
                idx = next((i for i, x in enumerate(items)
                            if not x.get("protected_primary") and not has_temp(x)), None)
                if idx is None:
                    break
                family = items[idx].get("family")
                candidate = self._candidate(intent, family=family, max_attempts=30)
                if not candidate or not has_temp(candidate):
                    continue
                old = items[idx]
                old_key = old["text"].casefold()
                new_key = candidate["text"].casefold()
                if new_key in self.global_seen:
                    continue
                self.global_seen.discard(old_key)
                self.intent_seen.setdefault(intent, set()).discard(old_key)
                old.clear()
                old.update(candidate)
                self.global_seen.add(new_key)
                self.intent_seen.setdefault(intent, set()).add(new_key)
                current += 1

    def _family_pair(self, intent: str, family: str) -> List[Dict]:
        objs = objects_for_intent(intent)
        if not objs:
            return []
        obj = self.rng.choice(objs)
        locs = combos_for(obj)
        loc = self.rng.choice(locs) if locs else None
        out: List[Dict] = []
        local_seen = self.intent_seen.setdefault(intent, set())

        for _ in range(60):
            if len(out) >= 2:
                break
            s = self._candidate(
                intent,
                family=family,
                force_obj=obj,
                force_loc=loc,
                max_attempts=20,
            )
            if not s:
                continue
            key = s["text"].casefold()
            if key in local_seen or key in {x["text"].casefold() for x in out}:
                continue
            out.append(s)
        return out

    # ======================== NOVO MÉTODO OTIMIZADO ========================
    def _hierarchical_candidate(self, intent: str, family: str, used_structures: set,
                               used_diversity: set, force_op=None):
        """Escolhe candidatos pela regra: estrutura nova -> combinação nova -> texto novo."""
        best = None
        best_score = None
        for attempt in range(40):
            planned_op = force_op
            if planned_op is None and intent in NUMERIC_INTENTS:
                planned_op = self._choose_operation_balanced(intent, family)
            s = None
            for special_category in self._special_category_order(intent):
                s = self._direct_family_seed(intent, family, force_op=planned_op, special_category=special_category)
                if s:
                    break
            if not s:
                s = self._direct_family_seed(intent, family, force_op=planned_op)
            if not s:
                s = self._candidate(intent, family=family, force_op=planned_op, max_attempts=4)
            if not s:
                continue
            txt = s.get("text", "")
            if not txt:
                continue
            struct = _v6311_structural_signature(s)
            div = _v6311_diversity_key(s)
            key = txt.casefold()
            if key in self.intent_seen.get(intent, set()) or key in self.global_seen:
                continue
            # prioridade absoluta: estrutura nunca usada nesta intenção/família;
            # depois combinação estrutural + entidades/operação/temporal.
            score = (int(struct in used_structures), int(div in used_diversity), attempt)
            if best is None or score < best_score:
                best, best_score = s, score
                if score[0] == 0 and score[1] == 0:
                    break
        return best

    # V63.19 — ÂNCORAS OBRIGATÓRIAS DE FRONTEIRA SEMÂNTICA
    CRITICAL_ANCHORS_V63_19 = {
        "TURN_ON": [("Ligue {DEVICE}.","IMPERATIVE"),("Deixe {DEVICE} ligado.","DECLARATIVE"),("Quero {DEVICE} ligado.","DECLARATIVE"),("Mantenha {DEVICE} aceso.","IMPERATIVE")],
        "TURN_OFF": [("Desligue {DEVICE}.","IMPERATIVE"),("Deixe {DEVICE} desligado.","DECLARATIVE"),("Quero {DEVICE} desligado.","DECLARATIVE"),("Mantenha {DEVICE} apagado.","IMPERATIVE")],
        "START": [("Comece {DEVICE}.","IMPERATIVE"),("Inicie {DEVICE}.","IMPERATIVE"),("Dê partida em {DEVICE}.","IMPERATIVE"),("Comece o ciclo de {DEVICE}.","DECLARATIVE"),("Inicie o ciclo de {DEVICE}.","DECLARATIVE")],
        "STOP": [("Pare {DEVICE}.","IMPERATIVE"),("Interrompa {DEVICE}.","IMPERATIVE"),("Pause {DEVICE}.","IMPERATIVE"),("Pare o ciclo de {DEVICE}.","DECLARATIVE")],
        "GET_STATUS": [("Como está {DEVICE}?","INTERROGATIVE"),("Qual é o estado de {DEVICE}?","INTERROGATIVE"),("Me diga se {DEVICE} está ligado.","INTERROGATIVE"),("Me diga se {DEVICE} está desligado.","INTERROGATIVE"),("Me diga se {DEVICE} está aberto.","INTERROGATIVE"),("Me diga se {DEVICE} está fechado.","INTERROGATIVE"),("Quero saber se {DEVICE} está funcionando.","INTERROGATIVE")],
        "OPEN": [("Abra {DEVICE}.","IMPERATIVE"),("Destranque {DEVICE}.","IMPERATIVE"),("Deixe {DEVICE} aberto.","DECLARATIVE")],
        "CLOSE": [("Feche {DEVICE}.","IMPERATIVE"),("Tranque {DEVICE}.","IMPERATIVE"),("Deixe {DEVICE} fechado.","DECLARATIVE")],
        "SET_SPEED": [("Coloque {DEVICE} mais rápido.","IMPERATIVE"),("Deixe {DEVICE} mais rápido.","DECLARATIVE"),("Aumente a velocidade de {DEVICE}.","IMPERATIVE"),("Reduza a velocidade de {DEVICE}.","IMPERATIVE"),("Acelere {DEVICE}.","IMPERATIVE"),("Desacelere {DEVICE}.","IMPERATIVE"),("Deixe {DEVICE} mais devagar.","DECLARATIVE")],
        "SET_VOLTAGE": [("Aumente a voltagem de {DEVICE}.","IMPERATIVE"),("Reduza a voltagem de {DEVICE}.","IMPERATIVE"),("Aumente a tensão de {DEVICE}.","IMPERATIVE"),("Reduza a tensão de {DEVICE}.","IMPERATIVE"),("Coloque {DEVICE} em 220 volts.","IMPERATIVE")],
        "SET_BRIGHTNESS": [("Aumente o brilho de {DEVICE}.","IMPERATIVE"),("Reduza o brilho de {DEVICE}.","IMPERATIVE"),("Aumente a luminosidade de {DEVICE}.","IMPERATIVE"),("Deixe {DEVICE} mais forte.","DECLARATIVE"),("Deixe {DEVICE} mais fraco.","DECLARATIVE"),
            ("Clareie {LOCATION_ARTICLE}.","IMPERATIVE"),("Escureça {LOCATION_ARTICLE}.","IMPERATIVE"),
            ("Clareia {LOCATION_ARTICLE}.","IMPERATIVE"),("Escurece {LOCATION_ARTICLE}.","IMPERATIVE"),
            ("Deixe {LOCATION_ARTICLE} mais claro.","DECLARATIVE"),("Deixe {LOCATION_ARTICLE} mais escuro.","DECLARATIVE")],
        "SET_COLOR": [("Coloque {DEVICE} em {COLOR}.","IMPERATIVE"),("Mude a cor de {DEVICE} para {COLOR}.","IMPERATIVE"),("Deixe {DEVICE} {COLOR}.","DECLARATIVE"),("Configure {DEVICE} na cor {COLOR}.","IMPERATIVE")],
        "SET_TEMPERATURE": [("Esfrie {DEVICE}.","IMPERATIVE"),("Aqueça {DEVICE}.","IMPERATIVE"),("Aumente a temperatura de {DEVICE}.","IMPERATIVE"),("Reduza a temperatura de {DEVICE}.","IMPERATIVE")],
        "SET_VOLUME": [("Aumente o volume de {DEVICE}.","IMPERATIVE"),("Abaixe o volume de {DEVICE}.","IMPERATIVE")],
    }

    def _insert_critical_anchors_v6319(self, outputs):
        contexts={"TURN_ON":("luz","sala"),"TURN_OFF":("luz","sala"),"START":("lavadora","banheiro"),"STOP":("ventilador","cozinha"),"GET_STATUS":("porta","garagem"),"OPEN":("porta","garagem"),"CLOSE":("porta","garagem"),"SET_SPEED":("ventilador","cozinha"),"SET_VOLTAGE":("fonte de bancada",None),"SET_BRIGHTNESS":("lâmpada","sala"),"SET_COLOR":("luz","cozinha"),"SET_TEMPERATURE":("ar condicionado","sala"),"SET_VOLUME":("tv","sala")}
        for intent,templates in self.CRITICAL_ANCHORS_V63_19.items():
            obj,loc=contexts[intent]
            if obj not in objects_for_intent(intent): continue
            if loc is not None and loc not in combos_for(obj): loc=combos_for(obj)[0] if combos_for(obj) else None
            device=surface_device(obj,self.rng,"base"); location=surface_location(loc,self.rng,"base") if loc else None
            loc_base = location or loc or "sala"
            loc_article = ("a " if LOCATIONS.get(loc, "f") == "f" else "o ") + loc_base
            for template,family in templates:
                if len(outputs[intent])>=self.cfg.samples_per_intent: break
                color=self.rng.choice(COLORS_SINGLE) if intent=="SET_COLOR" else None
                loc_base = location or loc or "sala"
                loc_article = ("a " if LOCATIONS.get(loc, "f") == "f" else "o ") + loc_base
                phrase=template.format(DEVICE=self.synt.frase_nominal(obj,loc,True,False,device,location),
                                        LOCATION=loc_base, LOCATION_ARTICLE=loc_article,
                                        COLOR=color or "azul")
                operation=infer_operation_from_text(phrase,intent) if intent in NUMERIC_INTENTS else "NONE"
                if operation not in COMPATIBLE_OPERATIONS[intent]: continue
                specs=[("DEVICE",device)]+([("LOCATION",location)] if location else [])
                if intent=="SET_COLOR": specs.append(("COLOR",color or "azul"))
                sample=self._make_sample(phrase,intent,operation,specs,family,"SEMANTIC_ANCHOR")
                if sample:
                    sample["protected_semantic_anchor"] = True
                if sample and sample["text"].casefold() not in self.global_seen and sample["text"].casefold() not in self.intent_seen[intent]:
                    self._append_unique(outputs[intent],sample,self.intent_seen[intent]); self._family_counts_by_intent[intent][family]+=1; self.audit["semantic_anchors"]+=1


    def _semantic_anchor_candidate_v6322(self, intent: str, family: str) -> Optional[Dict]:
        """Gera uma âncora semântica balanceável usando os moldes críticos da intenção.
        A âncora reforça o núcleo da intenção, mas não cria uma nova intenção.
        """
        templates = self.CRITICAL_ANCHORS_V63_19.get(intent, [])
        if not templates:
            return None
        obj_candidates = objects_for_intent(intent)
        if not obj_candidates:
            return None
        obj = self.rng.choice(obj_candidates)
        locs = combos_for(obj)
        loc = self.rng.choice(locs) if locs else None
        comp = self.rng.choice(("base", "possessive", "demonstrative"))
        device = surface_device(obj, self.rng, comp)
        location = surface_location(loc, self.rng, comp) if loc else None
        template, template_family = self.rng.choice(templates)
        # A família declarada pelo molde crítico é preservada para não criar
        # falsos positivos de família linguística.
        family = template_family
        color = self.rng.choice(COLORS_SINGLE)
        if obj in OBJECTS and OBJECTS[obj].get("genero") == "f" and color in COLOR_FEM:
            color = COLOR_FEM[color]
        loc_base = location or loc or "sala"
        loc_article = ("a " if LOCATIONS.get(loc, "f") == "f" else "o ") + loc_base
        phrase = template.format(DEVICE=self.synt.frase_nominal(obj, loc, True, False, device, location),
                                 LOCATION=loc_base,
                                 LOCATION_ARTICLE=loc_article,
                                 COLOR=color)
        if intent in NUMERIC_INTENTS:
            operation = infer_operation_from_text(phrase, intent)
            if operation not in COMPATIBLE_OPERATIONS[intent]:
                operation = "SET" if "SET" in COMPATIBLE_OPERATIONS[intent] else "INCREASE"
        elif intent == "SET_COLOR":
            operation = "SET"
        else:
            operation = "NONE"
        specs = [("DEVICE", device)]
        if location and location in phrase:
            specs.append(("LOCATION", location))
        if intent == "SET_COLOR" and color in phrase:
            specs.append(("COLOR", color))
        sample = self._make_sample(phrase, intent, operation, specs, family, "SEMANTIC_ANCHOR")
        if sample:
            sample["protected_semantic_anchor"] = True
        return sample

    def generate_all(self) -> List[Dict]:
        """
        Geração em matriz determinística:
        - exatamente o mesmo total por intenção;
        - todas as famílias em todas as intenções;
        - distribuição de família com diferença máxima de 1;
        - operações numéricas balanceadas;
        - composições balanceadas;
        - combinações dispositivo/local sempre filtradas pela ontologia;
        - temporais ajustados depois sem alterar a família;
        - validação antes de inserir qualquer amostra.
        """
        self.global_seen.clear()
        self.intent_seen = {i: set() for i in INTENT_MAP}
        self._op_counts = {i: Counter() for i in INTENT_MAP}
        self._op_targets = {
            i: _balanced_targets(self.cfg.samples_per_intent, ("SET", "INCREASE", "DECREASE"))
            for i in NUMERIC_INTENTS
        }
        self._composition_counts = {i: Counter() for i in INTENT_MAP}
        self._family_counts_by_intent = {i: Counter() for i in INTENT_MAP}
        self._op_schedule = {}
        self._special_counts = {i: Counter() for i in INTENT_MAP}
        # V63.21: déficits são métricas de cobertura, não exceções fatais.
        self.audit["shortage_intent"] = Counter()
        self.audit["shortage_family"] = Counter()
        self.audit["shortage_temporal"] = Counter()
        self.audit["shortage_schedule"] = Counter()
        self.audit["missing_family"] = Counter()
        self.audit["partial_families"] = Counter()
        special_total = (self.cfg.semantic_paraphrase_ratio + self.cfg.colloquial_noisy_ratio +
                         self.cfg.entity_negative_ratio + self.cfg.contrastive_boundary_ratio)
        if not 0.0 <= special_total <= 0.80:
            raise GenerationInvariantError("Soma das categorias especiais deve estar entre 0 e 0.80")
        self._special_targets = {}
        for i in INTENT_MAP:
            n = self.cfg.samples_per_intent
            self._special_targets[i] = {
                "SEMANTIC_PARAPHRASE": int(round(n * self.cfg.semantic_paraphrase_ratio)),
                "COLLOQUIAL_NOISY": int(round(n * self.cfg.colloquial_noisy_ratio)),
                "ENTITY_NEGATIVE": int(round(n * self.cfg.entity_negative_ratio)),
                "CONTRASTIVE_BOUNDARY": int(round(n * self.cfg.contrastive_boundary_ratio)),
            }
        # V63.22: quotas independentes para as duas categorias que antes
        # dependiam de poucas sementes. Elas são balanceadas por intenção.
        self._semantic_anchor_targets = {
            i: int(round(self.cfg.samples_per_intent * self.cfg.semantic_anchor_ratio))
            for i in INTENT_MAP
        }
        self._explicit_schedule_targets = {
            i: int(round(self.cfg.samples_per_intent * self.cfg.explicit_schedule_ratio))
            for i in INTENT_MAP
        }
        for i in NUMERIC_INTENTS:
            ops = []
            for op, n in self._op_targets[i].items():
                ops.extend([op] * n)
            self.rng.shuffle(ops)
            self._op_schedule[i] = ops

        intents = list(INTENT_MAP)
        family_order = tuple(LINGUISTIC_FAMILIES)
        outputs: Dict[str, List[Dict]] = {i: [] for i in intents}
        per_family = {f: self.cfg.samples_per_intent // len(family_order)
                      for f in family_order}
        remainder = self.cfg.samples_per_intent % len(family_order)
        # Resto distribuído de forma fixa para que a mesma seed seja reproduzível.
        for i in range(remainder):
            per_family[family_order[i]] += 1

        total_target = len(intents) * self.cfg.samples_per_intent
        pbar = tqdm(total=total_target, desc="Gerando dataset", unit="amostra") if tqdm else None

        try:
            # 0) ÂNCORAS SEMÂNTICAS OBRIGATÓRIAS
            self._insert_critical_anchors_v6319(outputs)
            if pbar:
                pbar.update(self.audit.get("semantic_anchors", 0))

            # 0.1) V63.22 — balanceamento real de SEMANTIC_ANCHOR.
            # A quota é por intenção. Se uma intenção esgotar moldes válidos,
            # registra o déficit e segue para a próxima.
            for intent in intents:
                target = self._semantic_anchor_targets[intent]
                attempts = 0
                while self._special_counts[intent]["SEMANTIC_ANCHOR"] < target and attempts < max(80, target * 40):
                    attempts += 1
                    family = self.rng.choice(tuple(LINGUISTIC_FAMILIES))
                    s = self._semantic_anchor_candidate_v6322(intent, family)
                    if not s:
                        continue
                    if self._append_unique(outputs[intent], s, self.intent_seen[intent]):
                        self._special_counts[intent]["SEMANTIC_ANCHOR"] += 1
                        self._family_counts_by_intent[intent][s["family"]] += 1
                        self.audit["semantic_anchors"] += 1
                        if pbar:
                            pbar.update(1)
                if self._special_counts[intent]["SEMANTIC_ANCHOR"] < target:
                    self.audit.setdefault("shortage_semantic_anchor", Counter())[intent] += (
                        target - self._special_counts[intent]["SEMANTIC_ANCHOR"]
                    )

            # 1) V63.22 — quota de programação explícita por intenção.
            # Não é mais apenas uma única semente global por intenção.
            if self.cfg.temporal_ratio > 0:
                for intent in intents:
                    target = self._explicit_schedule_targets[intent]
                    attempts = 0
                    while self._special_counts[intent]["explicit_schedule"] < target and attempts < max(80, target * 40):
                        attempts += 1
                        planned_op = self._choose_operation_balanced(intent, "IMPERATIVE") if intent in NUMERIC_INTENTS else None
                        scheduled = self._explicit_schedule_seed(intent, planned_op)
                        if scheduled and self._append_unique(outputs[intent], scheduled, self.intent_seen[intent]):
                            self._special_counts[intent]["explicit_schedule"] += 1
                            self.audit["explicit_schedules"] += 1
                            self._family_counts_by_intent[intent][scheduled["family"]] += 1
                            if pbar:
                                pbar.update(1)
                    if self._special_counts[intent]["explicit_schedule"] < target:
                        self.audit["shortage_schedule"][intent] += (
                            target - self._special_counts[intent]["explicit_schedule"]
                        )
                        logger.warning(
                            "Programação explícita abaixo da quota para %s; seguindo para a próxima intenção.",
                            intent,
                        )

            # 2) DEPOIS: crescimento circular. Em cada rodada cada família
            # recebe no máximo UMA amostra por intenção. Assim a diversidade
            # aparece primeiro e só depois as famílias são multiplicadas.
            progress = True
            while progress and any(len(outputs[i]) < self.cfg.samples_per_intent for i in intents):
                progress = False
                for family in family_order:
                    for intent in intents:
                        current = sum(1 for x in outputs[intent] if x.get("family") == family)
                        needed = per_family[family]
                        if current >= needed or len(outputs[intent]) >= self.cfg.samples_per_intent:
                            continue
                        attempts = 0
                        inserted = False
                        while attempts < max(24, needed * 8):
                            attempts += 1
                            # V63.11: cobertura hierárquica. Primeiro uma estrutura nova,
                            # depois uma combinação nova de entidades/operação/temporal,
                            # e só então reutilização controlada do molde.
                            used_structures = {
                                _v6311_structural_signature(x) for x in outputs[intent]
                                if x.get("family") == family
                            }
                            used_diversity = {
                                _v6311_diversity_key(x) for x in outputs[intent]
                                if x.get("family") == family
                            }
                            planned_op = self._choose_operation_balanced(intent, family) if intent in NUMERIC_INTENTS else None
                            s = self._hierarchical_candidate(intent, family, used_structures,
                                                             used_diversity, force_op=planned_op)
                            if self._append_unique(outputs[intent], s, self.intent_seen[intent]):
                                cat = s.get("category", "generative")
                                if cat in self._special_targets.get(intent, {}):
                                    self._special_counts[intent][cat] += 1
                                self._family_counts_by_intent[intent][family] += 1
                                if pbar:
                                    pbar.update(1)
                                progress = True
                                inserted = True
                                break
                        if not inserted and current < needed:
                            # V63.21: uma família pode esgotar suas combinações válidas.
                            # Isso NÃO interrompe a intenção nem as próximas intenções.
                            self.audit["shortage_family"][f"{intent}:{family}"] += needed - current
                            continue

            # Preenche qualquer diferença causada por sementes/duplicatas.
            # Se uma intenção não conseguir chegar à meta, ela é encerrada com
            # o que foi possível gerar e a próxima intenção continua normalmente.
            self._balanced_fill(outputs, self.cfg.samples_per_intent)

            # Ajuste temporal exato por intenção, preservando família/operação.
            desired_temporal = int(round(self.cfg.samples_per_intent * self.cfg.temporal_ratio))
            if self.cfg.temporal_ratio > 0:
                desired_temporal = max(1, desired_temporal)
            for intent, items in outputs.items():
                def has_temp(x):
                    return any(e.get("type") in TEMPORAL_ENTITY_TYPES for e in x.get("entities", []))
                current = sum(has_temp(x) for x in items)
                if current < desired_temporal:
                    candidates = [x for x in items if not x.get("protected_primary") and not x.get("protected_semantic_anchor") and not has_temp(x)]
                    self.rng.shuffle(candidates)
                    for old in candidates:
                        if current >= desired_temporal:
                            break
                        for _ in range(30):
                            new = self._temporalize_sample(old)
                            if not new or has_temp(old):
                                continue
                            new_key = new["text"].casefold()
                            if new_key in self.global_seen:
                                continue
                            old_key = old["text"].casefold()
                            self.global_seen.discard(old_key)
                            self.intent_seen[intent].discard(old_key)
                            old.clear(); old.update(new)
                            self.global_seen.add(new_key)
                            self.intent_seen[intent].add(new_key)
                            current += 1
                            break
                if current != desired_temporal:
                    # V63.21: temporalidade é melhor esforço quando uma intenção
                    # ficou abaixo da meta. O déficit é apenas registrado.
                    self.audit["shortage_temporal"][intent] += max(0, desired_temporal - current)

            # Auditoria estrutural tolerante: verifica somente o que foi gerado.
            # Uma família ausente por esgotamento não é erro fatal; o relatório
            # mostra exatamente onde houve déficit.
            for intent in intents:
                for family in family_order:
                    fam_items = [x for x in outputs[intent] if x.get("family") == family]
                    if not fam_items:
                        self.audit["missing_family"][f"{intent}:{family}"] += 1
                        continue
                    if len(fam_items) > 1 and len({_v6311_structural_signature(x) for x in fam_items}) < 1:
                        raise GenerationInvariantError(f"Assinatura estrutural inválida: {intent}/{family}")

            data = [s for intent in intents for s in outputs[intent]]
            for intent in intents:
                generated = len(outputs[intent])
                if generated < self.cfg.samples_per_intent:
                    self.audit["shortage_intent"][intent] += self.cfg.samples_per_intent - generated
                counts = Counter(x.get("family") for x in outputs[intent])
                if generated and set(counts) != set(family_order):
                    self.audit["partial_families"][intent] += len(set(family_order) - set(counts))

            # V63.21: programação explícita é best-effort. Se não houver
            # combinação válida para uma intenção, registra o déficit e segue.
            if self.cfg.temporal_ratio > 0:
                for intent in intents:
                    if not any(x.get("category") == "explicit_schedule" for x in outputs[intent]):
                        self.audit["shortage_schedule"].setdefault(intent, 0)
                        self.audit["shortage_schedule"][intent] += 1

            self.rng.shuffle(data)
            return data
        finally:
            if pbar:
                pbar.close()

# ======================== SPLIT E EXPORTAÇÃO ========================

def split_stratified(data: List[Dict], ratios=(0.8, 0.1, 0.1), seed: int = 42):
    by_intent = {}
    for s in data:
        by_intent.setdefault(s["intent"], []).append(s)
    rng = random.Random(seed)
    train, val, test = [], [], []
    for intent, items in by_intent.items():
        rng.shuffle(items)
        n = len(items)
        n_train = int(n * ratios[0])
        n_val = int(n * ratios[1])
        train.extend(items[:n_train])
        val.extend(items[n_train:n_train + n_val])
        test.extend(items[n_train + n_val:])
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return {"train": train, "validation": val, "test": test}

def compact(items: List[Dict]) -> List[Dict]:
    out = []
    for s in items:
        out.append({
            "text": s["text"],
            "intent": INTENT_MAP[s["intent"]],
            "operation": OPERATION_MAP[s["operation"]],
            "entities": [{"start": e["start"], "end": e["end"],
                          "type": ENTITY_TYPE_MAP[e["type"]], "value": e["value"]}
                         for e in s["entities"]],
            "action_mode": ACTION_MODE_MAP[s.get("action_mode", "IMMEDIATE")],
            "target_scope": TARGET_SCOPE_MAP[s.get("target_scope", "SINGLE")],
            "value_type": VALUE_TYPE_MAP[s.get("value_type", "UNKNOWN")],
        })
    return out

def validate_ontology_invariants() -> None:
    independent = {"furadeira", "Makita", "soprador térmico"}
    for name in independent:
        if name not in OBJECTS:
            raise ValueError(f"Dispositivo independente ausente: {name}")
    if "bancada" not in LOCATIONS:
        raise ValueError("'bancada' deve ser LOCAL")
    if "bancada" in OBJECTS:
        raise ValueError("'bancada' não pode estar em OBJECTS")

GENERALIZATION_TEST = [
    ("liga a lâmpada do quarto", "TURN_ON"),
    ("quero a luz do quarto ligada", "TURN_ON"),
    ("deixa a luz do corredor acesa", "TURN_ON"),
    ("não quero mais a lâmpada do quarto ligada", "TURN_OFF"),
    ("tira a luz da sala", "TURN_OFF"),
    ("para o aspirador da garagem imediatamente", "STOP"),
    ("interrompe o ventilador da cozinha", "STOP"),
    ("começa a lavadora do banheiro", "START"),
    ("pode destravar a janela da sala?", "OPEN"),
    ("fecha bem a porta da garagem", "CLOSE"),
    ("esfria o ar condicionado da sala", "SET_TEMPERATURE"),
    ("qual a temperatura atual do ar condicionado do quarto", "GET_STATUS"),
    ("deixa a lâmpada da sala em 40 por cento", "SET_BRIGHTNESS"),
    ("coloca a luz da cozinha em vermelho", "SET_COLOR"),
    ("sobe a voltagem da fonte de bancada", "SET_VOLTAGE"),
    ("reduz a velocidade do exaustor do banheiro", "SET_SPEED"),
    ("quero deixar a luz do quarto ligada", "TURN_ON"),
    ("quero deixar a luz do quarto desligada", "TURN_OFF"),
    ("faz o aspirador da garagem parar", "STOP"),
    ("inicia a máquina da lavanderia", "START"),
    ("destrava a janela da sala", "OPEN"),
    ("trava a porta da garagem", "CLOSE"),
    ("esquenta o ar condicionado do quarto", "SET_TEMPERATURE"),
    ("coloca a luz em azul escuro", "SET_COLOR"),
    ("liga a luz", "TURN_ON"),
    ("desliga a tv", "TURN_OFF"),
    ("abre a janela", "OPEN"),
    ("fecha a porta", "CLOSE"),
    ("liga a furadeira aí", "TURN_ON"),
    ("liga a Makita da bancada", "TURN_ON"),
    ("para o soprador térmico", "STOP"),
    ("liga a furadeira da bancada", "TURN_ON"),
    ("você poderia, por favor, ligar a furadeira que está na bancada?", "TURN_ON"),
    ("será que dá para desligar o soprador térmico da minha bancada agora?", "TURN_OFF"),
    ("eu gostaria que você parasse a furadeira da oficina quando puder", "STOP"),
    ("como está a luz da cozinha", "GET_STATUS"),
    ("deixa a lâmpada da sala em 40%", "SET_BRIGHTNESS"),
    ("altera a voltagem da fonte de bancada para 12V", "SET_VOLTAGE"),
    ("verifica o nível da bateria da garagem", "GET_STATUS"),
    ("liga a Makita Skil", "TURN_ON"),
    ("liga a Makita e a Skil", "TURN_ON"),
    ("liga o soprador térmico da bancada", "TURN_ON"),
    ("mais luz na sala", "SET_BRIGHTNESS"),
    ("quero mais claridade na sala", "SET_BRIGHTNESS"),
    ("pode deixar a sala mais iluminada", "SET_BRIGHTNESS"),
    ("clareia a sala", "SET_BRIGHTNESS"),
    ("escurece o quarto", "SET_BRIGHTNESS"),
    ("menos luz no quarto", "SET_BRIGHTNESS"),
    ("deixa a luz mais fraca", "SET_BRIGHTNESS"),
    ("coloca a luz da sala em 80%", "SET_BRIGHTNESS"),
    ("muda a cor da luz da sala para azul", "SET_COLOR"),
    ("escurece o quarto", "SET_BRIGHTNESS"),
    ("clareia a sala", "SET_BRIGHTNESS"),
    ("escurece o quarto", "SET_BRIGHTNESS"),
    ("aumenta a velocidade do ventilador", "SET_SPEED"),
    ("diminua o volume da televisão", "SET_VOLUME"),
]


def audit_generated_dataset(data: List[Dict], cfg: Config) -> Dict[str, Any]:
    problems = Counter()
    seen = set()
    by_intent = Counter()
    by_family_intent = Counter()
    by_operation = Counter()
    temporal_count = 0
    entity_counts = Counter()
    action_modes = Counter()
    target_scopes = Counter()
    value_types = Counter()
    categories = Counter()
    category_by_intent = Counter()

    for s in data:
        key = s["text"].casefold()
        if key in seen:
            problems["duplicate"] += 1
        seen.add(key)
        by_intent[s["intent"]] += 1
        cat = s.get("category", "generative")
        categories[cat] += 1
        category_by_intent[f"{s['intent']}::{cat}"] += 1
        fam = s.get("family", detect_linguistic_family(s["text"]))
        by_family_intent[f"{s['intent']}::{fam}"] += 1
        by_operation[s.get("operation", "NONE")] += 1
        if any(e.get("type") in TEMPORAL_ENTITY_TYPES for e in s.get("entities", [])):
            temporal_count += 1
            if not temporal_entities_are_unambiguous(s["text"], s.get("entities", [])):
                problems["ambiguous_temporal_entity"] += 1
        for e in s.get("entities", []):
            entity_counts[e["type"]] += 1

        action_modes[s.get("action_mode", "IMMEDIATE")] += 1
        target_scopes[s.get("target_scope", "SINGLE")] += 1
        value_types[s.get("value_type", "UNKNOWN")] += 1

        if not s.get("_validated", False) and not validate_sample(s):
            problems["invalid"] += 1

        for e in s.get("entities", []):
            if not (0 <= int(e["start"]) < int(e["end"]) <= len(s["text"])):
                problems["bad_span"] += 1
            elif s["text"][int(e["start"]):int(e["end"])] != e["value"]:
                problems["bad_span_value"] += 1

    # V63.21: cobertura abaixo da meta não é falha estrutural.
    # O relatório passa a separar "problemas" reais de "déficits de cobertura".
    missing_intents = [i for i in INTENT_MAP if i not in by_intent]
    shortage_intents = {
        i: max(0, cfg.samples_per_intent - by_intent[i])
        for i in INTENT_MAP
        if by_intent[i] < cfg.samples_per_intent
    }

    seed_floor = cfg.seed_per_family
    missing_family = []
    family_shortages = {}
    for intent in [i for i in INTENT_MAP]:
        for family in LINGUISTIC_FAMILIES:
            got = by_family_intent[f"{intent}::{family}"]
            if got < seed_floor:
                missing_family.append(f"{intent}:{family}")
                family_shortages[f"{intent}:{family}"] = max(0, seed_floor - got)

    boundary = run_intent_boundary_tests()
    contrast = run_contrastive_tests()

    generalization_collisions = []
    keys = {s["text"].casefold() for s in data}
    for phrase, expected in GENERALIZATION_TEST:
        if phrase.casefold() in keys:
            generalization_collisions.append(phrase)
    if generalization_collisions:
        problems["generalization_collision"] = len(generalization_collisions)

    semantic_pairs = [
        ("TURN_ON", "TURN_OFF"), ("START", "STOP"), ("OPEN", "CLOSE"),
        ("TURN_ON", "START"), ("TURN_OFF", "STOP"),
        ("GET_STATUS", "SET_TEMPERATURE"), ("GET_STATUS", "SET_SPEED"),
        ("GET_STATUS", "SET_BRIGHTNESS"), ("GET_STATUS", "SET_VOLUME"),
        ("GET_STATUS", "SET_VOLTAGE"), ("GET_STATUS", "SET_COLOR"),
        ("SET_BRIGHTNESS", "SET_COLOR"), ("SET_BRIGHTNESS", "TURN_ON"),
        ("SET_BRIGHTNESS", "TURN_OFF"),
    ]
    pair_report = {}
    for a,b in semantic_pairs:
        pair_report[f"{a}__{b}"] = {
            "a": by_intent[a], "b": by_intent[b],
            "same_entity_examples": sum(
                1 for s in data
                if s["intent"] in (a,b) and any(e["type"]=="DEVICE" for e in s["entities"])
            )
        }

    if not boundary["ok"]:
        problems["intent_boundary_tests_failed"] = sum(1 for x in boundary["cases"] if not x["ok"])
    if not contrast["ok"]:
        problems["contrastive_tests_failed"] = sum(1 for x in contrast["cases"] if not x["accepted"])
    ok = not problems
    report = {
        "ok": ok,
        "problems": dict(problems),
        "total": len(data),
        "por_intencao": dict(by_intent),
        "por_familia_intencao": dict(by_family_intent),
        "por_operacao": dict(by_operation),
        "temporal_count": temporal_count,
        "temporal_ratio": (temporal_count / len(data)) if data else 0.0,
        "entidades": dict(entity_counts),
        "missing_intents": missing_intents,
        "shortage_intents": shortage_intents,
        "missing_family_seed": missing_family[:100],
        "family_shortages": family_shortages,
        "shortage_schedule": {
            i: 1 for i in INTENT_MAP
            if not any(x.get("intent") == i and x.get("category") == "explicit_schedule" for x in data)
        },
        "generalization_collisions": generalization_collisions,
        "intent_boundary_tests_ok": boundary["ok"],
        "contrastive_tests_ok": contrast["ok"],
        "semantic_pairs": pair_report,
        "action_modes": dict(action_modes),
        "target_scopes": dict(target_scopes),
        "value_types": dict(value_types),
        "categorias": dict(categories),
        "categorias_por_intencao": dict(category_by_intent),
    }
    if cfg.audit_strict and not ok:
        raise ValueError("AUDITORIA DO DATASET FALHOU: " + json.dumps(report["problems"], ensure_ascii=False))
    return report


def coverage_report(data: List[Dict]) -> Dict:
    total = len(data)
    intents = Counter(s["intent"] for s in data)
    categories = Counter(s.get("category", "generative") for s in data)
    families = Counter(s.get("family", detect_linguistic_family(s["text"])) for s in data)
    operations = Counter(s.get("operation", "NONE") for s in data)
    temporal = Counter(any(e.get("type") in TEMPORAL_ENTITY_TYPES for e in s.get("entities", [])) for s in data)
    action_modes = Counter(s.get("action_mode") for s in data)
    target_scopes = Counter(s.get("target_scope") for s in data)
    value_types = Counter(s.get("value_type") for s in data)

    return {
        "total": total,
        "por_intencao": dict(intents),
        "por_categoria": dict(categories),
        "por_familia": dict(families),
        "por_operacao": dict(operations),
        "com_slot_temporal": {str(k): v for k, v in temporal.items()},
        "action_modes": dict(action_modes),
        "target_scopes": dict(target_scopes),
        "value_types": dict(value_types),
    }

def save_dataset(data: List[Dict], cfg: Config, report: Dict) -> Path:
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "dataset_compact.json"

    intent_map = dict(INTENT_MAP)
    entity_type_map = dict(ENTITY_TYPE_MAP)
    operation_map = dict(OPERATION_MAP)
    action_mode_map = dict(ACTION_MODE_MAP)
    target_scope_map = dict(TARGET_SCOPE_MAP)
    value_type_map = dict(VALUE_TYPE_MAP)

    compact = []
    for s in data:
        ents = []
        for e in s.get("entities", []):
            typ = str(e["type"])
            if typ not in entity_type_map:
                raise ValueError(f"Tipo de entidade desconhecido: {typ}")
            ents.append({
                "start": int(e["start"]),
                "end": int(e["end"]),
                "type": entity_type_map[typ],
                "value": e["value"]
            })
        compact.append({
            "text": s["text"],
            "intent": intent_map[str(s["intent"])],
            "operation": operation_map[str(s.get("operation", "NONE"))],
            "entities": ents,
            "action_mode": action_mode_map[s.get("action_mode", "IMMEDIATE")],
            "target_scope": target_scope_map[s.get("target_scope", "SINGLE")],
            "value_type": value_type_map[s.get("value_type", "UNKNOWN")]
        })

    payload = {
        "version": "63.20.0",
        "compact": True,
        "intent_map": intent_map,
        "entity_type_map": entity_type_map,
        "operation_map": operation_map,
        "action_mode_map": action_mode_map,
        "target_scope_map": target_scope_map,
        "value_type_map": value_type_map,
        "data": compact
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    (out_dir / "coverage_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Dataset compacto salvo em %s (%.2f MB)",
                path, path.stat().st_size / (1024 * 1024))
    return path

def run_intent_boundary_tests() -> Dict[str, Any]:
    results = []
    for phrase, expected in HARD_NEGATIVE_CASES:
        family = None
        low = phrase.casefold()
        if re.match(r"^(qual|como está|como esta|quanto está|quanto esta|verifica|consulte|me diga|status|estado|situação|situacao|cor da|cor do|nível da|nivel da)", low):
            family = "GET_STATUS"
        else:
            for intent, words in INTENT_LEXICAL_FAMILIES.items():
                if any(_contains_lexeme(low, w) for w in words):
                    family = intent
                    break
        results.append({"text": phrase, "expected": expected, "family": family,
                        "ok": family == expected})
    return {
        "ok": all(x["ok"] for x in results),
        "cases": results
    }

def run_contrastive_tests() -> Dict[str, Any]:
    cases=[
        ("liga a luz da sala", "TURN_ON"),
        ("desliga a luz da sala", "TURN_OFF"),
        ("inicia o ventilador da sala", "START"),
        ("para o ventilador da sala", "STOP"),
        ("abre a porta da sala", "OPEN"),
        ("fecha a porta da sala", "CLOSE"),
        ("esfria o ar condicionado da sala", "SET_TEMPERATURE"),
        ("qual a temperatura do ar condicionado da sala", "GET_STATUS"),
        ("aumenta a velocidade do ventilador da sala", "SET_SPEED"),
        ("qual a velocidade do ventilador da sala", "GET_STATUS"),
        ("aumenta o brilho da luz da sala", "SET_BRIGHTNESS"),
        ("qual o brilho da luz da sala", "GET_STATUS"),
        ("abaixa o volume da tv da sala", "SET_VOLUME"),
        ("qual o volume da tv da sala", "GET_STATUS"),
        ("sobe a voltagem da fonte de bancada", "SET_VOLTAGE"),
        ("qual a voltagem da fonte de bancada", "GET_STATUS"),
        ("coloca a luz da sala em vermelho", "SET_COLOR"),
        ("qual a cor da luz da sala", "GET_STATUS"),
        ("liga a luz da sala amanhã", "TURN_ON"),
        ("liga a luz da sala", "TURN_ON"),
        ("inicia o ventilador amanhã", "START"),
        ("inicia o ventilador", "START"),
        ("desliga a tv às 20:00", "TURN_OFF"),
        ("desliga a tv", "TURN_OFF"),
        ("fecha a porta todos os dias", "CLOSE"),
        ("fecha a porta", "CLOSE"),
        ("programa a luz para ligar amanhã", "TURN_ON"),
        ("agende a luz para desligar às 20:00", "TURN_OFF"),
        ("programe o ventilador para iniciar todos os dias", "START"),
    ]
    results=[]
    for phrase,expected in cases:
        results.append({"text":phrase,"expected":expected,
                        "accepted":intent_boundary_ok(phrase,expected)})
    return {"ok":all(x["accepted"] for x in results),"cases":results}

def run_semantic_firewall_tests() -> Dict[str, Any]:
    cases = [
        ("liga a luz da sala", "TURN_ON"),
        ("ativa a câmera da sala", "TURN_ON"),
        ("deixa a luz da sala ligada", "TURN_ON"),
        ("inicia a lavadora da lavanderia", "START"),
        ("dê partida na máquina da garagem", "START"),
        ("começa o ciclo da lavadora", "START"),
        ("desliga a luz da sala", "TURN_OFF"),
        ("desativa a tomada da sala", "TURN_OFF"),
        ("para o ciclo da lavadora", "STOP"),
        ("interrompe o aspirador da sala", "STOP"),
        ("pause a execução do aspirador", "STOP"),
        ("aumenta o brilho da luz da sala", "SET_BRIGHTNESS"),
        ("deixa a luz da sala mais forte", "SET_BRIGHTNESS"),
        ("coloque a luz da sala em vermelho", "SET_COLOR"),
        ("mude a cor da luz da sala para azul", "SET_COLOR"),
        ("cor da luz da sala: azul", "GET_STATUS"),
        ("qual a cor da luz da sala?", "GET_STATUS"),
    ]
    results = []
    for text, expected in cases:
        sample = {
            "text": text,
            "intent": expected,
            "operation": (
                "SET" if expected == "SET_COLOR" else
                "INCREASE" if expected == "SET_BRIGHTNESS" else "NONE"
            ),
            "entities": annotate(
                text,
                [("DEVICE", next((o for o in OBJECTS if o in text.casefold()), None)),
                 ("LOCATION", next((l for l in LOCATIONS if l in text.casefold()), None))]
                + ([("COLOR", next((c for c in COLORS if c in text.casefold()), None))]
                   if expected == "SET_COLOR" else [])
            ),
        }
        if expected == "SET_COLOR":
            sample["operation"] = "SET"
        ok = validate_sample(sample) if expected != "GET_STATUS" else intent_boundary_ok(text, expected)
        results.append({"text": text, "expected": expected, "ok": bool(ok)})
    return {"ok": all(x["ok"] for x in results), "cases": results}


def run_self_tests() -> Dict[str, Any]:
    results = []
    firewall = run_semantic_firewall_tests()
    if not firewall["ok"]:
        raise AssertionError(f"Falha no firewall semântico: {firewall}")
    results.append("semantic_firewall")
    for case in coordinated_device_examples():
        text_case = case["text"]
        entities = []
        for item in case["entities"]:
            pos = find_span(text_case, item["value"], [])
            if not pos:
                raise AssertionError(f"Entidade não encontrada: {item}")
            s, e = pos
            entities.append({"start": s, "end": e, "type": item["type"],
                             "value": text_case[s:e]})
        sample = {
            "text": text_case,
            "intent": "TURN_ON",
            "operation": "NONE",
            "entities": entities,
            "category": "test"
        }
        if not grammar_valid(text_case) or not validate_contextual_composition(text_case, entities):
            raise AssertionError(f"Falha de composição: {text_case}")
        results.append(text_case)
    for text_case, value, typ in [
        ("liga o soprador térmico da bancada", "soprador térmico", "DEVICE"),
        ("coloca a luz em azul escuro", "azul escuro", "COLOR"),
    ]:
        pos = find_span(text_case, value, [])
        if not pos:
            raise AssertionError(f"Composto não encontrado: {value}")
    temporal_cases = [
        ("liga a luz da sala amanhã", "TURN_ON"),
        ("liga a luz da sala às 18:00", "TURN_ON"),
        ("liga a luz da sala daqui a um minuto", "TURN_ON"),
        ("programe a luz da sala para desligar daqui a um minuto", "TURN_OFF"),
        ("inicia o ventilador todos os dias", "START"),
        ("fecha a porta às 20:00", "CLOSE"),
        ("ajuste a temperatura do ar condicionado para 22 graus às 18:00", "SET_TEMPERATURE"),
        ("coloque a luz da sala em vermelho amanhã", "SET_COLOR"),
        ("programa a luz para ligar amanhã", "TURN_ON"),
    ]
    for text_case, expected in temporal_cases:
        if not intent_boundary_ok(text_case, expected):
            raise AssertionError(f"Slot temporal rejeitado: {text_case}")
    mode_cases = [
        ("liga a luz da sala", "IMMEDIATE"),
        ("liga a luz da sala amanhã", "SCHEDULED"),
        ("liga a luz da sala às 18:00", "SCHEDULED"),
        ("liga a luz da sala daqui a um minuto", "SCHEDULED"),
        ("fecha a porta todos os dias", "RECURRING"),
    ]
    for text_case, expected_mode in mode_cases:
        ents = []
        for m in re.finditer(r"\bamanhã\b|\bàs?\s+\d{1,2}:\d{2}\b|\bdaqui a (?:um|uma|\d+) (?:minuto|minutos|hora|horas|dia|dias)\b|\btodos os dias\b", text_case, re.I):
            phrase = m.group(0)
            typ = "RECURRENCE" if phrase.casefold() == "todos os dias" else ("RELATIVE_TIME" if phrase.casefold().startswith("daqui a ") else ("DATE" if phrase.casefold() == "amanhã" else "TIME"))
            ents.append({"type": typ, "value": phrase, "start": m.start(), "end": m.end()})
        got = infer_action_mode(text_case, ents)
        if got != expected_mode:
            raise AssertionError(f"Action mode incorreto: {text_case}: esperado {expected_mode}, obtido {got}")
    return {"ok": True, "cases": results, "temporal_slot_tests": temporal_cases, "action_mode_tests": mode_cases}

def run(cfg: Config):
    validate_ontology_invariants()
    start = time.time()
    gen = DatasetGenerator(cfg)
    data = gen.generate_all()
    # As amostras já passam por validate_sample antes de entrarem no dataset
    # (inclusive após qualquer transformação temporal). Aqui fazemos somente
    # a checagem O(n) de unicidade; a auditoria abaixo executa a validação
    # semântica completa uma única vez, evitando uma segunda passagem pesada.
    problems = Counter()
    seen = set()
    for s in data:
        k = s["text"].casefold()
        if k in seen:
            problems["duplicate"] += 1
        seen.add(k)
    if problems:
        raise SystemExit(f"Dataset FALHOU na validação final: {dict(problems)}")
    audit = audit_generated_dataset(data, cfg)
    report = coverage_report(data)
    # Preserva no relatório os déficits informados pelo gerador, sem transformar
    # cobertura insuficiente em erro fatal.
    audit["generator_shortages"] = {
        "shortage_intent": dict(gen.audit.get("shortage_intent", {})),
        "shortage_family": dict(gen.audit.get("shortage_family", {})),
        "shortage_temporal": dict(gen.audit.get("shortage_temporal", {})),
        "shortage_schedule": dict(gen.audit.get("shortage_schedule", {})),
        "missing_family": dict(gen.audit.get("missing_family", {})),
        "partial_families": dict(gen.audit.get("partial_families", {})),
    }
    report["audit"] = audit
    path = save_dataset(data, cfg, report)
    logger.info("Concluído: %d amostras válidas | %.2fs | salvo em %s",
                len(data), time.time() - start, path)
    for intent, n in sorted(report["por_intencao"].items()):
        logger.info("  %-20s %d", intent, n)
    shortages = audit.get("shortage_intents", {})
    if shortages:
        logger.warning("Déficits de meta (não fatais): %s", shortages)
    family_shortages = audit.get("family_shortages", {})
    if family_shortages:
        logger.warning("Déficits de família (não fatais): %s", family_shortages)
    schedule_shortages = audit.get("shortage_schedule", {})
    if schedule_shortages:
        logger.warning("Déficits de programação explícita (não fatais): %s", schedule_shortages)
    return path, report

def main():
    ap = argparse.ArgumentParser(description="Gerador NLU V63.13 – Balanceamento circular, contraste temporal e robustez semântica")
    ap.add_argument("--samples-per-intent", type=int, default=600)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output-dir", type=str, default="./dataset_v63_8")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--smoke-test", action="store_true", help="gera 5 amostras por intenção para teste rápido")
    ap.add_argument("--samples-seed", type=int, default=1, help="sementes por família/intenção (mínimo)")
    ap.add_argument("--pair-fraction", type=float, default=0.35, help="fração inicial preenchida por pares contrastivos")
    ap.add_argument("--temporal-ratio", type=float, default=0.55, help="fração de exemplos com slot temporal explícito (padrão: 0.55)")
    ap.add_argument("--contrastive-boundary-ratio", type=float, default=0.18,
                    help="fração por intenção dedicada às fronteiras semânticas (padrão: 0.18)")
    ap.add_argument("--balance-strategy", choices=["equal", "natural"], default="equal",
                    help="equal: todas as famílias com mesmo número; natural: distribuição natural")
    args = ap.parse_args()
    if args.smoke_test:
        # Há 11 famílias linguísticas; menos de 11 não consegue satisfazer
        # a invariável de cobertura por intenção.
        args.samples_per_intent = max(11, args.samples_per_intent)
    if args.self_test:
        validate_ontology_invariants()
        print(json.dumps(run_self_tests(), ensure_ascii=False, indent=2))
        return
    cfg = Config(samples_per_intent=args.samples_per_intent,
                 random_seed=args.seed, output_dir=Path(args.output_dir),
                 seed_per_family=max(1, args.samples_seed),
                 pair_fraction=max(0.0, min(0.80, args.pair_fraction)),
                 temporal_ratio=max(0.0, min(1.0, args.temporal_ratio)),
                 contrastive_boundary_ratio=max(0.0, min(0.35, args.contrastive_boundary_ratio)),
                 balance_strategy=args.balance_strategy,
                 audit_strict=not args.smoke_test)
    run(cfg)

if __name__ == "__main__":
    main()